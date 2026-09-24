# This file is a part of TextSearch.jl

export save_lsi, load_lsi, QuantizedProjection, quantized_wordvectors

# ── the stored projection ────────────────────────────────────────────────────
#
# A fitted LSI is thrown away today: `fit_profile` uses `wordvectors(lsi)` for the expansion
# network and the lemma clusters, and nothing downstream can project anything new. Keeping it
# is what gives a consumer dense document vectors without an LLM, a BM25 + dense hybrid, and
# expansion at any neighbour count rather than the one frozen at fit time.
#
# It is NOT part of the profile, and that is the whole shape of this file. A profile is 3 MB
# where a projection is tens of megabytes, and most consumers never project anything; making
# every download pay for it would be the wrong default, and an optional member inside the
# profile would mean one artifact that is sometimes half-present. So the projection is its own
# artifact, and it NAMES the profile it belongs to rather than the other way round.
#
# What makes that safe is `profile_id`. The columns are indexed by vocabulary id, so against
# the wrong profile this does not fail -- every column is off and every answer is quietly
# wrong. `load_lsi` recomputes the id of the profile it is handed and refuses a mismatch,
# naming what the artifact expects. It never downloads: fetching hundreds of megabytes as a
# side effect of opening a file is too much to do implicitly, so a missing profile is an error
# that says which one, and the fetching is the caller's (or `textsearch`'s) to do.
#
# Stored as SQu8 -- per-column scalar quantization, one UInt8 per coordinate plus a (min,
# scale) pair per column. Per column rather than over the whole matrix, and that is measured:
# over 4,273 LSI columns a single global range gives a round-trip cosine of 0.850 mean / 0.636
# min between a column and itself, which wrecks every token-level use (the expansion network,
# lemma extension, an OOV backoff) while still looking fine on document retrieval, where the
# error averages out over a sum of many columns. Per column it is 0.99996 mean / 0.9998 min,
# for 8 extra bytes per column, and document retrieval improves too (top-10 overlap against
# the Float32 model, 0.971 -> 0.995). u8 at all costs a quarter of Float32.

"""
    QuantizedProjection <: AbstractMatrix{Float32}

An LSI projection held in its quantized form, dequantizing one coordinate at a time on access:
`codes[i, j] * scales[j] + mins[j]`.

This exists so the saving is not undone at load. Dequantizing into a `Matrix{Float32}` would
cost 4x the memory the quantization was chosen to avoid -- 187 MB against 47 MB for a
`64 x 730,320` projection -- and nothing in the LSI path needs a dense matrix:
`_project_sparse!` reads `P[row, t]` element by element, and `wordvectors` copies into a fresh
matrix of its own. `LatentSemanticIndexing` accepts it because its parameter asks only for an
`AbstractMatrix{Float32}`.

The same codes can also be handed to `SimilaritySearch`'s `SQu8Database`, which is an
`AbstractDatabase` with distances defined on the quantized form, for a search pipeline that
never dequantizes at all.
"""
struct QuantizedProjection <: AbstractMatrix{Float32}
    codes::Matrix{UInt8}
    mins::Vector{Float32}
    scales::Vector{Float32}

    function QuantizedProjection(codes::Matrix{UInt8}, mins::Vector{Float32}, scales::Vector{Float32})
        length(mins) == length(scales) == size(codes, 2) ||
            throw(DimensionMismatch(
                "QuantizedProjection: $(length(mins)) min(s) and $(length(scales)) scale(s) " *
                "for $(size(codes, 2)) column(s); there must be one of each per column"))
        new(codes, mins, scales)
    end
end

Base.size(P::QuantizedProjection) = size(P.codes)
Base.@propagate_inbounds Base.getindex(P::QuantizedProjection, i::Integer, j::Integer)::Float32 =
    Float32(P.codes[i, j]) * P.scales[j] + P.mins[j]

"""
    _quantize_columns(P::AbstractMatrix{Float32}) -> (codes, mins, scales)

Quantizes each column over its own extrema, the same way `SimilaritySearch`'s `SQu8` does, so
the stored codes can be read back either as a [`QuantizedProjection`](@ref) or as an
`SQu8Database` without being re-quantized into a different thing.
"""
function _quantize_columns(P::AbstractMatrix{Float32})
    k, n = size(P)
    codes = Matrix{UInt8}(undef, k, n)
    mins = Vector{Float32}(undef, n)
    scales = Vector{Float32}(undef, n)

    for j in 1:n
        col = view(P, :, j)
        lo, hi = extrema(col)
        # the epsilon matches SQu8's, so a column whose values are all equal still round trips
        # to itself rather than dividing by zero
        c = (hi - lo + 1f-6) / 255f0
        mins[j] = lo
        scales[j] = c
        inv = 1f0 / c
        @inbounds for i in 1:k
            codes[i, j] = round(UInt8, clamp((col[i] - lo) * inv, 0f0, 255f0))
        end
    end

    codes, mins, scales
end

const _LSI_FORMAT_VERSION = "1.0"
const _LSI_MANIFEST_NAME = "manifest.json"

"""
    save_lsi(dir, lsi::LatentSemanticIndexing, profile::TextProfile; name="", repo="", tag="")

Writes `lsi`'s projection into `dir` as its own artifact, bound to `profile`.

Only the projection is written: the singular values, the quantized matrix, and enough of the
manifest to find and verify the profile. Everything else an LSI needs -- the vocabulary, the
weights, the tokenizer -- is the profile's, and duplicating it is what this layout exists to
avoid.

`name`/`repo`/`tag` record where the profile can be fetched from, for a person or a tool to
act on; nothing here fetches anything. What makes the binding safe is not that reference but
[`profile_id`](@ref), recorded beside it and checked by [`load_lsi`](@ref).

Store one artifact at the largest `outdim` worth keeping: truncating a truncated SVD is exact,
so `load_lsi(...; outdim)` serves every smaller one from the same file.
"""
function save_lsi(dir::AbstractString, lsi::LatentSemanticIndexing, profile::TextProfile;
                  name::AbstractString="", repo::AbstractString="", tag::AbstractString="")
    lsi.model.voc.token == profile.model.voc.token ||
        error("save_lsi: this LSI was fitted over a different vocabulary than the profile it " *
              "is being bound to ($(vocsize(lsi.model)) token(s) against " *
              "$(vocsize(profile.model))). A projection's columns are vocabulary ids, so the " *
              "two have to be the same vocabulary, not merely similar ones.")

    mkpath(dir)
    codes, mins, scales = _quantize_columns(lsi.P)

    manifest = Dict{String,Any}(
        "format_version" => _LSI_FORMAT_VERSION,
        "profile" => Dict{String,Any}(
            "id" => profile_id(profile),
            "name" => String(name), "repo" => String(repo), "tag" => String(tag)),
        "outdim" => outdim(lsi),
        "maxoutdim" => lsi.maxoutdim,
        "scaling" => String(lsi.scaling),
        "codes" => _save_array(dir, "projection_codes.bin", codes),
        "mins" => _save_array(dir, "projection_mins.bin", mins),
        "scales" => _save_array(dir, "projection_scales.bin", scales),
        "singular_values" => _save_array(dir, "singular_values.bin", lsi.s),
    )

    _write_json(joinpath(dir, _LSI_MANIFEST_NAME), manifest)
    dir
end

"""
    load_lsi(path, profile::TextProfile; outdim=nothing) -> LatentSemanticIndexing

Reads the artifact at `path` (a directory or a zip) and rebuilds an LSI over `profile`.

`profile` is not fetched: a projection is bound to one profile and no other, so if the wrong
one is handed in this errors naming the one the artifact was fitted against. Getting that
profile is the caller's job -- see [`download_profile`](@ref) -- because downloading hundreds
of megabytes as a side effect of opening a file is not something this should decide.

`outdim` truncates: it is the number of LSI components to keep, the same quantity
[`outdim`](@ref) reports and `maxoutdim` requests -- not the neighbour count that
[`query_expansion`](@ref) calls `k`, which is a different number entirely. Truncating a
truncated SVD is exact, since the singular values come out ordered and the quantization is per
column, so the coordinates read out of a larger artifact are exactly the first ones it stores.
"""
function load_lsi(path::AbstractString, profile::TextProfile;
                  outdim::Union{Nothing,Integer}=nothing)
    read_bytes = _profile_reader(path)
    manifest = JSON3.read(read_bytes(_LSI_MANIFEST_NAME))

    version = String(get(manifest, :format_version, "(missing)"))
    version == _LSI_FORMAT_VERSION ||
        error("unsupported LSI artifact format_version: $version (this build reads " *
              "$_LSI_FORMAT_VERSION only). Refit it.")

    ref = manifest[:profile]
    want = String(ref[:id])
    got = profile_id(profile)
    if want != got
        named = isempty(String(ref[:name])) ? "" : " '$(String(ref[:name]))'"
        where = isempty(String(ref[:repo])) ? "" :
            " (from $(String(ref[:repo]))$(isempty(String(ref[:tag])) ? "" : " at $(String(ref[:tag]))"))"
        error("this LSI artifact was fitted against profile$named id=$want$where, and the " *
              "profile it was given has id=$got. A projection's columns are vocabulary ids, " *
              "so the wrong profile would not fail -- it would answer wrongly. Load the " *
              "profile it names.")
    end

    codes = _load_array(read_bytes, manifest[:codes])
    mins = Vector{Float32}(_load_array(read_bytes, manifest[:mins]))
    scales = Vector{Float32}(_load_array(read_bytes, manifest[:scales]))
    svals = Vector{Float32}(_load_array(read_bytes, manifest[:singular_values]))

    stored = Int(manifest[:outdim])
    size(codes, 1) == stored ||
        error("the LSI artifact says outdim=$stored and its codes have $(size(codes, 1)) row(s)")
    size(codes, 2) == vocsize(profile.model) ||
        error("the LSI artifact has $(size(codes, 2)) column(s) and the profile holds " *
              "$(vocsize(profile.model)) token(s); they must agree")

    want = outdim === nothing ? stored : Int(outdim)
    0 < want <= stored ||
        throw(ArgumentError("outdim must be in 1:$stored for this artifact, got $want"))
    if want < stored
        codes = codes[1:want, :]
        svals = svals[1:want]
    end

    P = QuantizedProjection(codes, mins, scales)
    LatentSemanticIndexing(profile.model, P, svals, want, Int(manifest[:maxoutdim]),
                           Symbol(String(manifest[:scaling])))
end


# ── the token embeddings, still quantized ────────────────────────────────────
#
# `wordvectors(lsi)` materializes a `(outdim, vocsize)` Float32 matrix and normalizes its
# columns, which is what a token-neighbour search needs and what `query_expansion` already
# uses. For a stored projection that copy is the thing this format exists to avoid: 187 MB for
# a 64 x 730,320 model, to hold what 47 MB of codes already say.
#
# Handing the STORED codes straight to `SQu8.NormCosine` is the obvious move and it is wrong,
# quietly. That distance is `1 - dot`, which is a cosine only for unit vectors -- its docstring
# says so -- and LSI columns are not unit vectors. Measured over 4,273 columns against the
# dense normalized answer:
#
#   raw codes + NormCosine        top-10 overlap 0.2526 (outdim 256) / 0.0952 (64)
#   raw codes + SqL2                             0.3419              / 0.3917
#   normalized + NormCosine                      0.9937              / 0.9913
#
# and a token is its own nearest neighbour only 7.1% / 1.7% of the time in the first row. That
# is not a degraded answer, it is a different function: with magnitudes spanning orders of
# magnitude, `1 - dot` ranks by length rather than by direction. Note the trap in the obvious
# diagnostic too -- `SqL2` puts every token first by construction (`d(x, x) = 0`) while still
# disagreeing with cosine on two thirds of the list.
#
# What makes the fix cheap is that per-column quantization is SCALE INVARIANT: it maps a
# column's own `[min, max]` onto `0:255`, and scaling the column scales both ends, so `v` and
# `v/‖v‖` quantize to the SAME codes. The whole difference lives in the `(min, scale)` pair.
# So this divides those by the column's norm and reuses the codes untouched -- no second
# quantization, no temporary matrix -- and measures 0.9939 / 0.9918, a shade ABOVE
# re-quantizing, because it avoids a second rounding.
#
# That is also why the artifact stores the raw projection and not the normalized one: raw to
# normalized is this parameter division, while normalized to raw would need the norms kept
# somewhere as well. The derivation only runs one way.

"""
    quantized_wordvectors(lsi::LatentSemanticIndexing) -> SQu8Database

The LSI embedding of every vocabulary token, unit-normalized and quantized, as a database that
`SimilaritySearch` can search directly -- with `ScalarQuant.SQu8.NormCosine()`, which is the
distance those normalized codes are for.

The dense counterpart is [`wordvectors`](@ref), which returns the same embeddings as a
`Float32` matrix. Prefer this one when the projection came from [`load_lsi`](@ref): it reuses
the stored codes rather than expanding them, so it costs the column norms and nothing else.

Do not reach past this and hand `SQu8.NormCosine` the codes a [`QuantizedProjection`](@ref)
holds. They are the *raw* projection's codes, that distance assumes unit vectors, and the
result is wrong in a way that looks plausible -- a top-10 overlap of 0.25 against the dense
answer, with a token its own nearest neighbour 7% of the time.
"""
function quantized_wordvectors(lsi::LatentSemanticIndexing{<:QuantizedProjection})
    P = lsi.P
    k, n = size(P)
    E = Vector{SQMinC}(undef, n)

    @inbounds for j in 1:n
        acc = 0f0
        for i in 1:k
            v = P[i, j]
            acc += v * v
        end
        nrm = sqrt(acc)
        # a column of zeros has no direction to normalize; leaving it be keeps it at the origin
        E[j] = nrm > 0f0 ? SQMinC(P.mins[j] / nrm, P.scales[j] / nrm) :
                           SQMinC(P.mins[j], P.scales[j])
    end

    SQu8Database(E, P.codes)
end

function quantized_wordvectors(lsi::LatentSemanticIndexing)
    # a dense projection has no codes to reuse, so this is the ordinary route: normalize (which
    # `wordvectors` already does) and quantize once
    SQu8.quantize(wordvectors(lsi).matrix)
end
