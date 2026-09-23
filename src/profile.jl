# This file is a part of TextSearch.jl

export save_profile, load_profile, zip_profile, download_profile, list_remote_profiles,
       profile_id

# Bumped to "1.1" for the expansion network's move to ids (see `_save_expansion`).
#
# The freeze at "1.0" was justified by there being no released consumers -- "nothing is
# published and no profile exists outside this repository" -- and that stopped being true at
# v1.1.0, which ships seven profiles as release assets and is what `textsearch download`
# fetches. So the version starts doing the job it was always shaped for: a file from before
# this is refused by version, with one sentence saying so, instead of being caught further in
# by an ad-hoc check on some key that happens to have moved.
#
# Under "1.1", also for the record: the network's move to ids, and the `id` field that names
# what a profile does to text (see `profile_id`). The field went in without a further bump for
# the same reason the freeze at "1.0" was right for as long as it was -- nothing is published
# at "1.1" yet, so there is no file out there for the version to protect.
#
# What changed under "1.0", for the record: the policy/artifacts split, the rename of the
# artifact from "synonyms" to "query_expansion" (manifest key and both files), and the
# distances becoming a quantized binary member. That rename mattered beyond tidiness -- calling
# it a synonym network invited judging its entries as substitutable words, and a long stretch
# of work went into filtering it on that basis before the objective was restated: the artifact
# exists to enrich a query, and topically related terms are what serve that. See the note in
# `src/lsi.jl`.
const _PROFILE_FORMAT_VERSION = "1.1"
const _PROFILE_MANIFEST_NAME = "manifest.json"

# ── weighting tag tables ─────────────────────────────────────────────────────

const _GLOBAL_WEIGHTING_TAG = Dict{DataType,String}(
    IdfWeighting => "idf",
    BinaryGlobalWeighting => "binary_global",
    EntropyWeighting => "entropy",
)
const _LOCAL_WEIGHTING_TAG = Dict{DataType,String}(
    TfWeighting => "tf",
    TpWeighting => "tp",
    FreqWeighting => "freq",
    BinaryLocalWeighting => "binary_local",
)
const _TAG_GLOBAL_WEIGHTING = Dict(v => k for (k, v) in _GLOBAL_WEIGHTING_TAG)
const _TAG_LOCAL_WEIGHTING = Dict(v => k for (k, v) in _LOCAL_WEIGHTING_TAG)

function _encode_global_weighting(gw)
    haskey(_GLOBAL_WEIGHTING_TAG, typeof(gw)) ||
        error("cannot serialize global_weighting of type $(typeof(gw)) into a profile")
    _GLOBAL_WEIGHTING_TAG[typeof(gw)]
end

function _encode_local_weighting(lw)
    haskey(_LOCAL_WEIGHTING_TAG, typeof(lw)) ||
        error("cannot serialize local_weighting of type $(typeof(lw)) into a profile")
    _LOCAL_WEIGHTING_TAG[typeof(lw)]
end

function _decode_global_weighting(tag::AbstractString)
    haskey(_TAG_GLOBAL_WEIGHTING, tag) || error("unknown global_weighting tag: $tag")
    _TAG_GLOBAL_WEIGHTING[tag]()
end

function _decode_local_weighting(tag::AbstractString)
    haskey(_TAG_LOCAL_WEIGHTING, tag) || error("unknown local_weighting tag: $tag")
    _TAG_LOCAL_WEIGHTING[tag]()
end

# ── policy (normalization / tokenization) ────────────────────────────────────
#
# Both halves are small -- a handful of flags, three regex patterns, the emoji set -- so they
# stay inlined in the manifest. There is no transformation to encode: a profile's
# transformation is *derived* from its artifacts by `textconfig`, so serializing it would be
# storing the same stopword set and lemma map a second time. That second copy is exactly what
# used to drift out of sync with the first.

function _encode_normalization(n::NormalizationConfig)
    Dict(
        "del_diac" => n.del_diac, "del_dup" => n.del_dup, "del_punc" => n.del_punc,
        "group_num" => n.group_num, "group_url" => n.group_url, "group_usr" => n.group_usr,
        "group_emo" => n.group_emo, "lc" => n.lc,
        "re_user" => n.re_user.pattern, "re_url" => n.re_url.pattern, "re_num" => n.re_num.pattern,
        # sorted because the source is a `Set{Char}`, whose iteration order is arbitrary:
        # unsorted, two saves of one profile produced manifests that differed in this array
        # alone, which defeats diffing them and makes `profile_id` unstable
        "emojis" => [string(c) for c in sort!(collect(n.emojis))],
    )
end

function _decode_normalization(d)
    NormalizationConfig(;
        del_diac=Bool(d[:del_diac]), del_dup=Bool(d[:del_dup]), del_punc=Bool(d[:del_punc]),
        group_num=Bool(d[:group_num]), group_url=Bool(d[:group_url]), group_usr=Bool(d[:group_usr]),
        group_emo=Bool(d[:group_emo]), lc=Bool(d[:lc]),
        re_user=Regex(String(d[:re_user])), re_url=Regex(String(d[:re_url])), re_num=Regex(String(d[:re_num])),
        emojis=Set{Char}(only(String(s)) for s in d[:emojis]),
    )
end

function _encode_tokenization(t::TokenizationConfig)
    isempty(t.generators) ||
        error("cannot serialize a TokenizationConfig with custom (non-empty) generators into a profile")
    Dict("nlist" => Int.(t.nlist), "mark_token_type" => t.mark_token_type)
end

_decode_tokenization(d) = TokenizationConfig(nlist=Int8.(d[:nlist]), mark_token_type=Bool(d[:mark_token_type]))

function _encode_policy(tc::TextConfig)
    Dict("normalization" => _encode_normalization(tc.normalization),
         "tokenization" => _encode_tokenization(tc.tokenization),
         "language" => String(tc.language))
end

_decode_policy(d) = TextConfig(normalization=_decode_normalization(d[:normalization]),
                               tokenization=_decode_tokenization(d[:tokenization]),
                               language=Symbol(get(d, :language, "unknown")))

# ── lineage ──────────────────────────────────────────────────────────────────

_encode_lineage(l::AbstractVector{LineageStep}) =
    [Dict("stage" => String(s.stage), "params" => s.params) for s in l]

_decode_lineage(d) =
    LineageStep[LineageStep(Symbol(s[:stage]),
                            Dict{String,Any}(String(k) => v for (k, v) in pairs(s[:params])))
                for s in d]

# ── file-backed I/O helpers (directory or zip, symmetrically) ──────────────

_write_json(path::AbstractString, data) = open(io -> JSON3.write(io, data), path, "w")

"""
    _profile_reader(path::AbstractString) -> read_bytes::Function

Returns a `read_bytes(name::AbstractString) -> Vector{UInt8}` closure that fetches a named
member of the profile at `path` -- a plain directory if `isdir(path)`, otherwise a `.zip`
archive (opened once and re-read from memory for every subsequent call). This is what lets
[`load_profile`](@ref) not care which of the two forms it was handed.

It hands back **bytes** rather than parsed JSON so that one closure serves both kinds of member:
`load_profile` wraps it in `JSON3.read` for the text ones and passes it straight to
[`_load_array`](@ref) for the binary ones. Parsing here would have meant a second, parallel
reader for the binary path -- and two readers that must agree about a directory-versus-zip
distinction is exactly the kind of duplicated knowledge this format has been bitten by before.
"""
function _profile_reader(path::AbstractString)
    if isdir(path)
        name -> read(joinpath(path, name))
    else
        buf = read(path)
        zr = ZipArchives.ZipReader(buf)
        name -> ZipArchives.zip_readentry(zr, name)
    end
end

# ── profile identity ─────────────────────────────────────────────────────────
#
# An id for WHAT A PROFILE DOES TO TEXT, so an artifact fitted against one can say which one
# and be checked. The case that needs it is a stored LSI projection: its columns are indexed
# by vocabulary id, so pointed at the wrong profile it does not fail -- every column is off by
# some amount and every answer is quietly wrong.
#
# What goes in is chosen by that job, not by "what is in the file":
#
#   in   the policy, and the artifacts the pipeline APPLIES -- they decide which tokens a text
#        yields at all
#   in   the token sequence in id order -- it IS the column index of anything fitted over the
#        vocabulary
#   in   the counters and the weighting scheme -- they decide the coordinates, and the weight
#        vector follows from them by recomputation, so it needs no separate contribution
#   out  the expansion network, the artifacts that are carried but not applied, the lineage --
#        a profile can gain or lose any of them and still turn text into the same vector
#
# The consequences are the point: a profile RE-SAVED keeps its id, and one whose lineage note
# or expansion network changed keeps it too. An id that moved every time the file was rewritten
# would be no use to anything.
#
# This is deliberately NOT an integrity check. A profile travels as a zip, and ZipArchives
# verifies the CRC32 of every entry it reads, so damage in transit is already caught by the
# container -- and covering the whole file here would mean rehashing megabytes on every load to
# re-prove what the reader just proved. `save_profile` records the id so a tool can read it
# without parsing a vocabulary; anything that must not be fooled recomputes it with
# `profile_id`, which cannot be edited.

"""
    profile_id(p::TextProfile) -> String

A 16-hex-character identifier for what `p` does to text: its policy, the artifacts it applies,
its token sequence, its counters and its weighting. Two profiles share an id exactly when they
turn any text into the same vector, so an artifact fitted against one -- a stored dense
projection, say -- can record the id and refuse anything else.

Derived rather than stored, so it cannot disagree with the profile it describes.
[`save_profile`](@ref) also writes it to the manifest as `id`, for tools that want it without
loading anything, and that copy is a convenience: it is not what a check should read.
"""
function profile_id(p::TextProfile)
    ctx = SHA2_256_CTX()
    txt(s) = update!(ctx, codeunits(string(s)))
    field(k, v) = (txt(k); txt("\x1f"); txt(v); txt("\x1e"))

    # the scheme itself is versioned: changing what goes in must change every id
    txt("textsearch-profile-id/1\x1e")

    _hash_canonical(ctx, _encode_policy(getpolicy(p)))
    field("applied.stopwords", p.applied.stopwords)
    field("applied.lemmas", p.applied.lemmas)
    if p.applied.stopwords
        for w in sort!(collect(p.stopwords)); field("stopword", w); end
    end
    if p.applied.lemmas
        for k in sort!(collect(keys(p.lemmas))); field(k, p.lemmas[k]); end
    end

    model = p.model
    field("global_weighting", _encode_global_weighting(model.global_weighting))
    field("local_weighting", _encode_local_weighting(model.local_weighting))

    voc = model.voc
    field("trainsize", gettrainsize(voc))
    field("numtokens", getnumtokens(voc))
    field("vocsize", vocsize(voc))
    for id in eachindex(voc); txt(gettoken(voc, id)); txt("\x1f"); end
    txt("\x1e")
    _hash_le(ctx, voc.occs)
    _hash_le(ctx, voc.ndocs)

    bytes2hex(digest!(ctx))[1:16]
end

"""
    _hash_canonical(ctx, v)

Folds `v` into `ctx` in an order that does not depend on how a `Dict` happens to iterate --
which in Julia is not a stable order, and would otherwise make an id differ between two runs
over the same profile.
"""
function _hash_canonical(ctx, v)
    if v isa AbstractDict
        for k in sort!(collect(keys(v)); by=string)
            update!(ctx, codeunits(string(k))); update!(ctx, codeunits("\x1f"))
            _hash_canonical(ctx, v[k])
        end
    elseif v isa AbstractVector || v isa AbstractSet
        for x in v; _hash_canonical(ctx, x); end
    else
        update!(ctx, codeunits(string(v)))
    end
    update!(ctx, codeunits("\x1e"))
end

"""
    _hash_le(ctx, A)

Folds a numeric array into `ctx` in **little-endian** order, for the same reason
`arraystore.jl` writes its members that way: a profile is published and read back by whoever,
and an id that depended on the host's byte order would not survive the trip.
"""
function _hash_le(ctx, A::AbstractArray{T}) where {T}
    if _islittle()
        update!(ctx, reinterpret(UInt8, A))
    else
        buf = Vector{UInt8}(undef, sizeof(T))
        for v in A
            u = htol(v)
            unsafe_copyto!(pointer(buf), Ptr{UInt8}(pointer_from_objref(Ref(u))), sizeof(T))
            update!(ctx, buf)
        end
    end
end

# ── save_profile / load_profile / zip_profile ────────────────────────────────

"""
    save_profile(dir::AbstractString, p::TextProfile) -> dir

Serializes a [`TextProfile`](@ref) into `dir` (created if missing) as a small directory of
plain, human-readable JSON files: one per "large" piece -- `vocabulary.json`, `weights.json`,
and `stopwords.json`/`lemmas.json` for whichever artifacts are non-empty -- tied together by a
`manifest.json` holding everything else.

The exception is the **expansion network**, which is three binary members over vocabulary ids
(`query_expansion_counts.bin`, `query_expansion_neighbors.bin` and, when the profile carries
them, `u8`-quantized `query_expansion_distances.bin`). Both departures from text are measured
rather than assumed. On the published Spanish profile the network and its distances were
141,088,462 bytes of JSON -- 86% of the whole file -- against 28,680,775 as ids, because every
neighbour string was already in `vocabulary.json` and was being stored again in every list that
named it. The quantization is retrieval-identical (`arraystore.jl` has those numbers). Every
other file stays text, and a reader who wants tokens joins against the vocabulary, which is the
same join the loader does.

The manifest keeps policy and artifacts apart, which is the point of the layout:

```
id:         "914ca66f4ddd5767"          # see `profile_id`
policy:     { normalization: {...}, tokenization: {...} }
artifacts:  { stopwords: {file, applied}, lemmas: {file, applied},
              query_expansion: {applied, layout: "csc",
                                counts:    {file, dtype, shape},
                                neighbors: {file, dtype, shape},
                                distances: {file, dtype, shape, quant, columns}} }
lineage:    [ {stage, params}, ... ]
```

Each artifact is named **once**, with the marker saying whether the profile applies it. The
token transformation is not serialized at all: it is derived from these on load, so the
applied lemma map cannot differ from the saved one.

Deliberately NOT a generic object-graph dump (unlike e.g. JLD2): every field is encoded by
hand into a small, versioned schema, so every file is fully inspectable/diffable/portable and
there is nothing pointer- or code-shaped to accidentally serialize.

Load it back with [`load_profile`](@ref), or package it for distribution with
[`zip_profile`](@ref). A `TokenizationConfig` with custom (non-empty) `generators` errors
clearly rather than silently mis-saving.
"""
function save_profile(dir::AbstractString, p::TextProfile)
    mkpath(dir)
    voc = p.model.voc
    model = p.model

    _write_json(joinpath(dir, "vocabulary.json"), Dict(
        "tokens" => voc.token,
        "occs" => voc.occs,
        "ndocs" => voc.ndocs,
        "trainsize" => voc.trainsize[],
        "numtokens" => voc.numtokens[],
    ))

    _write_json(joinpath(dir, "weights.json"), Dict("weight" => model.weight))

    artifacts = Dict{String,Any}()

    if !isempty(p.stopwords)
        _write_json(joinpath(dir, "stopwords.json"), sort!(collect(p.stopwords)))
        artifacts["stopwords"] = Dict("file" => "stopwords.json", "applied" => p.applied.stopwords)
    end

    if !isempty(p.lemmas)
        _write_json(joinpath(dir, "lemmas.json"), p.lemmas)
        artifacts["lemmas"] = Dict("file" => "lemmas.json", "applied" => p.applied.lemmas)
    end

    if !isempty(p.query_expansion)
        artifacts["query_expansion"] = _save_expansion(dir, p)
    end

    _write_json(joinpath(dir, _PROFILE_MANIFEST_NAME), Dict(
        "format_version" => _PROFILE_FORMAT_VERSION,
        # Recorded so a tool can read it without parsing a vocabulary. It is a convenience
        # copy: a check that must not be fooled calls `profile_id` on what it loaded.
        "id" => profile_id(p),
        "policy" => _encode_policy(getpolicy(p)),
        "artifacts" => artifacts,
        "vocabulary_file" => "vocabulary.json",
        "weighting" => Dict(
            "global_weighting" => _encode_global_weighting(model.global_weighting),
            "local_weighting" => _encode_local_weighting(model.local_weighting),
            "maxoccs" => model.maxoccs,
            "weight_file" => "weights.json",
        ),
        "lineage" => _encode_lineage(p.lineage),
    ))

    dir
end


# ── the expansion network, as CSC over vocabulary ids ────────────────────────
#
# The network used to be a `Dict{String,Vector{String}}` in JSON, which stored every neighbour
# as a full token string in every list it appears in, plus the keys. `vocabulary.json` already
# holds all of those strings once, so the file was, byte for byte, mostly a permuted second copy
# of the vocabulary. Measured on the published Spanish profile (vocsize 730,320, 5,590,091
# edges, k = 8):
#
#   query_expansion.json             70,933,553 B stored     33,744,114 B deflated
#   query_expansion_distances.json   70,154,909 B            31,351,092 B
#   the three arrays below           28,680,775 B            20,741,393 B   (-80% / -68%)
#
# and the whole profile goes 163,945,326 -> 28,651,426, i.e. 17% of what shipped.
#
# The layout is CSC with the SOURCE TOKEN AS THE COLUMN. That orientation is forced rather than
# chosen: every consumer asks "give me this token's neighbours", so the source has to be the
# contiguous axis, which in column-major Julia means the column -- and it matches `lsi.jl`'s
# term = row, item = column convention.
#
# Entries within a column stay in RANK ORDER, which is why this is deliberately not handed out
# as a `SparseMatrixCSC`: that type requires `rowval` ascending within a column and does not
# check it, so an unsorted one misbehaves silently rather than erroring. Sorting by id instead
# would be worth 2.8% of the bytes and would cost the ranking, because rank cannot be recovered
# from the score once it is quantized -- measured, the u8 step leaves 57% of columns with ties
# against 1.7% at Float32.
#
# The column axis is stored as PER-COLUMN COUNTS, not offsets, and the loader cumulates them:
# 115,248 deflated bytes against 1,037,000, because counts are a short run-heavy alphabet (here
# {0, 8} almost everywhere) while offsets are 730,321 distinct rising integers. The dtype is
# whatever fits -- `u8` while no token has more than 255 neighbours, `u32` beyond that -- and
# `arraystore.jl` already records dtype per member, so this needs no new format concept and
# imposes no cap.
#
# The one thing this layout cannot express is a token that is not in the vocabulary, and
# `save_profile` refuses rather than dropping it. Every path the library itself has already
# guarantees the property -- `_restrict_query_expansion` enforces it on a refit, `fit_profile`
# derives the network from the vocabulary, and the published merged profile measured 100% on
# both keys and neighbours -- and an entry naming a non-vocabulary token could never have
# matched anything anyway, since the query path skips a token whose id is 0.

"""
    _save_expansion(dir, p::TextProfile) -> Dict{String,Any}

Writes the expansion network under `dir` as three binary members over vocabulary ids -- column
counts, concatenated neighbour ids in rank order, and optionally their quantized distances --
and returns the manifest entry describing them.

Errors if the network names a token the vocabulary lacks; see the note above for why that is a
refusal rather than a silent drop.
"""
function _save_expansion(dir::AbstractString, p::TextProfile)
    voc = p.model.voc
    n = vocsize(voc)
    counts = zeros(UInt32, n)

    examples = String[]
    nbad = 0
    note(t) = (nbad += 1; length(examples) < 5 && push!(examples, repr(t)); nothing)

    for (tok, neighbors) in p.query_expansion
        id = token2id(voc, tok)
        id == 0 && (note(tok); continue)
        kept = 0
        for s in neighbors
            token2id(voc, s) == 0 ? note(s) : (kept += 1)
        end
        counts[id] = kept
    end

    nbad == 0 || error(
        "save_profile: the query_expansion network names $nbad token(s) that are not in the " *
        "vocabulary (e.g. $(join(examples, ", "))). The network is stored as ids over the " *
        "vocabulary, so such an entry cannot be written -- and it could never have matched " *
        "anything either, since the query path skips a token whose id is 0. Restrict the " *
        "network to the vocabulary first; `refit_profile` does that on its own.")

    nnz = Int(sum(counts))
    neighbors_flat = Vector{UInt32}(undef, nnz)
    dvalues = Float32[]
    dcolumns = UInt32[]
    ncolumns = 0

    at = 1
    for id in 1:n
        counts[id] == 0 && continue
        ncolumns += 1
        tok = gettoken(voc, id)
        neighbors = p.query_expansion[tok]
        for s in neighbors
            j = token2id(voc, s)
            j == 0 && continue
            neighbors_flat[at] = j
            at += 1
        end
        # All or nothing per column: a distance list that does not cover its own ranking could
        # not be lined up against it on the way back in.
        ds = p.query_expansion_distances === nothing ? nothing :
            get(p.query_expansion_distances, tok, nothing)
        if ds !== nothing && length(ds) == Int(counts[id])
            push!(dcolumns, UInt32(id))
            append!(dvalues, ds)
        end
    end

    # u8 while it fits, which is every profile this has been run on; the dtype travels in the
    # manifest, so widening is a data question and not a format change
    counts_stored = maximum(counts; init=UInt32(0)) <= 255 ? UInt8.(counts) : counts
    entry = Dict{String,Any}(
        "applied" => p.applied.query_expansion,
        "layout" => "csc",
        "counts" => _save_array(dir, "query_expansion_counts.bin", counts_stored),
        "neighbors" => _save_array(dir, "query_expansion_neighbors.bin", neighbors_flat),
    )

    if !isempty(dvalues)
        d = _save_array(dir, "query_expansion_distances.bin", dvalues; quantize=true)
        d["columns"] = length(dcolumns) == ncolumns ? "all" :
            _save_array(dir, "query_expansion_distance_columns.bin", dcolumns)
        entry["distances"] = d
    end

    entry
end

"""
    _load_expansion(read_bytes, entry, voc) -> (query_expansion, distances)

Rebuilds the network from the members [`_save_expansion`](@ref) wrote, turning ids back into the
tokens `voc` holds. `distances` is `nothing` when the profile carries only the ranking.
"""
function _load_expansion(read_bytes, entry, voc::Vocabulary)
    String(_entry(entry, :layout, "csc")) == "csc" ||
        error("unknown query_expansion layout: $(repr(String(_entry(entry, :layout))))")

    counts = _load_array(read_bytes, _entry(entry, :counts))
    neighbors_flat = _load_array(read_bytes, _entry(entry, :neighbors))
    n = vocsize(voc)
    length(counts) == n ||
        error("the query_expansion column counts describe $(length(counts)) token(s) but the " *
              "vocabulary holds $n; the manifest and the vocabulary disagree")
    Int(sum(counts)) == length(neighbors_flat) ||
        error("the query_expansion column counts add up to $(Int(sum(counts))) neighbour(s) " *
              "and the member carries $(length(neighbors_flat)); they must agree")

    words = Dict{String,Vector{String}}()
    starts = Vector{Int}(undef, n)     # where each column begins, for the distance pass
    at = 1
    for id in 1:n
        c = Int(counts[id])
        starts[id] = at
        c == 0 && continue
        run = Vector{String}(undef, c)
        for r in 1:c
            j = Int(neighbors_flat[at + r - 1])
            1 <= j <= n ||
                error("the query_expansion network names token id $j, outside a vocabulary of $n")
            run[r] = gettoken(voc, j)
        end
        words[gettoken(voc, id)] = run
        at += c
    end

    dists = if _haskey(entry, :distances)
        de = _entry(entry, :distances)
        flat = _load_array(read_bytes, de)
        k = _entry(de, :columns)
        columns = if k isa AbstractString
            String(k) == "all" ? [id for id in 1:n if counts[id] > 0] :
                error("unknown query_expansion distances coverage: $(repr(String(k)))")
        else
            Int[Int(i) for i in _load_array(read_bytes, k)]
        end

        dd = Dict{String,Vector{Float32}}()
        seen = 0
        for id in columns
            1 <= id <= n && counts[id] > 0 ||
                error("the query_expansion distances name column $id, which carries no " *
                      "neighbours; the manifest and the network disagree")
            c = Int(counts[id])
            seen + c <= length(flat) ||
                error("the query_expansion distances member is shorter than the network it " *
                      "annotates; the manifest and the network disagree")
            dd[gettoken(voc, id)] = flat[(seen + 1):(seen + c)]
            seen += c
        end
        seen == length(flat) ||
            error("the query_expansion distances member carries $(length(flat)) value(s), " *
                  "$(length(flat) - seen) more than the network accounts for")
        dd
    else
        nothing
    end

    words, dists
end

"""
    load_profile(path::AbstractString) -> TextProfile

Reads back a profile written by [`save_profile`](@ref). `path` may be the directory it
produced or a `.zip` archive of it (see [`zip_profile`](@ref)); this is auto-detected via
`isdir(path)`, and a `.zip` is read directly from memory with no extraction.

The returned [`TextProfile`](@ref) rebuilds its own `TextConfig` from the stored policy and
the artifacts marked applied, so what it tokenizes with always matches what it carries.

A profile written by an older format version is refused by name rather than half-parsed:
there is no compatibility path, since carrying two layouts is what let the applied and saved
copies of an artifact drift apart in the first place.
"""
function load_profile(path::AbstractString)
    read_bytes = _profile_reader(path)
    read_file(name) = JSON3.read(read_bytes(name))
    manifest = read_file(_PROFILE_MANIFEST_NAME)
    version = String(get(manifest, :format_version, "(missing)"))
    version == _PROFILE_FORMAT_VERSION ||
        error("unsupported profile format_version: $version (this build reads " *
              "$_PROFILE_FORMAT_VERSION only, and has no conversion path). Refit the profile.")

    pol = _decode_policy(manifest[:policy])

    vocd = read_file(String(manifest[:vocabulary_file]))
    tokens = String.(vocd[:tokens])
    tok2id = Dict{String,UInt32}(tok => UInt32(i) for (i, tok) in enumerate(tokens))
    voc = Vocabulary(
        pol,
        tokens,
        Int32.(vocd[:occs]),
        Int32.(vocd[:ndocs]),
        tok2id,
        Ref{Int64}(Int64(vocd[:trainsize])),
        Ref{Int64}(Int64(vocd[:numtokens])),
    )

    wd = manifest[:weighting]
    gw = _decode_global_weighting(String(wd[:global_weighting]))
    lw = _decode_local_weighting(String(wd[:local_weighting]))
    weightd = read_file(String(wd[:weight_file]))
    model = VectorModel(gw, lw, voc, Int32(wd[:maxoccs]), Float32.(weightd[:weight]))

    art = manifest[:artifacts]

    stopwords, sw_applied = if haskey(art, :stopwords)
        e = art[:stopwords]
        Set{String}(String(w) for w in read_file(String(e[:file]))), Bool(e[:applied])
    else
        Set{String}(), false
    end

    lemmas, lem_applied = if haskey(art, :lemmas)
        e = art[:lemmas]
        d = read_file(String(e[:file]))
        Dict{String,String}(String(k) => String(v) for (k, v) in pairs(d)), Bool(e[:applied])
    else
        Dict{String,String}(), false
    end

    query_expansion, syndists, syn_applied = if haskey(art, :query_expansion)
        e = art[:query_expansion]
        words, dists = _load_expansion(read_bytes, e, voc)
        words, dists, Bool(e[:applied])
    else
        Dict{String,Vector{String}}(), nothing, false
    end

    # A `variants` entry written by an earlier build is ignored rather than rejected: the map is
    # derived from the vocabulary now (see `derive_variants`), so a stored one is dead weight, not
    # a conflict, and profiles fitted before the change stay loadable.
    TextProfile(model, stopwords, lemmas, query_expansion, syndists,
                AppliedArtifacts(stopwords=sw_applied, lemmas=lem_applied,
                                 query_expansion=syn_applied),
                _decode_lineage(manifest[:lineage]))
end

"""
    zip_profile(dir, zippath=dir * ".zip"; compress=true, compression_level=-1) -> zippath

Packages a profile directory (as written by [`save_profile`](@ref)) into a single `.zip`
archive at `zippath`, ready to distribute as one file. [`load_profile`](@ref) reads a
`.zip` produced this way directly (no extraction needed).

# Compression

Entries are **deflated**. They used to be stored uncompressed -- `zip_writefile` has no
compression option and always stores -- which went unnoticed because a profile zip is within a
kilobyte of the sum of its members, and that reads like framing overhead rather than like a
missing feature.

It was the single largest saving available to this format until the expansion network moved to
ids, and it still costs nothing in compatibility. Measured on the published Spanish profile
rebuilt under the current layout (`vocsize` 730,320, 5,590,091 edges), **48,392,862 bytes of
directory against 27,638,434 deflated, a 43% reduction**. Per member, the text ones are where
it comes from -- `vocabulary.json` 12,566,619 → 5,449,046, `weights.json` 7,117,381 → 1,283,405
-- while the binary ones give up least, which is what one would want: `neighbors.bin`
22,360,364 → 15,949,444 and the u8-quantized `distances.bin` 5,590,091 → 4,676,701. Bytes that
were already dense stay dense.

The two changes compose rather than compete: that same profile shipped at 163,945,326 bytes,
deflating alone would have made it 73,005,239, and the id layout takes it the rest of the way.

`compression_level` is passed through to ZipArchives (`1` fastest, `9` smallest, `-1` its
default compromise). `compress=false` restores the old stored behaviour, which is worth keeping
reachable for a caller that is about to compress the archive again anyway.
"""
function zip_profile(dir::AbstractString, zippath::AbstractString=dir * ".zip";
                     compress::Bool=true, compression_level::Integer=-1)
    isdir(dir) || error("zip_profile: not a directory: $dir")
    ZipArchives.ZipWriter(zippath) do w
        for name in sort(readdir(dir))
            fpath = joinpath(dir, name)
            isfile(fpath) || continue
            if compress
                # `zip_writefile` stores unconditionally, so a compressed entry has to be opened,
                # written and committed rather than written in one call
                ZipArchives.zip_newfile(w, name; compress=true,
                                        compression_level=Int(compression_level))
                write(w, read(fpath))
                ZipArchives.zip_commitfile(w)
            else
                ZipArchives.zip_writefile(w, name, read(fpath))
            end
        end
    end
    zippath
end

"""
    list_remote_profiles(; repo::AbstractString="sadit/TextSearch.jl",
                           tag::AbstractString="v1.1.0",
                           url::Union{Nothing,AbstractString}=nothing) -> Vector{NamedTuple}

Queries and returns available pre-computed linguistic profiles from GitHub releases or a custom URL.
Returns a vector of `(name=nickname, filename=name, size=size_in_bytes, url=download_url, tag=tag)`.
"""
function list_remote_profiles(; repo::AbstractString="sadit/TextSearch.jl",
                                tag::AbstractString="v1.1.0",
                                url::Union{Nothing,AbstractString}=nothing)
    api_url = url !== nothing ? String(url) :
              (tag == "latest" ?
                  "https://api.github.com/repos/$repo/releases/latest" :
                  "https://api.github.com/repos/$repo/releases/tags/$tag")
    tmppath = tempname() * ".json"
    data = try
        Downloads.download(api_url, tmppath; headers=["User-Agent" => "TextSearch.jl"])
        JSON3.read(read(tmppath, String))
    finally
        rm(tmppath; force=true)
    end

    results = NamedTuple{(:name, :filename, :size, :url, :tag), Tuple{String, String, Int, String, String}}[]
    if haskey(data, :assets)
        for asset in data.assets
            name = String(asset.name)
            if endswith(name, ".zip")
                nickname = first(splitext(name))
                sz = Int(asset.size)
                dl_url = String(asset.browser_download_url)
                push!(results, (name=nickname, filename=name, size=sz, url=dl_url, tag=String(tag)))
            end
        end
    elseif data isa AbstractVector
        for item in data
            if haskey(item, :assets)
                rtag = haskey(item, :tag_name) ? String(item.tag_name) : String(tag)
                for asset in item.assets
                    name = String(asset.name)
                    if endswith(name, ".zip")
                        nickname = first(splitext(name))
                        sz = Int(asset.size)
                        dl_url = String(asset.browser_download_url)
                        push!(results, (name=nickname, filename=name, size=sz, url=dl_url, tag=rtag))
                    end
                end
            elseif haskey(item, :name) && endswith(String(item.name), ".zip")
                name = String(item.name)
                nickname = first(splitext(name))
                sz = haskey(item, :size) ? Int(item.size) : 0
                dl_url = haskey(item, :url) ? String(item.url) : (haskey(item, :browser_download_url) ? String(item.browser_download_url) : "")
                push!(results, (name=nickname, filename=name, size=sz, url=dl_url, tag=String(tag)))
            end
        end
    end
    results
end

"""
    download_profile(nickname_or_url::AbstractString;
                     repo::AbstractString="sadit/TextSearch.jl",
                     tag::AbstractString="v1.1.0",
                     dest::Union{Nothing,AbstractString}=nothing,
                     url::Union{Nothing,AbstractString}=nothing,
                     force::Bool=false) -> String

Downloads a pre-computed linguistic profile (`<nickname>.zip`) from a GitHub release or direct URL
and saves it locally. By default, installs under `~/.textsearch/profiles/<nickname>.zip` (or
`\$TEXTSEARCH_HOME/profiles/<nickname>.zip`), or into `dest` if explicitly specified.
Returns the file path of the downloaded archive.
"""
function download_profile(nickname_or_url::AbstractString;
                          repo::AbstractString="sadit/TextSearch.jl",
                          tag::AbstractString="v1.1.0",
                          dest::Union{Nothing,AbstractString}=nothing,
                          url::Union{Nothing,AbstractString}=nothing,
                          force::Bool=false)
    is_direct_url = startswith(nickname_or_url, "http://") || startswith(nickname_or_url, "https://")
    nickname = is_direct_url ? first(splitext(basename(nickname_or_url))) : String(nickname_or_url)

    target = dest === nothing ?
        joinpath(get(ENV, "TEXTSEARCH_HOME", joinpath(homedir(), ".textsearch")), "profiles", "$nickname.zip") :
        String(dest)
    if isfile(target) && !force
        return target
    end
    mkpath(dirname(target))

    dl_url = if is_direct_url
        String(nickname_or_url)
    elseif url !== nothing
        endswith(url, ".zip") ? String(url) : joinpath(String(url), "$nickname.zip")
    else
        "https://github.com/$repo/releases/download/$tag/$nickname.zip"
    end

    tmppath = tempname() * ".zip"
    try
        Downloads.download(dl_url, tmppath; headers=["User-Agent" => "TextSearch.jl"])
        mv(tmppath, target; force=true)
    catch e
        rm(tmppath; force=true)
        rethrow(e)
    end
    target
end
