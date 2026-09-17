# This file is a part of TextSearch.jl

export save_profile, load_profile, zip_profile, download_profile, list_remote_profiles

# Bumped from "1.0" with the policy/artifact split. The freeze at "1.0" was right while every
# schema change was additive and older files still loaded; this one changes the layout and
# drops compatibility, so the version's job flips from "irrelevant" to "refuse an older file
# with a sentence that says what happened" rather than half-parsing it.
# Held at 1.0 deliberately. The format has no released consumers -- nothing is published and no
# profile exists outside this repository -- so the version is not yet a compatibility mechanism
# and bumping it on every layout change buys nothing. `load_profile` still refuses a mismatch, so
# the field is ready to do that job from the first release onward; until then a layout change just
# means the profiles in `corpus-profiles/` get refitted, which they need anyway.
#
# For the record of what has changed under 1.0: the policy/artifacts split, and the rename of the
# artifact from "synonyms" to "query_expansion" (manifest key and both files). That rename
# mattered beyond tidiness -- calling it a synonym network invited judging its entries as
# substitutable words, and a long stretch of work went into filtering it on that basis before the
# objective was restated: the artifact exists to enrich a query, and topically related terms are
# what serve that. See the note in `src/lsi.jl`.
const _PROFILE_FORMAT_VERSION = "1.0"
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
        "emojis" => [string(c) for c in n.emojis],
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

# ── save_profile / load_profile / zip_profile ────────────────────────────────

"""
    save_profile(dir::AbstractString, p::TextProfile) -> dir

Serializes a [`TextProfile`](@ref) into `dir` (created if missing) as a small directory of
plain, human-readable JSON files: one per "large" piece -- `vocabulary.json`, `weights.json`,
and `stopwords.json`/`lemmas.json`/`query_expansion.json` for whichever artifacts are non-empty --
tied together by a `manifest.json` holding everything else.

The one exception is `query_expansion_distances.bin`, which is a **binary** member: tens of
thousands of cosine distances, `u8`-quantized, described entirely by its manifest entry (see
`arraystore.jl`). Every other file stays text, and the reason this one does not is measured
rather than assumed -- as JSON it was 569,045 bytes against 47,501, i.e. 44% of a whole profile
spent on digits nobody reads, and the quantization is retrieval-identical. The same mechanism is
what a stored dense projection will use.

The manifest keeps policy and artifacts apart, which is the point of the layout:

```
policy:     { normalization: {...}, tokenization: {...} }
artifacts:  { stopwords: {file, applied}, lemmas: {file, applied},
              query_expansion: {file, applied,
                                distances: {file, dtype, shape, quant, keys}} }
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
        syn_order = sort!(collect(keys(p.query_expansion)))
        _write_json(joinpath(dir, "query_expansion.json"),
            Dict(tok => syns for (tok, syns) in p.query_expansion))
        entry = Dict{String,Any}("file" => "query_expansion.json", "applied" => p.applied.query_expansion)

        # Only for tokens the ranking carries: a distance list without its words could not be
        # interpreted, and the distances live in their own member so a consumer that needs only
        # the ranking can skip them.
        #
        # That member is BINARY, and quantized to u8, which is the one place this format spends
        # its inspectability. The content is tens of thousands of cosine distances that nobody
        # reads by eye, and JSON text costs 12x for them: measured on a real profile, 569,045
        # bytes against 47,501, on a file that is 44% of the whole profile. The quantization is
        # free where it counts -- Float32 against u8 gives identical top-1 results and 0.9995
        # top-10 overlap over 400 queries. See `arraystore.jl`.
        #
        # Stored FLAT, against the network's own sorted key order, and naming neither the tokens
        # nor the per-token lengths -- both are derivable from `query_expansion.json`, which is
        # right there. Writing them anyway was the first cut of this, and it cost more than the
        # binary member saved: 6,068 tokens in the manifest took it from 28,055 bytes to 99,528,
        # a third copy of the vocabulary, which is what this layout exists to avoid.
        #
        # What is NOT derivable is which tokens carry distances at all: a ranking may carry none
        # (`_restrict_query_expansion` keeps a distance list only when it covers the whole list,
        # all or nothing). So coverage is recorded -- as the string "all" in the normal case,
        # costing nothing, and as a binary array of positions into the sorted key order otherwise.
        if p.query_expansion_distances !== nothing
            covered = Int[]
            flat = Float32[]
            for (i, tok) in enumerate(syn_order)
                ds = get(p.query_expansion_distances, tok, nothing)
                (ds === nothing || isempty(ds)) && continue
                length(ds) == length(p.query_expansion[tok]) || continue
                push!(covered, i)
                append!(flat, ds)
            end
            if !isempty(flat)
                e = _save_array(dir, "query_expansion_distances.bin", flat; quantize=true)
                e["keys"] = length(covered) == length(syn_order) ? "all" :
                    _save_array(dir, "query_expansion_distances_keys.bin", UInt32.(covered))
                entry["distances"] = e
            end
        end
        artifacts["query_expansion"] = entry
    end

    _write_json(joinpath(dir, _PROFILE_MANIFEST_NAME), Dict(
        "format_version" => _PROFILE_FORMAT_VERSION,
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
        net = read_file(String(e[:file]))
        words = Dict{String,Vector{String}}(
            String(tok) => String[String(s) for s in syns] for (tok, syns) in pairs(net))
        # A profile written before the distances became a binary member names them under
        # `distances_file`. Say so instead of silently loading a profile with no distances at
        # all: the artifact would simply vanish, `expand_query!` would fall back to rank
        # weighting, and nothing anywhere would report that it had happened. This is the job the
        # format version cannot do while it is deliberately held at 1.0.
        haskey(e, :distances_file) &&
            error("this profile stores its query_expansion distances as JSON " *
                  "('$(String(e[:distances_file]))'), which this build no longer reads: they " *
                  "are now a quantized binary member. Refit the profile.")

        # One flat binary array, cut back up with what is already loaded: the network's sorted
        # key order says WHICH token each run belongs to, and that token's own neighbour list
        # says HOW LONG the run is. See `save_profile` for why neither is stored.
        dists = if haskey(e, :distances)
            de = e[:distances]
            flat = _load_array(read_bytes, de)
            order = sort!(collect(keys(words)))
            k = _entry(de, :keys)
            covered = if k isa AbstractString
                String(k) == "all" ? collect(eachindex(order)) :
                    error("unknown query_expansion distances coverage: $(repr(String(k)))")
            else
                Int[Int(i) for i in _load_array(read_bytes, k)]
            end

            dd = Dict{String,Vector{Float32}}()
            at = 1
            for i in covered
                1 <= i <= length(order) ||
                    error("the query_expansion distances name key $i of $(length(order)); " *
                          "the manifest and the network disagree")
                tok = order[i]
                n = length(words[tok])
                at + n - 1 <= length(flat) ||
                    error("the query_expansion distances member is shorter than the network it " *
                          "annotates; the manifest and the network disagree")
                dd[tok] = flat[at:(at + n - 1)]
                at += n
            end
            at == length(flat) + 1 ||
                error("the query_expansion distances member carries $(length(flat)) value(s), " *
                      "$(length(flat) - at + 1) more than the network accounts for")
            dd
        else
            nothing
        end
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
    zip_profile(dir::AbstractString, zippath::AbstractString=dir * ".zip") -> zippath

Packages a profile directory (as written by [`save_profile`](@ref)) into a single `.zip`
archive at `zippath`, ready to distribute as one file. [`load_profile`](@ref) reads a
`.zip` produced this way directly (no extraction needed).
"""
function zip_profile(dir::AbstractString, zippath::AbstractString=dir * ".zip")
    isdir(dir) || error("zip_profile: not a directory: $dir")
    ZipArchives.ZipWriter(zippath) do w
        for name in sort(readdir(dir))
            fpath = joinpath(dir, name)
            isfile(fpath) || continue
            ZipArchives.zip_writefile(w, name, read(fpath))
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
