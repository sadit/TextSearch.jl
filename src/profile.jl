# This file is a part of TextSearch.jl

export save_profile, load_profile, zip_profile

# Bumped from "1.0" with the policy/artifact split. The freeze at "1.0" was right while every
# schema change was additive and older files still loaded; this one changes the layout and
# drops compatibility, so the version's job flips from "irrelevant" to "refuse an older file
# with a sentence that says what happened" rather than half-parsing it.
const _PROFILE_FORMAT_VERSION = "2.0"
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
         "tokenization" => _encode_tokenization(tc.tokenization))
end

_decode_policy(d) = TextConfig(normalization=_decode_normalization(d[:normalization]),
                               tokenization=_decode_tokenization(d[:tokenization]))

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
    _profile_reader(path::AbstractString) -> read_file::Function

Returns a `read_file(name::AbstractString) -> JSON3` closure that fetches and parses a
named member of the profile at `path` -- a plain directory if `isdir(path)`, otherwise a
`.zip` archive (opened once and re-read from memory for every subsequent `read_file` call).
This is what lets [`load_profile`](@ref) not care which of the two forms it was handed.
"""
function _profile_reader(path::AbstractString)
    if isdir(path)
        name -> JSON3.read(read(joinpath(path, name)))
    else
        buf = read(path)
        zr = ZipArchives.ZipReader(buf)
        name -> JSON3.read(ZipArchives.zip_readentry(zr, name))
    end
end

# ── save_profile / load_profile / zip_profile ────────────────────────────────

"""
    save_profile(dir::AbstractString, p::TextProfile) -> dir

Serializes a [`TextProfile`](@ref) into `dir` (created if missing) as a small directory of
plain, human-readable JSON files: one per "large" piece -- `vocabulary.json`, `weights.json`,
and `stopwords.json`/`lemmas.json`/`synonyms.json`/`synonym_distances.json` for whichever
artifacts are non-empty -- tied together by a `manifest.json` holding everything else.

The manifest keeps policy and artifacts apart, which is the point of the layout:

```
policy:     { normalization: {...}, tokenization: {...} }
artifacts:  { stopwords: {file, applied}, lemmas: {file, applied}, synonyms: {file, ...} }
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

    if !isempty(p.synonyms)
        _write_json(joinpath(dir, "synonyms.json"),
            Dict(tok => syns for (tok, syns) in p.synonyms))
        entry = Dict{String,Any}("file" => "synonyms.json", "applied" => p.applied.synonyms)

        # Only for tokens the ranking carries: a distance list without its words could not be
        # interpreted, and the distances live in their own file so a consumer that needs only
        # the ranking -- which is the normal case -- can skip the bulk of the network.
        if p.synonym_distances !== nothing
            dd = Dict{String,Vector{Float32}}()
            for (tok, ds) in p.synonym_distances
                haskey(p.synonyms, tok) && !isempty(ds) || continue
                dd[tok] = ds
            end
            if !isempty(dd)
                _write_json(joinpath(dir, "synonym_distances.json"), dd)
                entry["distances_file"] = "synonym_distances.json"
            end
        end
        artifacts["synonyms"] = entry
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
    read_file = _profile_reader(path)
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

    synonyms, syndists, syn_applied = if haskey(art, :synonyms)
        e = art[:synonyms]
        net = read_file(String(e[:file]))
        words = Dict{String,Vector{String}}(
            String(tok) => String[String(s) for s in syns] for (tok, syns) in pairs(net))
        dists = if haskey(e, :distances_file)
            dd = read_file(String(e[:distances_file]))
            Dict{String,Vector{Float32}}(
                String(tok) => Float32[Float32(d) for d in ds] for (tok, ds) in pairs(dd))
        else
            nothing
        end
        words, dists, Bool(e[:applied])
    else
        Dict{String,Vector{String}}(), nothing, false
    end

    TextProfile(model, stopwords, lemmas, synonyms, syndists,
                AppliedArtifacts(stopwords=sw_applied, lemmas=lem_applied, synonyms=syn_applied),
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
