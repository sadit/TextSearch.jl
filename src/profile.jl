# This file is a part of TextSearch.jl

export save_profile, load_profile, zip_profile

# Kept at "1.0" while the schema is still being actively developed (lemmas/encoder/
# stopword_candidates sections added on top of it are all optional/additive, so old
# profiles without them keep loading fine) -- bump only once the schema is genuinely
# settled, not for every incremental addition.
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

# ── transformation tagged union ──────────────────────────────────────────────
#
# Every "big" piece of a profile (vocabulary, weights, synonyms, and -- if present --
# a stopwords list) lives in its OWN file inside the profile directory, referenced by
# name from `manifest.json`; only small scalar/tag data is inlined there. This keeps
# each file greppable/diffable on its own and lets a consumer skip loading the pieces
# it doesn't need.

"""
    _decode_snowball_transformation(algorithm::AbstractString, charenc::AbstractString)

Reconstructs a [`SnowballTokenTransformation`](@ref) from a profile by dispatching into
`TextSearchSnowballExt` (the `Snowball`/`Languages` package extension), looked up at
runtime via `Base.get_extension` rather than as an overridable method -- a package
extension is not allowed to overwrite a method already defined in the parent package, so
the extension instead defines a same-named function inside its OWN module
(`TextSearchSnowballExt._construct_snowball_transformation`), which this looks up
dynamically. Errors clearly if the extension isn't loaded yet, naming what to `using` first.
"""
function _decode_snowball_transformation(algorithm::AbstractString, charenc::AbstractString)
    ext = Base.get_extension(TextSearch, :TextSearchSnowballExt)
    ext === nothing && error(
        "loading a profile with a Snowball-stemmed transformation (algorithm=\"$algorithm\") " *
        "requires `using Snowball, Languages` to be active first, so the TextSearchSnowballExt " *
        "package extension can reconstruct the stemmer."
    )
    ext._construct_snowball_transformation(algorithm, charenc)
end

"""
    _transformation_filename(counter, base) -> String

Names the side file for a transformation step that carries bulk data, `"\$base.json"` for
the first step of that kind and `"\$(base)_2.json"`, `"\$(base)_3.json"`, ... for any
further ones. `counter` tallies per `base` rather than across all kinds, so a chain of an
`IgnoreStopwords` and a `LemmaTransformation` yields `"stopwords.json"` and
`"lemma_map.json"` -- not a `"lemma_map_2.json"` whose suffix would suggest a second lemma
map that does not exist.
"""
function _transformation_filename(counter::Dict{String,Int}, base::AbstractString)
    n = counter[base] = get(counter, base, 0) + 1
    n == 1 ? "$base.json" : "$(base)_$n.json"
end

"""
    _encode_transformation_with_files(tt, counter=Dict{String,Int}()) -> (json, files)

Encodes `tt` into its tagged-union JSON representation, EXCEPT the bulk payload of any step
that has one -- an `IgnoreStopwords`' word list, a `LemmaTransformation`'s mapping -- which
is instead written to its own side file (named by [`_transformation_filename`](@ref)):
`json` gets a `"file"` reference instead of the inlined data, and `files` collects
`filename => payload` pairs to be written by the caller (`save_profile`).
"""
function _encode_transformation_with_files(tt::IdentityTokenTransformation, counter::Dict{String,Int}=Dict{String,Int}())
    Dict("kind" => "identity"), Pair{String,Any}[]
end

function _encode_transformation_with_files(tt::IgnoreStopwords, counter::Dict{String,Int}=Dict{String,Int}())
    fname = _transformation_filename(counter, "stopwords")
    Dict("kind" => "stopwords", "file" => fname), Pair{String,Any}[fname => collect(tt.stopwords)]
end

function _encode_transformation_with_files(tt::LemmaTransformation, counter::Dict{String,Int}=Dict{String,Int}())
    fname = _transformation_filename(counter, "lemma_map")
    Dict("kind" => "lemmas", "file" => fname), Pair{String,Any}[fname => tt.lemmas]
end

function _encode_transformation_with_files(tt::SnowballTokenTransformation, counter::Dict{String,Int}=Dict{String,Int}())
    Dict("kind" => "snowball", "algorithm" => tt.stemmer.alg, "charenc" => tt.stemmer.enc), Pair{String,Any}[]
end

function _encode_transformation_with_files(tt::ChainTransformation, counter::Dict{String,Int}=Dict{String,Int}())
    steps = Any[]
    files = Pair{String,Any}[]
    for s in tt.list
        sjson, sfiles = _encode_transformation_with_files(s, counter)
        push!(steps, sjson)
        append!(files, sfiles)
    end
    Dict("kind" => "chain", "steps" => steps), files
end

_encode_transformation_with_files(tt, counter::Dict{String,Int}=Dict{String,Int}()) =
    error("cannot serialize transformation of type $(typeof(tt)) into a profile")

"""
    _decode_transformation(d, read_file::Function)

Decodes a transformation's tagged-union JSON `d` back into an
[`AbstractTokenTransformation`](@ref); `read_file(name::AbstractString)` fetches and
JSON3-parses a referenced file (the `"stopwords"` and `"lemmas"` kinds) -- the caller
supplies one that reads from a plain directory or from an open zip archive, so this
function itself doesn't care which.
"""
function _decode_transformation(d, read_file::Function)
    kind = String(d[:kind])
    if kind == "identity"
        IdentityTokenTransformation()
    elseif kind == "stopwords"
        words = read_file(String(d[:file]))
        IgnoreStopwords(Set{String}(String(w) for w in words))
    elseif kind == "lemmas"
        map = read_file(String(d[:file]))
        LemmaTransformation(Dict{String,String}(String(k) => String(v) for (k, v) in pairs(map)))
    elseif kind == "snowball"
        _decode_snowball_transformation(String(d[:algorithm]), String(d[:charenc]))
    elseif kind == "chain"
        ChainTransformation(AbstractTokenTransformation[_decode_transformation(s, read_file) for s in d[:steps]])
    else
        error("unknown transformation kind: $kind")
    end
end

# ── TextConfig (normalization / tokenization / transformation) ──────────────
#
# `normalization`/`tokenization` are always small (a handful of flags, regex patterns, and
# the emoji set) so they stay inlined in the manifest; only `transformation`'s stopwords
# (if any) are split out, via `_encode_transformation_with_files` above.

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
    save_profile(dir::AbstractString, model::VectorModel;
                 synonyms::AbstractDict=Dict{String,Vector{Pair{String,Float32}}}(),
                 lemmas::AbstractDict{String,String}=Dict{String,String}(),
                 stopword_candidates::AbstractVector{<:AbstractString}=String[],
                 encoder::Union{Nothing,NamedTuple}=nothing) -> dir

Serializes `model` -- its `voc`'s [`TextConfig`](@ref) and vocabulary counters, its
weighting scheme, and its precomputed `weight` vector -- together with a `synonyms`
network (e.g. as produced by [`LSI.synonyms`](@ref)), a `lemmas` map (e.g. as produced by
[`lemma_clusters`](@ref)), a `stopword_candidates` list (e.g. as produced by
[`stopword_candidates`](@ref)), and `encoder` provenance metadata, into `dir` (created if
missing) as a small directory of plain, human-readable JSON files: one per "large" piece --
`vocabulary.json`, `weights.json`, `synonyms.json`/`lemmas.json`/`stopword_candidates.json`
(each only written if non-empty), and `stopwords.json` (only written if `model`'s
`transformation` involves one) -- tied together by a small `manifest.json` that holds
everything else (normalization/tokenization flags, weighting tags, encoder metadata, and
file references). Load it back with [`load_profile`](@ref), or package it for
distribution with [`zip_profile`](@ref).

`lemmas` should map only non-identity tokens (a lookup miss on load means "token is its
own lemma"). `stopword_candidates` is distinct from and additive to the `stopwords.json`
mechanism above: that one is the *applied* `IgnoreStopwords` set baked into the
transformation, this one is the *candidate* list a frequency-based detector produced for
review -- a profile can have neither, either, or both (they typically coincide when a
detector's candidates were the ones actually wired into the transformation). `encoder` is
a `NamedTuple` such as `(; kind=:lsi, outdim=128, scaling=:none, source_path="")` recording
which encoder produced `synonyms`/`lemmas` and its hyperparameters -- for provenance only;
the encoder's own projection (e.g. an LSI `P` matrix) is not persisted.

Deliberately NOT a generic object-graph dump (unlike e.g. JLD2): every field is encoded
by hand into a small, versioned schema, so every file is fully inspectable/diffable/portable
and there is nothing pointer- or code-shaped to accidentally serialize.

A `TokenizationConfig` with custom (non-empty) `generators`, or a `transformation` that
isn't `IdentityTokenTransformation`/`IgnoreStopwords`/`SnowballTokenTransformation`/
`ChainTransformation` of those, errors clearly rather than silently mis-saving.
"""
function save_profile(dir::AbstractString, model::VectorModel;
                       synonyms::AbstractDict=Dict{String,Vector{Pair{String,Float32}}}(),
                       lemmas::AbstractDict{String,String}=Dict{String,String}(),
                       stopword_candidates::AbstractVector{<:AbstractString}=String[],
                       encoder::Union{Nothing,NamedTuple}=nothing)
    mkpath(dir)
    voc = model.voc

    _write_json(joinpath(dir, "vocabulary.json"), Dict(
        "tokens" => voc.token,
        "occs" => voc.occs,
        "ndocs" => voc.ndocs,
        "trainsize" => voc.trainsize[],
        "numtokens" => voc.numtokens[],
    ))

    _write_json(joinpath(dir, "weights.json"), Dict("weight" => model.weight))

    transformation_json, transformation_files = _encode_transformation_with_files(voc.textconfig.transformation)
    for (fname, payload) in transformation_files
        _write_json(joinpath(dir, fname), payload)
    end

    manifest = Dict(
        "format_version" => _PROFILE_FORMAT_VERSION,
        "textconfig" => Dict(
            "normalization" => _encode_normalization(voc.textconfig.normalization),
            "tokenization" => _encode_tokenization(voc.textconfig.tokenization),
            "transformation" => transformation_json,
            "expand_query_synonyms" => voc.textconfig.expand_query_synonyms,
        ),
        "vocabulary_file" => "vocabulary.json",
        "weighting" => Dict(
            "global_weighting" => _encode_global_weighting(model.global_weighting),
            "local_weighting" => _encode_local_weighting(model.local_weighting),
            "maxoccs" => model.maxoccs,
            "weight_file" => "weights.json",
        ),
    )

    if !isempty(synonyms)
        _write_json(joinpath(dir, "synonyms.json"),
            Dict(tok => [[syn, Float32(dist)] for (syn, dist) in syns] for (tok, syns) in synonyms))
        manifest["synonyms_file"] = "synonyms.json"
    end

    if !isempty(lemmas)
        _write_json(joinpath(dir, "lemmas.json"), Dict(lemmas))
        manifest["lemmas_file"] = "lemmas.json"
    end

    if !isempty(stopword_candidates)
        _write_json(joinpath(dir, "stopword_candidates.json"), collect(stopword_candidates))
        manifest["stopword_candidates_file"] = "stopword_candidates.json"
    end

    if encoder !== nothing
        manifest["encoder"] = Dict(String(k) => (v isa Symbol ? String(v) : v) for (k, v) in pairs(encoder))
    end

    _write_json(joinpath(dir, _PROFILE_MANIFEST_NAME), manifest)
    dir
end

"""
    load_profile(path::AbstractString) -> (; model, synonyms, lemmas, stopword_candidates, encoder)

Reads back a profile written by [`save_profile`](@ref) -- `path` may be either the
directory `save_profile` produced, or a `.zip` archive of it (see [`zip_profile`](@ref));
this is auto-detected via `isdir(path)`, and a `.zip` is read directly from memory, no
extraction to disk needed. Reconstructs the `Vocabulary` (rebuilding `token2id` from the
stored `tokens` list) and `VectorModel`, and returns a `NamedTuple`:

- `model::VectorModel`
- `synonyms::Dict{String,Vector{Pair{String,Float32}}}` (empty if none saved)
- `lemmas::Dict{String,String}` (empty if none saved)
- `stopword_candidates::Vector{String}` (empty if none saved)
- `encoder::Union{Nothing,Dict{String,Any}}` (`nothing` if none saved)

`model`/`synonyms` are ready to pass straight into e.g. `TextInvertedFile(model;
synonyms, ...)`.

Errors if the profile's `transformation` needs `Snowball`/`Languages` to reconstruct (a
`"snowball"` tagged-union entry) and those packages aren't loaded yet.
"""
function load_profile(path::AbstractString)
    read_file = _profile_reader(path)
    manifest = read_file(_PROFILE_MANIFEST_NAME)
    manifest[:format_version] == _PROFILE_FORMAT_VERSION ||
        error("unsupported profile format_version: $(manifest[:format_version]) (expected $_PROFILE_FORMAT_VERSION)")

    tc = manifest[:textconfig]
    textconfig = TextConfig(
        normalization=_decode_normalization(tc[:normalization]),
        tokenization=_decode_tokenization(tc[:tokenization]),
        transformation=_decode_transformation(tc[:transformation], read_file),
        expand_query_synonyms=Bool(get(tc, :expand_query_synonyms, false)),
    )

    vocd = read_file(String(manifest[:vocabulary_file]))
    tokens = String.(vocd[:tokens])
    tok2id = Dict{String,UInt32}(tok => UInt32(i) for (i, tok) in enumerate(tokens))
    voc = Vocabulary(
        textconfig,
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

    synonyms = if haskey(manifest, :synonyms_file)
        synd = read_file(String(manifest[:synonyms_file]))
        Dict{String,Vector{Pair{String,Float32}}}(
            String(tok) => [Pair(String(syn), Float32(dist)) for (syn, dist) in syns]
            for (tok, syns) in synd
        )
    else
        Dict{String,Vector{Pair{String,Float32}}}()
    end

    lemmas = if haskey(manifest, :lemmas_file)
        lemd = read_file(String(manifest[:lemmas_file]))
        Dict{String,String}(String(tok) => String(lemma) for (tok, lemma) in lemd)
    else
        Dict{String,String}()
    end

    stopword_candidates = if haskey(manifest, :stopword_candidates_file)
        String.(read_file(String(manifest[:stopword_candidates_file])))
    else
        String[]
    end

    encoder = if haskey(manifest, :encoder)
        Dict{String,Any}(String(k) => v for (k, v) in pairs(manifest[:encoder]))
    else
        nothing
    end

    (; model, synonyms, lemmas, stopword_candidates, encoder)
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
