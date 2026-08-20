# This file is a part of TextSearch.jl

export save_profile, load_profile, zip_profile

const _PROFILE_FORMAT_VERSION = 2
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
    _encode_transformation_with_files(tt, counter=Ref(0)) -> (json, files)

Encodes `tt` into its tagged-union JSON representation, EXCEPT any `IgnoreStopwords` step's
word list, which is instead written to its own `"stopwords.json"`-style file (numbered on a
second, third, ... occurrence via `counter`): `json` gets a `"file"` reference instead of an
inlined `"words"` array, and `files` collects `filename => word_vector` pairs to be written
by the caller (`save_profile`).
"""
function _encode_transformation_with_files(tt::IdentityTokenTransformation, counter::Ref{Int}=Ref(0))
    Dict("kind" => "identity"), Pair{String,Vector{String}}[]
end

function _encode_transformation_with_files(tt::IgnoreStopwords, counter::Ref{Int}=Ref(0))
    counter[] += 1
    fname = counter[] == 1 ? "stopwords.json" : "stopwords_$(counter[]).json"
    Dict("kind" => "stopwords", "file" => fname), [fname => collect(tt.stopwords)]
end

function _encode_transformation_with_files(tt::SnowballTokenTransformation, counter::Ref{Int}=Ref(0))
    Dict("kind" => "snowball", "algorithm" => tt.stemmer.alg, "charenc" => tt.stemmer.enc), Pair{String,Vector{String}}[]
end

function _encode_transformation_with_files(tt::ChainTransformation, counter::Ref{Int}=Ref(0))
    steps = Any[]
    files = Pair{String,Vector{String}}[]
    for s in tt.list
        sjson, sfiles = _encode_transformation_with_files(s, counter)
        push!(steps, sjson)
        append!(files, sfiles)
    end
    Dict("kind" => "chain", "steps" => steps), files
end

_encode_transformation_with_files(tt, counter::Ref{Int}=Ref(0)) =
    error("cannot serialize transformation of type $(typeof(tt)) into a profile")

"""
    _decode_transformation(d, read_file::Function)

Decodes a transformation's tagged-union JSON `d` back into an
[`AbstractTokenTransformation`](@ref); `read_file(name::AbstractString)` fetches and
JSON3-parses a referenced file (`"stopwords"` kind) -- the caller supplies one that reads
from a plain directory or from an open zip archive, so this function itself doesn't care
which.
"""
function _decode_transformation(d, read_file::Function)
    kind = String(d[:kind])
    if kind == "identity"
        IdentityTokenTransformation()
    elseif kind == "stopwords"
        words = read_file(String(d[:file]))
        IgnoreStopwords(Set{String}(String(w) for w in words))
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
                 synonyms::AbstractDict=Dict{String,Vector{Pair{String,Float32}}}()) -> dir

Serializes `model` -- its `voc`'s [`TextConfig`](@ref) and vocabulary counters, its
weighting scheme, and its precomputed `weight` vector -- together with a `synonyms`
network (e.g. as produced by [`LSI.synonyms`](@ref)) into `dir` (created if missing) as a
small directory of plain, human-readable JSON files: one per "large" piece --
`vocabulary.json`, `weights.json`, `synonyms.json` (only written if `synonyms` is
non-empty), and `stopwords.json` (only written if `model`'s `transformation` involves
one) -- tied together by a small `manifest.json` that holds everything else
(normalization/tokenization flags, weighting tags, and file references). Load it back
with [`load_profile`](@ref), or package it for distribution with [`zip_profile`](@ref).

Deliberately NOT a generic object-graph dump (unlike e.g. JLD2): every field is encoded
by hand into a small, versioned schema, so every file is fully inspectable/diffable/portable
and there is nothing pointer- or code-shaped to accidentally serialize.

A `TokenizationConfig` with custom (non-empty) `generators`, or a `transformation` that
isn't `IdentityTokenTransformation`/`IgnoreStopwords`/`SnowballTokenTransformation`/
`ChainTransformation` of those, errors clearly rather than silently mis-saving.
"""
function save_profile(dir::AbstractString, model::VectorModel;
                       synonyms::AbstractDict=Dict{String,Vector{Pair{String,Float32}}}())
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

    transformation_json, stopword_files = _encode_transformation_with_files(voc.textconfig.transformation)
    for (fname, words) in stopword_files
        _write_json(joinpath(dir, fname), words)
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

    _write_json(joinpath(dir, _PROFILE_MANIFEST_NAME), manifest)
    dir
end

"""
    load_profile(path::AbstractString) -> (model::VectorModel, synonyms::Dict{String,Vector{Pair{String,Float32}}})

Reads back a profile written by [`save_profile`](@ref) -- `path` may be either the
directory `save_profile` produced, or a `.zip` archive of it (see [`zip_profile`](@ref));
this is auto-detected via `isdir(path)`, and a `.zip` is read directly from memory, no
extraction to disk needed. Reconstructs the `Vocabulary` (rebuilding `token2id` from the
stored `tokens` list) and `VectorModel`, and returns the attached synonym network
alongside it (an empty `Dict` if none was saved), ready to pass straight into e.g.
`TextInvertedFile(model; synonyms, ...)`.

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

    model, synonyms
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
