# This file is a part of TextSearch.jl

export save_profile, load_profile

const _PROFILE_FORMAT_VERSION = 1

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

_encode_transformation(::IdentityTokenTransformation) = Dict("kind" => "identity")

_encode_transformation(tt::IgnoreStopwords) = Dict("kind" => "stopwords", "words" => collect(tt.stopwords))

_encode_transformation(tt::SnowballTokenTransformation) =
    Dict("kind" => "snowball", "algorithm" => tt.stemmer.alg, "charenc" => tt.stemmer.enc)

_encode_transformation(tt::ChainTransformation) =
    Dict("kind" => "chain", "steps" => [_encode_transformation(s) for s in tt.list])

_encode_transformation(tt) = error("cannot serialize transformation of type $(typeof(tt)) into a profile")

function _decode_transformation(d)
    kind = String(d[:kind])
    if kind == "identity"
        IdentityTokenTransformation()
    elseif kind == "stopwords"
        IgnoreStopwords(Set{String}(String(w) for w in d[:words]))
    elseif kind == "snowball"
        _decode_snowball_transformation(String(d[:algorithm]), String(d[:charenc]))
    elseif kind == "chain"
        ChainTransformation(AbstractTokenTransformation[_decode_transformation(s) for s in d[:steps]])
    else
        error("unknown transformation kind: $kind")
    end
end

# ── TextConfig (normalization / tokenization / transformation) ──────────────

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

function _encode_textconfig(c::TextConfig)
    Dict(
        "normalization" => _encode_normalization(c.normalization),
        "tokenization" => _encode_tokenization(c.tokenization),
        "transformation" => _encode_transformation(c.transformation),
        "expand_query_synonyms" => c.expand_query_synonyms,
    )
end

function _decode_textconfig(d)
    TextConfig(
        normalization=_decode_normalization(d[:normalization]),
        tokenization=_decode_tokenization(d[:tokenization]),
        transformation=_decode_transformation(d[:transformation]),
        expand_query_synonyms=Bool(get(d, :expand_query_synonyms, false)),
    )
end

# ── save_profile / load_profile ──────────────────────────────────────────────

"""
    save_profile(path::AbstractString, model::VectorModel;
                 synonyms::AbstractDict=Dict{String,Vector{Pair{String,Float32}}}()) -> path

Serializes `model` -- its `voc`'s [`TextConfig`](@ref) and vocabulary counters, its
weighting scheme, and its precomputed `weight` vector -- together with a `synonyms`
network (e.g. as produced by [`LSI.synonyms`](@ref)) into a plain, human-readable JSON
file at `path`. Load it back with [`load_profile`](@ref).

Deliberately NOT a generic object-graph dump (unlike e.g. JLD2): every field is encoded
by hand into a small, versioned schema, so the file is fully inspectable/diffable/portable
and there is nothing pointer- or code-shaped to accidentally serialize.

A `TokenizationConfig` with custom (non-empty) `generators`, or a `transformation` that
isn't `IdentityTokenTransformation`/`IgnoreStopwords`/`SnowballTokenTransformation`/
`ChainTransformation` of those, errors clearly rather than silently mis-saving.
"""
function save_profile(path::AbstractString, model::VectorModel;
                       synonyms::AbstractDict=Dict{String,Vector{Pair{String,Float32}}}())
    voc = model.voc
    data = Dict(
        "format_version" => _PROFILE_FORMAT_VERSION,
        "textconfig" => _encode_textconfig(voc.textconfig),
        "vocabulary" => Dict(
            "tokens" => voc.token,
            "occs" => voc.occs,
            "ndocs" => voc.ndocs,
            "trainsize" => voc.trainsize[],
            "numtokens" => voc.numtokens[],
        ),
        "weighting" => Dict(
            "global_weighting" => _encode_global_weighting(model.global_weighting),
            "local_weighting" => _encode_local_weighting(model.local_weighting),
            "maxoccs" => model.maxoccs,
            "weight" => model.weight,
        ),
        "synonyms" => Dict(tok => [[syn, Float32(dist)] for (syn, dist) in syns] for (tok, syns) in synonyms),
    )

    open(path, "w") do io
        JSON3.write(io, data)
    end

    path
end

"""
    load_profile(path::AbstractString) -> (model::VectorModel, synonyms::Dict{String,Vector{Pair{String,Float32}}})

Reads back a profile written by [`save_profile`](@ref): reconstructs the `Vocabulary`
(rebuilding `token2id` from the stored `tokens` list) and `VectorModel`, and returns the
attached synonym network alongside it, ready to pass straight into e.g.
`TextInvertedFile(model; synonyms, ...)`.

Errors if the file's `transformation` needs `Snowball`/`Languages` to reconstruct
(a `"snowball"` tagged-union entry) and those packages aren't loaded yet.
"""
function load_profile(path::AbstractString)
    data = JSON3.read(read(path, String))
    data[:format_version] == _PROFILE_FORMAT_VERSION ||
        error("unsupported profile format_version: $(data[:format_version]) (expected $_PROFILE_FORMAT_VERSION)")

    textconfig = _decode_textconfig(data[:textconfig])

    vocd = data[:vocabulary]
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

    wd = data[:weighting]
    gw = _decode_global_weighting(String(wd[:global_weighting]))
    lw = _decode_local_weighting(String(wd[:local_weighting]))
    model = VectorModel(gw, lw, voc, Int32(wd[:maxoccs]), Float32.(wd[:weight]))

    synonyms = Dict{String,Vector{Pair{String,Float32}}}(
        String(tok) => [Pair(String(syn), Float32(dist)) for (syn, dist) in syns]
        for (tok, syns) in data[:synonyms]
    )

    model, synonyms
end
