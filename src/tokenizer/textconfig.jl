# This file is a part of TextSearch.jl

export TextConfig, alltokengenerators

"""
    TextConfig(;
        del_diac::Bool=true,
        del_dup::Bool=false,
        del_punc::Bool=false,
        group_num::Bool=true,
        group_url::Bool=true,
        group_usr::Bool=false,
        group_emo::Bool=false,
        lc::Bool=true,
        collocations::Int8=0,
        qlist::Vector=Int8[],
        nlist::Vector=Int8[],
        slist::Vector{Skipgram}=Skipgram[],
        mark_token_type::Bool = true
        re_user::Regex=DEFAULT_RE_USER,
        re_url::Regex=DEFAULT_RE_URL,
        re_num::Regex=DEFAULT_RE_NUM,
        emojis::Set{Char}=DEFAULT_EMOJIS,
        generators::Vector{<:AbstractTokenGenerator}=AbstractTokenGenerator[],
        tt=IdentityTokenTransformation()
    )

Defines a preprocessing and tokenization pipeline

- `del_diac`: indicates if diacritic symbols should be removed
- `del_dup`: indicates if duplicate contiguous symbols must be replaced for a single symbol
- `del_punc`: indicates if punctuaction symbols must be removed
- `group_num`: indicates if numbers should be grouped _num
- `group_url`: indicates if urls should be grouped as _url
- `group_usr`: indicates if users (@usr) should be grouped as _usr
- `group_emo`: indicates if emojis should be grouped as _emo
- `lc`: indicates if the text should be normalized to lower case
- `collocations`: window to expand collocations as tokens, please take into account that:
  - 0 => disables collocations
  - 1 => will compute words (ignored in favor of use typical unigrams)
  - 2 => will compute bigrams (don't use this, but not disabled)
  - 3 <= typical values
- `qlist`: a list of character q-grams to use
- `nlist`: a list of words n-grams to use
- `slist`: a list of skip-grams tokenizers to use
- `mark_token_type`: each token is `marked` with its type (qgram, skipgram, nword) when is true.
- `re_user`, `re_url`, `re_num`: the regexes used to detect `@user` mentions, URLs, and
  numbers when their corresponding `group_*` flag is set (see [`normalize_text`](@ref)).
  Override them to customize detection (e.g. for a different language or domain).
- `emojis`: the set of emoji characters grouped when `group_emo` is set (see [`isemoji`](@ref)).
- `generators`: extra [`AbstractTokenGenerator`](@ref)s to run in addition to the ones
  `qlist`/`nlist`/`slist`/`collocations` build; this is the extension point for adding
  new kinds of tokens without needing a new `TextConfig` keyword argument (see
  [`alltokengenerators`](@ref)).
- `tt`: An `AbstractTokenTransformation` struct

Note: If qlist, nlist, slist, and generators are all empty, then it defaults to nlist=[1]

# Example

```julia
julia> cfg = TextConfig(nlist=[1], qlist=[3]);

julia> collect(tokenize(cfg, "cats"))
[" ca\tq", "cat\tq", "ats\tq", "ts \tq", "cats"]
```
"""
Base.@kwdef struct TextConfig
    del_diac::Bool  = true
    del_dup::Bool   = false
    del_punc::Bool  = false
    group_num::Bool = true
    group_url::Bool = true
    group_usr::Bool = false
    group_emo::Bool = false
    lc::Bool        = true
    collocations::Int8 = 0
    mark_token_type::Bool = true
    qlist::Vector{Int8} = Int8[]
    nlist::Vector{Int8} = Int8[]
    slist::Vector{Skipgram} = Skipgram[]
    re_user::Regex = DEFAULT_RE_USER
    re_url::Regex = DEFAULT_RE_URL
    re_num::Regex = DEFAULT_RE_NUM
    emojis::Set{Char} = DEFAULT_EMOJIS
    generators::Vector{AbstractTokenGenerator} = AbstractTokenGenerator[]
    tt::AbstractTokenTransformation = IdentityTokenTransformation()

    function TextConfig(del_diac::Bool, del_dup::Bool, del_punc::Bool, group_num::Bool, group_url::Bool, group_usr::Bool, group_emo::Bool, lc::Bool, collocations::Integer,
            mark_token_type::Bool, qlist::AbstractVector, nlist::AbstractVector, slist::AbstractVector,
            re_user::Regex, re_url::Regex, re_num::Regex, emojis::Set{Char}, generators::AbstractVector, tt)
        if length(qlist) == length(nlist) == length(slist) == length(generators) == 0
            nlist = [1]
        end
        qlist = sort!(Vector{Int8}(qlist))
        nlist = sort!(Vector{Int8}(nlist))
        slist = sort!(Vector{Skipgram}(slist))
        generators = Vector{AbstractTokenGenerator}(generators)

        new(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, collocations, mark_token_type,
            qlist, nlist, slist, re_user, re_url, re_num, emojis, generators, tt)
    end
end

_show_field(io, f::Symbol, v::Set{Char}) = println(io, f, ": ", length(v), " emoji chars")
_show_field(io, f::Symbol, v) = println(io, f, ": ", v)

function Base.show(io::IO, c::TextConfig; prefix="", indent="  ")
    println(io, prefix, "TextConfig:")
    for f in fieldnames(TextConfig)
        print(io, prefix, indent)
        _show_field(io, f, getfield(c, f))
    end
end

function TextConfig(c::TextConfig;
        del_diac::Bool=c.del_diac,
        del_dup::Bool=c.del_dup,
        del_punc::Bool=c.del_punc,
        group_num::Bool=c.group_num,
        group_url::Bool=c.group_url,
        group_usr::Bool=c.group_usr,
        group_emo::Bool=c.group_emo,
        lc::Bool=c.lc,
        collocations=c.collocations,
        mark_token_type=c.mark_token_type,
        qlist=c.qlist,
        nlist=c.nlist,
        slist=c.slist,
        re_user::Regex=c.re_user,
        re_url::Regex=c.re_url,
        re_num::Regex=c.re_num,
        emojis::Set{Char}=c.emojis,
        generators::AbstractVector=c.generators,
        tt::AbstractTokenTransformation=c.tt
    )

    TextConfig(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, collocations, mark_token_type,
        qlist, nlist, slist, re_user, re_url, re_num, emojis, generators, tt)
end

Base.broadcastable(c::TextConfig) = (c,)

"""
    alltokengenerators(cfg::TextConfig)::Vector{AbstractTokenGenerator}

Builds the full, ordered list of [`AbstractTokenGenerator`](@ref)s `cfg` runs: the
built-in ones implied by `cfg.qlist`/`cfg.nlist`/`cfg.slist`/`cfg.collocations`,
followed by `cfg.generators` (any extra/custom generators). Called once per
[`tokenize`](@ref) invocation.

# Example

```julia
julia> alltokengenerators(TextConfig(nlist=[1, 2]))
2-element Vector{AbstractTokenGenerator}:
 UnigramGenerator()
 NWordGenerator(2)
```
"""
function alltokengenerators(cfg::TextConfig)
    gens = AbstractTokenGenerator[]

    for q in cfg.qlist
        push!(gens, QGramGenerator(q))
    end

    for q in cfg.nlist
        push!(gens, q == 1 ? UnigramGenerator() : NWordGenerator(q))
    end

    for s in cfg.slist
        push!(gens, SkipgramGenerator(s))
    end

    cfg.collocations > 1 && push!(gens, CollocationGenerator(cfg.collocations))
    append!(gens, cfg.generators)
    gens
end
