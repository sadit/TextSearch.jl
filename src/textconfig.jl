# This file is a part of TextSearch.jl

export TextConfig, Skipgram

"""
    Skipgram(qsize, skip)

A skipgram is a kind of tokenization where `qsize` words having `skip` separation are used as a single token.
"""
struct Skipgram
    qsize::Int8
    skip::Int8
end

Base.isless(a::Skipgram, b::Skipgram) = isless((a.qsize, a.skip), (b.qsize, b.skip))
Base.isequal(a::Skipgram, b::Skipgram) = a.qsize == b.qsize && a.skip == b.skip

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
- `tt`: An `AbstractTokenTransformation` struct

Note: If qlist, nlist, and slists are all empty arrays, then it defaults to nlist=[1]
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
    tt::AbstractTokenTransformation = IdentityTokenTransformation()

    function TextConfig(del_diac::Bool, del_dup::Bool, del_punc::Bool, group_num::Bool, group_url::Bool, group_usr::Bool, group_emo::Bool, lc::Bool, collocations::Integer,
            mark_token_type::Bool, qlist::AbstractVector, nlist::AbstractVector, slist::AbstractVector, tt)
        if length(qlist) == length(nlist) == length(slist) == 0
            nlist = [1]
        end
        qlist = sort!(Vector{Int8}(qlist))
        nlist = sort!(Vector{Int8}(nlist))
        slist = sort!(Vector{Skipgram}(slist))

        new(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, collocations, mark_token_type, qlist, nlist, slist, tt)
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
        tt::AbstractTokenTransformation=c.tt
    )

    TextConfig(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, collocations, mark_token_type, qlist, nlist, slist, tt)
end

Base.broadcastable(c::TextConfig) = (c,)
