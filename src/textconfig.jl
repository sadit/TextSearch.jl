# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextConfig, Skipgram

# SKIP_WORDS = set(["…", "..", "...", "...."])


"""
    Skipgram(qsize, skip)

A skipgram is a kind of tokenization where `qsize` words having `skip` separation are used as a single token.
"""
struct Skipgram
    qsize::Int8
    skip::Int8
end

Base.isless(a::Skipgram, b::Skipgram) = isless((a.qsize, a.skip), (b.qsize, b.skip))

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
        qlist::Vector=Int8[],
        nlist::Vector=Int8[1],
        slist::Vector{Skipgram}=Skipgram[]
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
- `qlist`: a list of character q-grams to use
- `nlist`: a list of words n-grams to use
- `slist`: a list of skip-grams tokenizers to use

"""
struct TextConfig
    del_diac::Bool
    del_dup::Bool
    del_punc::Bool
    group_num::Bool
    group_url::Bool
    group_usr::Bool
    group_emo::Bool
    lc::Bool
    qlist::Vector{Int8}
    nlist::Vector{Int8}
    slist::Vector{Skipgram}

    function TextConfig(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, qlist, nlist, slist)
        qlist = sort!(Vector{Int8}(qlist))
        nlist = sort!(Vector{Int8}(nlist))
        slist = sort!(Vector{Skipgram}(slist))

        new(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, qlist, nlist, slist)
    end
end

StructTypes.StructType(::Type{Skipgram}) = StructTypes.Struct()
StructTypes.StructType(::Type{TextConfig}) = StructTypes.Struct()

function TextConfig(;
        del_diac::Bool=true,
        del_dup::Bool=false,
        del_punc::Bool=false,
        group_num::Bool=true,
        group_url::Bool=true,
        group_usr::Bool=false,
        group_emo::Bool=false,
        lc::Bool=true,
        qlist::AbstractVector=[],
        nlist::AbstractVector=[1],
        slist::AbstractVector=[]
    )
 
    TextConfig(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, qlist, nlist, slist)
end

function Base.copy(c::TextConfig;
        del_diac=c.del_diac,
        del_dup=c.del_dup,
        del_punc=c.del_punc,
        group_num=c.group_num,
        group_url=c.group_url,
        group_usr=c.group_usr,
        group_emo=c.group_emo,
        lc=c.lc,
        qlist=c.qlist,
        nlist=c.nlist,
        slist=c.slist
    )
    
    TextConfig(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo,
        lc, qlist, nlist, slist)
end

Base.broadcastable(c::TextConfig) = (c,)