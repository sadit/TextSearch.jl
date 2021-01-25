# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextConfig, WeightingType, Skipgram

# SKIP_WORDS = set(["…", "..", "...", "...."])

abstract type WeightingType end

struct Skipgram
    qsize::Int8
    skip::Int8
end

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
end

function TextConfig(;
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
    TextConfig(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc,
        eltype(qlist) <: Int8 ? qlist : Vector{Int8}(qlist),
        eltype(nlist) <: Int8 ? nlist : Vector{Int8}(nlist),
        slist)
end

function Base.copy(c;
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
