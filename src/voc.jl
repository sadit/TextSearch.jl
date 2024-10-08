# This file is a part of TextSearch.jl

export Vocabulary, AbstractTokenLookup, TokenLookup, occs, ndocs, token, vocsize, trainsize, filter_tokens, tokenize_and_append!, merge_voc, update_voc!, vocabulary_from_thesaurus, token2id, 
       encode, decode, table

abstract type AbstractTokenLookup end

struct Vocabulary{Lookup<:AbstractTokenLookup}
    lookup::Lookup
    textconfig::TextConfig
    token::Vector{String}
    occs::Vector{Int32}
    ndocs::Vector{Int32}
    token2id::Dict{String,UInt32}
    corpuslen::Int
end

struct TokenLookup <: AbstractTokenLookup
end

token2id(voc::Vocabulary{TokenLookup}, tok::AbstractString) = get(voc.token2id, tok, zero(UInt32))

function Vocabulary(voc::Vocabulary; lookup=voc.lookup, textconfig=voc.textconfig, token=voc.token, occs=voc.occs, ndocs=voc.ndocs, token2id=voc.token2id, corpuslen=voc.corpuslen)
    Vocabulary(lookup, textconfig, token, occs, ndocs, token2id, corpuslen)
end

function decode(voc::Vocabulary, bow::Dict)
    Dict(voc.token[k] => v for (k, v) in bow)
end

function encode(voc::Vocabulary, bow::Dict)
    Dict(token2id(voc, k) => v for (k, v) in bow)
end

function table(voc::Vocabulary, TableConstructor)
    TableConstructor(; voc.token, voc.ndocs, voc.occs)
end

function vocabulary_from_thesaurus(textconfig::TextConfig, tokens::AbstractVector)
    n = length(tokens)
    voc = Vocabulary(textconfig, n)
    for t in tokens
        push_token!(voc, t, 1, 1)
    end

    voc
end

"""
    Vocabulary(textconfig::TextConfig, n::Integer)

Creates a `Vocabulary` struct
"""
function Vocabulary(lookup::AbstractTokenLookup, textconfig::TextConfig, n::Integer)
    # n == 0 means unknown
    voc = Vocabulary(lookup, textconfig, String[], Int32[], Int32[], Dict{String,UInt32}(), n)
    vocsize = ceil(Int, n^0.6)  # approx based on Heaps law
    sizehint!(voc.token, vocsize)
    sizehint!(voc.occs, vocsize)
    sizehint!(voc.ndocs, vocsize)
    sizehint!(voc.token2id, vocsize)
    voc 
end

Vocabulary(textconfig::TextConfig, n::Integer) = Vocabulary(TokenLookup(), textconfig, n)

"""
    Vocabulary(textconfig, corpus; minbatch=0)

Computes a vocabulary from a corpus using the TextConfig `textconfig`.
"""
function vocab_from_small_collection(textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    voc = Vocabulary(TokenLookup(), textconfig, length(corpus))
    tokenize_and_append!(voc, corpus; minbatch)
    voc
end

function Vocabulary(textconfig::TextConfig, corpusgenerator::Union{Base.EachLine,Base.Generator,AbstractVector}; minbatch=0, buffsize=2^16, verbose=true)
    if corpusgenerator isa AbstractVector && length(corpusgenerator) <= buffsize
        return vocab_from_small_collection(textconfig, corpusgenerator; minbatch)
    end

    voc = Vocabulary(TokenLookup(), textconfig, 0)
    len = 0
    corpus = []
    sizehint!(corpus, buffsize)
    @showprogress dt=1 enabled=verbose desc="vocabulary:" for doc in corpusgenerator
        push!(corpus, doc)

        if length(corpus) == buffsize
            # verbose && (@info "computing vocabulary -- advance: $len - buffsize: $buffsize")
            len += buffsize
            tokenize_and_append!(voc, corpus; minbatch)
            empty!(corpus) 
        end 
    end

    if length(corpus) > 0
        len += length(corpus)
        tokenize_and_append!(voc, corpus; minbatch)
    end

    Vocabulary(voc; corpuslen=len)
end

function locked_tokenize_and_push(voc, doc, buff, l)
    empty!(buff)

    for token in tokenize(borrowtokenizedtext, voc.textconfig, doc, buff)
        id = 0
        lock(l)
        try
            id = push_token!(voc, token, 1, 0)
        finally
            unlock(l)
            buff.bow[id] = 1
        end
    end

    lock(l)
    try
        for id in keys(buff.bow)
            voc.ndocs[id] += 1
        end
    finally
        unlock(l)
    end
end

"""
    tokenize_and_append!(voc::Vocabulary, corpus; minbatch=0)

Parse each document in the given corpus and appends each token to the vocabulary.
"""
function tokenize_and_append!(voc::Vocabulary, corpus; minbatch=0)
    l = Threads.SpinLock()
    n = length(corpus)
    minbatch = getminbatch(minbatch, n)
 
    Threads.@threads for i in 1:n # @batch per=thread minbatch=minbatch for i in 1:n
        doc = corpus[i]
        buff = take!(TEXT_SEARCH_CACHES)

        try
            if doc isa AbstractVector
                for text in doc
                    locked_tokenize_and_push(voc, text, buff, l)
                end
            else # if doc isa AbstractString
                locked_tokenize_and_push(voc, doc, buff, l)
            end
        finally
            put!(TEXT_SEARCH_CACHES, buff)
        end
    end

    voc
end

Base.length(voc::Vocabulary) = length(voc.occs)
Base.eachindex(voc::Vocabulary) = eachindex(voc.occs)
vocsize(voc::Vocabulary) = length(voc)
trainsize(voc::Vocabulary) = voc.corpuslen
ndocs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.ndocs)) : voc.ndocs[tokenID]
occs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.occs)) : voc.occs[tokenID]
token(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? "" : voc.token[tokenID]

@inline occs(voc::Vocabulary) = voc.occs
@inline ndocs(voc::Vocabulary) = voc.ndocs
@inline token(voc::Vocabulary) = voc.token

function push_token!(voc::Vocabulary, token, occs::Integer, ndocs::Integer)
    id = token2id(voc, token)

    if id == 0
        id = length(voc) + 1
        push!(voc.token, token)
        push!(voc.occs, occs)
        push!(voc.ndocs, ndocs)
        voc.token2id[token] = id
    else
        voc.occs[id] += occs
        voc.ndocs[id] += ndocs
    end

    id
end

function push_token!(voc::Vocabulary, token; occs::Integer=0, ndocs::Integer=0)
    push_token!(voc, token, occs, ndocs)
end

function append_tokens!(voc::Vocabulary, tokens; occs::Integer=0, ndocs::Integer=0)
    for token in tokens
        push_token!(voc, token, occs, ndocs)
    end
end

itertokenid(idlist::AbstractVector) = idlist 
itertokenid(idlist::AbstractVector{IdWeight}) = (p.id for p in idlist) 
itertokenid(idlist::AbstractVector{IdIntWeight}) = (p.id for p in idlist) 
itertokenid(idlist::AbstractVector{<:NamedTuple}) = (p.id for p in idlist) 
itertokenid(idlist::Dict) = keys(idlist) 
itertokenid(idlist::KnnResult) = IdView(idlist)

function Base.getindex(voc::Vocabulary, idlist)
    [voc[i] for i in itertokenid(idlist)]
end

Base.getindex(voc::Vocabulary, token::AbstractString) = voc[get(voc.token2id, token, 0)]

function Base.getindex(voc::Vocabulary, tokenID::Integer)
    id = convert(UInt32, tokenID)

    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), token="")
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], token=voc.token[id])
    end
end

