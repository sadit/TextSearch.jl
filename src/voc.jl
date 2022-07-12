# This file is a part of TextSearch.jl

export Vocabulary, occs, ndocs, weight, token, vocsize, trainsize, filter_tokens, tokenize_and_append!

struct Vocabulary
    token::Vector{String}
    occs::Vector{Int32}
    ndocs::Vector{Int32}
    weight::Vector{Float32}
    token2id::Dict{String,UInt32}
    corpuslen::Int
end

"""
    Vocabulary(n::Integer)

Creates a `Vocabulary` struct
"""
function Vocabulary(n::Integer) 
    # n == 0 means unknown
    voc = Vocabulary(String[], Int32[], Int32[], Float32[], Dict{String,UInt32}(), n)
    vocsize = ceil(Int, n^0.6)  # approx based on Heaps law
    sizehint!(voc.token, vocsize)
    sizehint!(voc.occs, vocsize)
    sizehint!(voc.ndocs, vocsize)
    sizehint!(voc.weight, vocsize)
    sizehint!(voc.token2id, vocsize)
    voc
end

function locked_tokenize_and_push(voc, textconfig, doc, buff, l)
    empty!(buff)

    for token in tokenize(textconfig, doc, buff)
        lock(l)
        try
            push!(voc, token, 1, 1, 0f0)
        finally
            unlock(l)
        end
    end
end

"""
    Vocabulary(textconfig, corpus; minbatch=0)

Computes a vocabulary from a corpus using the TextConfig `textconfig`
"""
function Vocabulary(textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    voc = Vocabulary(length(corpus))
    tokenize_and_append!(voc, textconfig, corpus; minbatch)
end

"""
    tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0)

Parsers each document in the given corpus and appends each token in corpus to the vocabulary.
"""
function tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0)
    l = Threads.SpinLock()
    n = length(corpus)
    minbatch = getminbatch(minbatch, n)

    Threads.@threads for i in 1:n
        doc = corpus[i]
        
        buff = take!(CACHES)
        try
            if doc isa AbstractString
                locked_tokenize_and_push(voc, textconfig, doc, buff, l)
            else
                for text in doc
                    locked_tokenize_and_push(voc, textconfig, text, buff, l)
                end
            end
        finally
            put!(CACHES, buff)
        end
    end

    voc
end

"""
    Vocabulary(corpus)

Computes a vocabulary from an already tokenized corpus
"""
function Vocabulary(corpus)
    voc = Vocabulary(length(corpus))
    
    for tokens in corpus
        for token in tokens
            push!(voc, token, 1, 1, 0f0)
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
weight(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.weight)) : voc.weight[tokenID]
token(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? "" : voc.token[tokenID]

function Base.push!(voc::Vocabulary, token::String, occs::Integer, ndocs::Integer, weight::Real)
    id = get(voc.token2id, token, zero(UInt32))

    if id == 0
        id = length(voc) + 1
        push!(voc.token, token)
        push!(voc.occs, occs)
        push!(voc.ndocs, ndocs)
        push!(voc.weight, weight)
        voc.token2id[token] = id
    else
        voc.occs[id] += occs
        voc.ndocs[id] += ndocs
        voc.weight[id] = weight
    end

    voc
end

function Base.push!(voc::Vocabulary, token::String; occs::Integer=0, ndocs::Integer=0, weight::Real=0)
    push!(voc, token, occs, ndocs, weight)
end

Base.get(voc::Vocabulary, token::String, default)::UInt32 = get(voc.token2id, token, default)

function Base.getindex(voc::Vocabulary, token::String)
    getindex(voc, get(voc, token, 0))
end

function Base.getindex(voc::Vocabulary, tokenID::Integer)
    id = convert(UInt32, tokenID)

    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), weight=zero(eltype(voc.weight)), token="")
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], weight=voc.weight[id], token=voc.token[id])
    end
end


"""
    filter_tokens(pred::Function, voc::Vocabulary)

Returns a copy of reduced vocabulary based on evaluating `pred` function for each entry in `voc`
"""
function filter_tokens(pred::Function, voc::Vocabulary)
    V = Vocabulary(voc.corpuslen)

    for i in eachindex(voc)
        v = voc[i]
        if pred(v)
            push!(V, v.token, v.occs, v.ndocs, v.weight)
        end
    end

    V
end