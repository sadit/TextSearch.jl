# This file is a part of TextSearch.jl

export Vocabulary, occs, ndocs, weight, token, vocsize, trainsize

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

"""
    Vocabulary(tok, corpus)

Computes a vocabulary from a corpus using the tokenizer `tok`
"""
function Vocabulary(tok::Tokenizer, corpus)
    voc = Vocabulary(length(corpus))

    for doc in corpus
        if doc isa AbstractString
            for token in tokenize(tok, doc)
                push!(voc, token, 1, 1, 0f0)
            end
        else
            for text in doc
                for token in tokenize(tok, text)
                    push!(voc, token, 1, 1, 0f0)
                end
    
            end
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

Base.get(voc::Vocabulary, token::String, default)::UInt32 = get(voc.token2id, token, default)

function Base.getindex(voc::Vocabulary, token::String)
    getindex(voc, get(voc, token, 0))
end

function Base.getindex(voc::Vocabulary, tokenID::Integer)
    id = convert(UInt32, tokenID)

    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), weight=zero(eltype(voc.weight)))
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], weight=voc.weight[id])
    end
end