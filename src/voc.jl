# This file is a part of TextSearch.jl

export Vocabulary

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
        for token in tokenize(tok, doc)
            push!(voc, token, 1, 1, 0f0)
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

function Base.push!(voc::Vocabulary, token::String, occs::Integer, ndocs::Integer, weight::Real)
    id = get(voc.token2id, token, UInt32(0))
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

Base.get(voc::Vocabulary, token::String, default) = get(voc.token2id, token, default)
