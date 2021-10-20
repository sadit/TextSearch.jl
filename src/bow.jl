# This file is a part of TextSearch.jl

export compute_bow_corpus, compute_bow, bow, dvec
const BOW = DVEC{UInt64,Int32}

"""
    compute_bow(tokenlist::AbstractVector{S}, bow::BOW=BOW()) where {S<:Symbol}
    compute_bow(tok::Tokenizer, text::Symbol, bow::BOW=BOW())
    compute_bow(tok::Tokenizer, text::AbstractVector, bow::BOW=BOW())
    

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function compute_bow(tokenlist::AbstractVector, bow::BOW=BOW())
    for sym in tokenlist
        bow[sym] = get(bow, sym, 0) + 1
    end

    bow
end

function compute_bow(tok::Tokenizer, text::AbstractString, bow::BOW=BOW())
    compute_bow(tokenize(tok, text), bow)
end

"""
    compute_bow(messages::AbstractVector, bow::BOW=BOW())

Computes a bag of words from messages
"""
function compute_bow(tok::Tokenizer, messages::AbstractVector, bow::BOW=BOW())
    for text in messages
        empty!(tok)
        tokenize(tok, text)
        compute_bow(tok.tokens, bow)
    end

    bow
end

"""
    compute_bow_corpus(tok::Tokenizer, corpus::AbstractVector)

Computes a list of bag of words from a corpus
"""
function compute_bow_corpus(tok::Tokenizer, corpus::AbstractVector, bow::BOW=BOW())
    X = Vector{BOW}(undef, length(corpus))
    for i in eachindex(corpus)
        empty!(tok)
        empty!(bow)
        compute_bow(tok, corpus[i], bow)
        X[i] = copy(bow)
    end

    X
end


"""
    bow(model::Tokenizer, x::AbstractSparseVector)
    bow(model::Tokenizer, x::DVEC{Ti,Tv}) where {Ti<:Integer,Tv<:Number}
    
Creates a bag of words using the sparse vector `x` and the text model `model`
"""
function bow(m::Tokenizer, x::AbstractSparseVector)
    DVEC{String,eltype{x.nzval}}(decode(m, t) => v for (t, v) in zip(x.nzind, x.nzval))
end

function bow(m::Tokenizer, x::DVEC{Ti,Tv}) where {Ti<:Integer,Tv<:Number}
    DVEC{String,Tv}(decode(m, t) => v for (t, v) in x)
end

"""
    dvec(m::Tokenizer, x::DVEC{Symbol,Tv}, Ti=Int) where {Tv<:Number}

Creates a DVEC sparse vector from a bag of words sparse vector (i.e., with type DVEC{Symbol,Tv}),
using the inverse map `m`
"""
function dvec(m::Tokenizer, x::DVEC{String,Tv}, Ti=Int) where {Tv<:Number}
    DVEC{Ti,Tv}(encode(m, t) => v for (t, v) in x)
end
