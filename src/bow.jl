# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export compute_bow_corpus, compute_bow

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
        compute_bow(tok.output, bow)
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
