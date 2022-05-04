# This file is a part of TextSearch.jl

export vectorize_corpus, vectorize, bow
const BOW = DVEC{UInt32,Int32}

"""
    vectorize(voc::Vocabulary, tokenlist::AbstractVector; bow::BOW=BOW())
    vectorize(voc::Vocabulary, tok::Tokenizer, text::AbstractString; bow::BOW=BOW())

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function vectorize(voc::Vocabulary, tokenlist::AbstractVector; bow::BOW=BOW())
    z = zero(UInt32)
    for token in tokenlist
        tokenID = get(voc, token, z)
        if z != tokenID
            bow[tokenID] = get(bow, tokenID, Int32(0)) + Int32(1)
        end
    end

    bow
end

function vectorize(voc::Vocabulary, tok::Tokenizer, text::AbstractString; bow::BOW=BOW())
    vectorize(voc, tokenize(tok, text); bow)
end

"""
    vectorize(voc::Vocabulary, tok::Tokenizer, messages::AbstractVector; bow::BOW=BOW())

Computes a bag of words from messages
"""
function vectorize(voc::Vocabulary, tok::Tokenizer, messages::AbstractVector; bow::BOW=BOW())
    for text in messages
        empty!(tok)
        tokenize(tok, text)
        vectorize(voc, tok.tokens; bow)
    end

    bow
end

"""
    vectorize_corpus(tok::Tokenizer, tok::Tokenizer, corpus::AbstractVector)

Computes a list of bag of words from a corpus
"""
function vectorize_corpus(voc::Vocabulary, tok::Tokenizer, corpus::AbstractVector; bow::BOW=BOW())
    X = Vector{BOW}(undef, length(corpus))
    for i in eachindex(corpus)
        empty!(tok)
        empty!(bow)
        vectorize(voc, tok, corpus[i]; bow)
        X[i] = copy(bow)
    end

    X
end
