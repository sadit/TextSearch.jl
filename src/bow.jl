# This file is a part of TextSearch.jl

export vectorize_corpus, vectorize, bow

"""
    vectorize(voc::Vocabulary, tokenlist::AbstractVector, buff=BOW())
    vectorize(voc::Vocabulary, tok::Tokenizer, text::AbstractString, buff=BOW())

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function vectorize(voc::Vocabulary, tokenlist::AbstractVector, bow=BOW())
    z = zero(UInt32)
    for token in tokenlist
        tokenID = get(voc, token, z)
        if z != tokenID
            bow[tokenID] = get(bow, tokenID, zero(Int32)) + one(Int32)
        end
    end

    bow
end

function vectorize(voc::Vocabulary, tok::Tokenizer, text::AbstractString, buff)
    tokens = tokenize(tok, text)
    vectorize(voc, tokens, buff.bow)
end

function vectorize(voc::Vocabulary, tok::Tokenizer, text::AbstractString)
    buff = take!(CACHES)
    empty!(buff)
    try
        copy(vectorize(voc, tok, text, buff))
    finally
        put!(CACHES, buff)
    end
end

"""
    vectorize(voc::Vocabulary, tok::Tokenizer, messages::AbstractVector)

Computes a bag of words from messages
"""
function vectorize(voc::Vocabulary, tok::Tokenizer, messages::AbstractVector, buff)
    empty!(buff.bow)
    for text in messages
        empty!(buff.normtext); empty!(buff.tokens); empty!(buff.unigrams); 
        tokens = tokenize(tok, text)
        vectorize(voc, tokens, buff.bow)
    end

    buff.bow
end

function vectorize(voc::Vocabulary, tok::Tokenizer, messages::AbstractVector)
    buff = take!(CACHES)
    try
        copy(vectorize(voc, tok, messages, buff))
    finally
        put!(CACHES, buff)
    end
end

"""
    vectorize_corpus(tok::Tokenizer, tok::Tokenizer, corpus::AbstractVector)

Computes a list of bag of words from a corpus
"""
function vectorize_corpus(voc::Vocabulary, tok::Tokenizer, corpus::AbstractVector)
    n = length(corpus)
    X = Vector{BOW}(undef, n)
    minbatch = getminbatch(0, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        X[i] = vectorize(voc, tok, corpus[i])
    end

    X
end
