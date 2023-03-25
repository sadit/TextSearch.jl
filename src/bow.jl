# This file is a part of TextSearch.jl

export vectorize_corpus, vectorize

"""
    vectorize!(bow::BOW, voc::Vocabulary, tokenlist::TokenizedText)
    vectorize!(buff::TextSearchBuffer, voc::Vocabulary, textconfig::TextConfig, text)
    vectorize(voc::Vocabulary, textconfig::TextConfig, messages)

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function vectorize!(bow::BOW, voc::Vocabulary, tokenlist::TokenizedText)
    for token in tokenlist
        tokenID = get(voc, token, zero(UInt32))
        if zero(UInt32) != tokenID
            bow[tokenID] = get(bow, tokenID, zero(Int32)) + one(Int32)
        end
    end

    bow
end

function vectorize_(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text)
    buff = take!(TEXT_SEARCH_CACHES)
    empty!(buff)
    try
        copy_(vectorize!(buff, voc, textconfig, text).bow)
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

"""
    vectorize(voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector)
    vectorize!(buff, voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector)

Computes a bag of words from messages
"""
function vectorize!(buff::TextSearchBuffer, voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector)
    empty!(buff.bow)
    for text in messages
        empty!(buff.normtext); empty!(buff.tokens); empty!(buff.unigrams)
        tokens = tokenize(borrowtokenizedtext, textconfig, text, buff)
        vectorize!(buff.bow, voc, tokens)
    end

    buff
end

function vectorize!(buff::TextSearchBuffer, voc::Vocabulary, textconfig::TextConfig, text::AbstractString)
    tokens = tokenize(borrowtokenizedtext, textconfig, text, buff)
    vectorize!(buff.bow, voc, tokens)
    buff
end

function vectorize!(buff::TextSearchBuffer, voc::Vocabulary, textconfig::TextConfig, tokens::TokenizedText)
    vectorize!(buff.bow, voc, tokens)
    buff
end

vectorize(voc::Vocabulary, textconfig::TextConfig, messages) = vectorize_(copy, voc, textconfig, messages)

"""
    vectorize_corpus(textconfig::TextConfig, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)

Computes a list of bag of words from a corpus
"""
function vectorize_corpus(voc::Vocabulary, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    n = length(corpus)
    X = [vectorize(voc, textconfig, corpus[1])]
    resize!(X, n)
    minbatch = getminbatch(minbatch, n)

    #@batch minbatch=minbatch per=thread 
    Threads.@threads for i in 2:n
        X[i] = vectorize(voc, textconfig, corpus[i])
    end

    X
end

