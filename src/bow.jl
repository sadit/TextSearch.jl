# This file is a part of TextSearch.jl

export vectorize_corpus, vectorize

"""
    vectorize(voc::Vocabulary, tokenlist::AbstractVector, bow)
    vectorize(voc::Vocabulary, textconfig::TextConfig, text, buff)
    vectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text, buff)
    vectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text)
    vectorize(voc::Vocabulary, textconfig::TextConfig, text)

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function vectorize(voc::Vocabulary, tokenlist::AbstractVector, bow::BOW)
    for token in tokenlist
        tokenID = get(voc, token, zero(UInt32))
        if zero(UInt32) != tokenID
            bow[tokenID] = get(bow, tokenID, zero(Int32)) + one(Int32)
        end
    end

    bow
end

function vectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text::AbstractString, buff::TextSearchBuffer)
    tokens = tokenize(identity, textconfig, text, buff)
    copy_(vectorize(voc, tokens, buff.bow))
end

function vectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text)
    buff = take!(TEXT_SEARCH_CACHES)
    empty!(buff)
    try
        vectorize(copy_, voc, textconfig, text, buff)
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

"""
    vectorize(voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector)
    vectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector, buff)

Computes a bag of words from messages
"""
function vectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector, buff::TextSearchBuffer)
    empty!(buff.bow)
    for text in messages
        empty!(buff.normtext); empty!(buff.tokens); empty!(buff.unigrams)
        tokens = tokenize(identity, textconfig, text, buff)
        vectorize(voc, tokens, buff.bow)
    end

    copy_(buff.bow)
end

vectorize(voc::Vocabulary, textconfig::TextConfig, messages) = vectorize(copy, voc, textconfig, messages)

"""
    vectorize_corpus(textconfig::TextConfig, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    vectorize_corpus(copy_::Function, voc::Vocabulary, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)

Computes a list of bag of words from a corpus
"""
function vectorize_corpus(copy_::Function, voc::Vocabulary, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    n = length(corpus)
    X = [vectorize(copy_, voc, textconfig, corpus[1])]
    resize!(X, n)
    minbatch = getminbatch(minbatch, n)

    #@batch minbatch=minbatch per=thread 
    Threads.@threads for i in 2:n
        X[i] = vectorize(copy_, voc, textconfig, corpus[i])
    end

    X
end

vectorize_corpus(voc::Vocabulary, textconfig::TextConfig, corpus::AbstractVector; minbatch=0) =
    vectorize_corpus(copy, voc, textconfig, corpus; minbatch)