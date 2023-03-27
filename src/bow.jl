# This file is a part of TextSearch.jl

export bagofwords_corpus, bagofwords

"""
    bagofwords!(bow::BOW, voc::Vocabulary, tokenlist::TokenizedText)
    bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, text)
    bagofwords(voc::Vocabulary, messages)

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function bagofwords!(bow::BOW, voc::Vocabulary, tokenlist::TokenizedText)
    for token in tokenlist
        tokenID = token2id(voc, token)
        if zero(UInt32) != tokenID
            bow[tokenID] = get(bow, tokenID, zero(Int32)) + one(Int32)
        end
    end

    bow
end

function bagofwords_(copy_::Function, voc::Vocabulary, text)
    buff = take!(TEXT_SEARCH_CACHES)
    empty!(buff)
    try
        copy_(bagofwords!(buff, voc, text).bow)
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

"""
    bagofwords(voc::Vocabulary, messages::AbstractVector)
    bagofwords!(buff, voc::Vocabulary, messages::AbstractVector)

Computes a bag of words from messages
"""
function bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, messages::AbstractVector)
    empty!(buff.bow)
    for text in messages
        empty!(buff.normtext); empty!(buff.tokens); empty!(buff.unigrams)
        tokens = tokenize(borrowtokenizedtext, voc.textconfig, text, buff)
        bagofwords!(buff.bow, voc, tokens)
    end

    buff
end

function bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, text::AbstractString)
    tokens = tokenize(borrowtokenizedtext, voc.textconfig, text, buff)
    bagofwords!(buff.bow, voc, tokens)
    buff
end

function bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, tokens::TokenizedText)
    bagofwords!(buff.bow, voc, tokens)
    buff
end

bagofwords(voc::Vocabulary, messages) = bagofwords_(copy, voc, messages)
bagofwords(voc::Vocabulary, messages::BOW) = messages

"""
    bagofwords_corpus(voc::Vocabulary, corpus::AbstractVector; minbatch=0)

Computes a list of bag of words from a corpus
"""
bagofwords_corpus(voc::Vocabulary, corpus::AbstractVector{BOW}; minbatch=0) = corpus
function bagofwords_corpus(voc::Vocabulary, corpus::AbstractVector; minbatch=0)
    n = length(corpus)
    X = [bagofwords(voc, corpus[1])]
    resize!(X, n)
    minbatch = getminbatch(minbatch, n)

    #@batch minbatch=minbatch per=thread 
    Threads.@threads for i in 2:n
        X[i] = bagofwords(voc, corpus[i])
    end

    X
end

