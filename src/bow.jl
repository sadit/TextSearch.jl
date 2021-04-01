# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export compute_bow_corpus, compute_bow

"""
    compute_bow(tokenlist::AbstractVector{S}, bow::BOW=BOW()) where {S<:Symbol}
    compute_bow(config::TextConfig, text::Symbol, bow::BOW=BOW())
    compute_bow(config::TextConfig, text::AbstractVector, bow::BOW=BOW())
    

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

function compute_bow(config::TextConfig, text::AbstractString, bow::BOW=BOW(), buff::TokenizerBuffer=TokenizerBuffer())
    compute_bow(tokenize(config, text, buff), bow)
end

"""
    compute_bow(config::TextConfig, messages::AbstractVector, bow::BOW=BOW())

Computes a bag of words from messages
"""
function compute_bow(config::TextConfig, messages::AbstractVector, bow::BOW=BOW(), buff::TokenizerBuffer=TokenizerBuffer())
    for text in messages
        empty!(buff)
        tokenize(config, text, buff)
        compute_bow(buff.output, bow)
    end

    bow
end

"""
    compute_bow_corpus(config::TextConfig, corpus::AbstractVector)

Computes a list of bag of words from a corpus
"""
function compute_bow_corpus(config::TextConfig, corpus::AbstractVector, bow::BOW=BOW(), buff::TokenizerBuffer=TokenizerBuffer())
    X = Vector{BOW}(undef, length(corpus))
    for i in eachindex(corpus)
        empty!(buff)
        empty!(bow)
        compute_bow(config, corpus[i], bow, buff)
        X[i] = copy(bow)
    end

    X
end
