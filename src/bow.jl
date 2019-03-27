export TokenData, compute_bow


"""
    mutable struct TokenData

Stores an identifier and its frequency
"""
mutable struct TokenData
    id::Int32
    freq::Int32
    TokenData() = new(0, 0)
    TokenData(a, b) = new(a, b)
end

const UNKNOWN_TOKEN = TokenData(0, 0)

"""
    compute_vocabulary(config::TextConfig, text::String, voc::Dict{Symbol,TokenData})

Computes the vocabulary and the maximum term's frequency of a single text
"""
function compute_vocabulary(config::TextConfig, text::String, voc::Dict{Symbol,TokenData})
    maxfreq = 1
    for token in tokenize(config, text)
        sym = Symbol(token)
        h = get(voc, sym, UNKNOWN_TOKEN)
        if h.freq == 0
            m = 1
            voc[sym] = TokenData(length(voc), 1)
        else
            h.freq += 1
            m = h.freq
        end

        maxfreq = max(m, maxfreq)
    end

    voc, maxfreq
end

"""

    compute_vocabulary(config::TextConfig, arr::AbstractVector{String}, voc::Dict{Symbol,TokenData})

Computes the vocabulary of an item and the maximum term's frequency; an item item is represented as an array of texts
"""
function compute_vocabulary(config::TextConfig, arr::AbstractVector{String}, voc::Dict{Symbol,TokenData})
    maxfreq = 0
    for text in arr
       _, maxfreq = compute_vocabulary(config, text, voc)
    end

    voc, maxfreq
end

"""
    compute_bow(config::TextConfig, text::String, voc::Dict{Symbol,Int})

Computes a bag of words and the maximum frequency of the bag
"""
function compute_bow(config::TextConfig, text::String, voc::Dict{Symbol,Int})
    maxfreq = 0
    for token in tokenize(config, text)
        sym = Symbol(token)
        m = get(voc, sym, 0) + 1
        voc[sym] = m
        maxfreq = max(m, maxfreq)
    end

    voc, maxfreq
end


"""
    compute_bow(config::TextConfig, text::String)

Compute a bag of words and the maximum frequency of the bag
"""
compute_bow(config::TextConfig, text::String) = compute_bow(config, text, Dict{Symbol,Int}())

"""
    compute_bow(config::TextConfig, arr::AbstractVector{String})

Computes a bag of word and the maximum frequency of the bag; the input is an array of strings that represent a single object
"""
function compute_bow(config::TextConfig, arr::AbstractVector{String})
    D = Dict{Symbol,Int}()
    maxfreq = 0
    for text in arr
       _, maxfreq = compute_bow(config, text, D)
    end

    D, maxfreq
end