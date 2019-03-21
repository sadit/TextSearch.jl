export TokenData, compute_bow

mutable struct TokenData
    id::Int32
    freq::Int32
    TokenData() = new(0, 0)
    TokenData(a, b) = new(a, b)
end

const UNKNOWN_TOKEN = TokenData(0, 0)

"""
Computes the vocabulary of a single text
"""
function compute_vocabulary(config::TextConfig, text::String, voc::Dict{Symbol,TokenData})
    for token in tokenize(config, text)
        sym = Symbol(token)
        h = get(voc, sym, UNKNOWN_TOKEN)
        if h.freq == 0
            voc[sym] = TokenData(length(voc), 1)
        else
            h.freq += 1
        end
    end

    voc
end

function compute_vocabulary(config::TextConfig, arr::AbstractVector{String}, voc::Dict{Symbol,TokenData})
    for text in arr
       compute_vocabulary(config, text, voc)
    end

    voc
end

"""
Computes a BOW
"""
function compute_bow(config::TextConfig, text::String, voc::Dict{Symbol,Int})
    for token in tokenize(config, text)
        sym = Symbol(token)
        voc[sym] = get(voc, sym, 0) + 1
    end

    voc
end

compute_bow(config::TextConfig, text::String) = compute_bow(config, text, Dict{Symbol,Int}())

function compute_bow(config::TextConfig, arr::AbstractVector{String})
    D = Dict{Symbol,Int}()
    for text in arr
       compute_bow(config, text, D)
    end

    D
end