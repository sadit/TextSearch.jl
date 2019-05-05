export TokenData, compute_bow

import Base: +, *, ==, length, transpose
import LinearAlgebra: dot
import SimilaritySearch: normalize!, cosine_distance, angle_distance

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

"""
    normalize!(bow::Dict{Symbol, Float64})

Inplace normalization of `bow`
"""
function normalize!(bow::Dict{Symbol, Float64})
    s = 0.0
    for w in values(bow)
        s += w * w
    end

    s = 1.0 / sqrt(s)
    for (k, v) in bow
        bow[k] = v * s
    end

    bow
end


function normalize!(matrix::AbstractVector{Dict{Symbol, Float64}})
    for bow in matrix
        normalize!(bow)
    end
end

function dot(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})
    if length(b) < length(a)
        a, b = b, a  # a must be the smallest bow
    end
    
    s = 0.0
    for (k, v) in a
        w = get(b, k, 0.0)
        s += v * w
    end

    s
end

function +(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})
    if length(a) < length(b) 
        a, b = b, a  # a must be the largest bow
    end
    
    c = copy(a)
    for k in keys(b)
        w = get(b, k, 0.0)
        if w != 0.0
            c[k] += b[k]
        end
    end

    c
end

function *(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})
    if length(b) < length(a)
        a, b = b, a  # a must be the smallest bow
    end
    
    c = copy(a)
    for k in keys(a)
        w = get(b, k, 0.0)
        if w == 0.0
            delete!(c, k)
        else
            c[k] *= w
        end
    end

    c
end

"""
cosine_distance

Computes the cosine_distance between two weighted bags

It supposes that bags are normalized (see `normalize!` function)

"""
function cosine_distance(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})::Float64
    return 1.0 - dot(a, b)
end

"""
angle_distance

Computes the angle  between two weighted bags

It supposes that all bags are normalized (see `normalize!` function)

"""
function angle_distance(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})
    d = dot(a, b)

    if d <= -1.0
        return π
    elseif d >= 1.0
        return 0.0
    elseif d == 0  # turn around for zero vectors, in particular for denominator=0
        return π_2
    else
        return acos(d)
    end
end

function cosine(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})::Float64
    return dot(a, b) # * a.invnorm * b.invnorm # it is already normalized
end