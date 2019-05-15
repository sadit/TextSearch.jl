export TokenData, compute_bow

import Base: +, *, /, ==, length, transpose
import LinearAlgebra: dot
import SimilaritySearch: normalize!, cosine_distance, angle_distance
export BOW, compute_bow

# const BOW = Dict{Symbol,Int}
const BOW = Dict{Symbol,Float64}

"""
    compute_bow(config::TextConfig, text::String, voc::BOW)

Computes a bag of words and the maximum frequency of the bag
"""
function compute_bow(config::TextConfig, text::String, voc::BOW)
    maxfreq::Int = 0
    for token in tokenize(config, text)
        sym = Symbol(token)
        m = floor(Int, get(voc, sym, 0.0)) + 1
        voc[sym] = m
        maxfreq = max(m, maxfreq)
    end

    voc, maxfreq
end


"""
    compute_bow(config::TextConfig, text::String)

Compute a bag of words and the maximum frequency of the bag
"""
compute_bow(config::TextConfig, text::String) = compute_bow(config, text, BOW())

"""
    compute_bow(config::TextConfig, arr::AbstractVector{String})

Computes a bag of word and the maximum frequency of the bag; the input is an array of strings that represent a single object
"""
function compute_bow(config::TextConfig, arr::AbstractVector{String})
    D = BOW()
    maxfreq::Int = 0
    for text in arr
       _, maxfreq = compute_bow(config, text, D)
    end

    D, maxfreq
end

"""
    normalize!(bow::BOW)

Inplace normalization of `bow`
"""
function normalize!(bow::BOW)
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


function normalize!(matrix::AbstractVector{BOW})
    for bow in matrix
        normalize!(bow)
    end
end

function dot(a::BOW, b::BOW)
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

function +(a::BOW, b::BOW)
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

function *(a::BOW, b::BOW)
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

function *(a::BOW, b::F) where F <: Real
    c = copy(a)
    for (k, v) in a
        c[k] = v * b
    end

    c
end

function *(b::F, a::BOW) where F <: Real
    a * b
end

function /(a::BOW, b::F) where F <: Real
    c = copy(a)
    for (k, v) in a
        c[k] = v / b
    end

    c
end


function +(a::BOW, b::F) where F <: Real
    c = copy(a)
    for (k, v) in a
        c[k] = v + b
    end

    c
end

function +(b::F, a::BOW) where F <: Real
    a + b
end

"""
cosine_distance

Computes the cosine_distance between two weighted bags

It supposes that bags are normalized (see `normalize!` function)

"""
function cosine_distance(a::BOW, b::BOW)::Float64
    return 1.0 - dot(a, b)
end

"""
angle_distance

Computes the angle  between two weighted bags

It supposes that all bags are normalized (see `normalize!` function)

"""
function angle_distance(a::BOW, b::BOW)
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

function cosine(a::BOW, b::BOW)::Float64
    return dot(a, b) # * a.invnorm * b.invnorm # it is already normalized
end