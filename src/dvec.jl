# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: +, -, *, /, ==, transpose, zero
using LinearAlgebra, SparseArrays
import LinearAlgebra: dot, norm, normalize!
import SparseArrays: nnz
using SimilaritySearch
import SimilaritySearch: evaluate
export centroid, evaluate, NormalizedAngleDistance, NormalizedCosineDistance, AngleDistance, NormalizedAngleDistance

const DVEC{Ti,Tv<:Number} = Dict{Ti,Tv}
const BOW = DVEC{UInt64,Int32}
const SVEC = DVEC{UInt64,Float32}

export DVEC, SVEC, BOW
nnz(dvec::DVEC) = length(dvec)

"""
    Base.maximum(voc::DVEC)

Computes the maximum (weight) value in a BOW
"""
function Base.maximum(voc::DVEC)
    m = 0
    for v in values(voc)
        if v > m
            m = v
        end
    end

    m
end

"""
    normalize!(bow::DVEC)

Inplace normalization of `bow`
"""
function normalize!(bow::DVEC{Ti,Tv}) where {Ti,Tv<:AbstractFloat}
    s = 1.0 / norm(bow)
    for (k, v) in bow
        bow[k] = convert(Tv, v * s)
    end

    bow
end

function normalize!(bow::DVEC{Ti,Tv}) where {Ti,Tv<:Integer}
    s = 1.0 / norm(bow)
    for (k, v) in bow
        bow[k] = round(Tv, v * s)
    end

    bow
end

function normalize!(matrix::AbstractVector{DVEC})
    for bow in matrix
        normalize!(bow)
    end

    matrix
end

"""
    dot(a::DVEC, b::DVEC)::Float64 

Computes the dot product for two DVEC vectors
"""
function dot(a::DVEC, b::DVEC)::Float64 
    if length(b) < length(a)
        a, b = b, a  # a must be the smallest bow
    end
    
    s = 0.0
    for (k, v) in a
        w = convert(Float64, get(b, k, 0))
        s += v * w
    end

    s
end

"""
    norm(a::DVEC)::Float64

Computes a normalized DVEC vector
"""
function norm(a::DVEC)::Float64
    s = 0.0
    for w in values(a)
        s += w * w
    end

    sqrt(s)
end

"""
    zero(::Type{DVEC{Ti,Tv}}) where {Ti,Tv}

Creates an empty DVEC vector
"""
function zero(::Type{DVEC{Ti,Tv}}) where {Ti,Tv}
    DVEC{Ti,Tv}()
end

## inplace sum
"""
    add!(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    add!(a::DVEC{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}
    add!(a::DVEC{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}

Updates `a` to the sum of `a+b`
"""
function add!(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    for (k, w) in b
        if w != 0
            a[k] = get(a, k, zero(Tv)) + w
        end
    end

    a
end

function add!(a::DVEC{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}
    for (k, w) in zip(b.nzind, b.nzval)
        if w != 0
            a[k] = get(a, k, zero(Tv)) + w
        end
    end

    a
end

function add!(a::DVEC{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}
    k, w = b
    a[k] = get(a, k, zero(Tv)) + w
    a
end

"""
    Base.sum(col::AbstractVector{T}) where {T<:DVEC}

Computes the sum of the given list of vectors
"""
function Base.sum(col::AbstractVector{T}) where {T<:DVEC}
    v = copy(col[1])
    for i in 2:length(col)
        add!(v, col[i])
    end

    v
end


"""
    centroid(cluster::AbstractVector{<:DVEC})

Computes a centroid of the given list of DVEC vectors
"""
function centroid(cluster::AbstractVector{<:DVEC})
    normalize!(sum(cluster))
end

"""
    +(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    +(a::DVEC, b::Pair)

Computes the sum of `a` and `b`
"""
function +(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    if length(a) < length(b) 
        a, b = b, a  # a must be the largest bow
    end
    
    c = copy(a)
    for (k, w) in b
        if w != 0
            c[k] = get(c, k, zero(Tv)) + w 
        end
    end

    c
end

function +(a::DVEC, b::Pair)
    c = copy(a)
    add!(c, b)
end

## definitions for substraction

"""
    -(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}

Substracts of `b` of `a`
"""
function -(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    c = copy(a)
    for (k, w) in b
        if w != 0
            c[k] = get(c, k, zero(Tv)) - w 
        end
    end

    c
end

## definitions for product
"""
    *(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    *(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}

Computes the element-wise product of a and b
"""
function *(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}
    if length(b) < length(a)
        a, b = b, a  # a must be the smallest bow
    end
    
    c = copy(a)
    for k in keys(a)
        w = get(b, k, zero(Tv))
        if w == 0
            delete!(c, k)
        else
            c[k] = convert(Tv, c[k] * w)
        end
    end

    c
end

function *(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}
    c = copy(a)
    for (k, v) in a
        c[k] = convert(V, v * b)
    end

    c
end

function *(b::F, a::DVEC) where {F<:Real}
    a * b
end

"""
    /(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}

Computes the element-wise division of a and b
"""
function /(a::DVEC{K,V}, b::F) where K where {V<:Real} where {F<:Real}
    a * (1.0/b)
end


"""
    evaluate(::NormalizedCosineDistance, a::DVEC, b::DVEC)::Float64

Computes the cosine distance between two DVEC sparse vectors

It supposes that bags are normalized (see `normalize!` function)

"""
function evaluate(::NormalizedCosineDistance, a::DVEC, b::DVEC)::Float64
    1.0 - dot(a, b)
end

"""
    evaluate(::CosineDistance, a::DVEC, b::DVEC)::Float64

Computes the cosine distance between two DVEC sparse vectors

"""
function evaluate(::CosineDistance, a::DVEC, b::DVEC)::Float64
    1.0 - full_cosine(a, b)
end

const π_2 = π / 2

"""
    evaluate(::NormalizedAngleDistance, a::DVEC, b::DVEC)::Float64

Computes the angle  between two DVEC sparse vectors

It supposes that all bags are normalized (see `normalize!` function)

"""
function evaluate(::NormalizedAngleDistance, a::DVEC, b::DVEC)::Float64
    d = dot(a, b)

    if d <= -1.0
        π
    elseif d >= 1.0
        0.0
    elseif d == 0  # turn around for zero vectors, in particular for denominator=0
        π_2
    else
        acos(d)
    end
end

"""
    evaluate(::AngleDistance, a::DVEC, b::DVEC)::Float64

Computes the angle between two DVEC sparse vectors

"""

function evaluate(::AngleDistance, a::DVEC, b::DVEC)::Float64
    d = full_cosine(a, b)

    if d <= -1.0
        π
    elseif d >= 1.0
        0.0
    elseif d == 0  # turn around for zero vectors, in particular for denominator=0
        π_2
    else
        acos(d)
    end
end


function full_cosine(a::DVEC, b::DVEC)::Float64
    return dot(a, b) / (norm(a) * norm(b))
end
