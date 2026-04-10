# This file is a part of TextSearch.jl

import Base: +, -, *, /, ==, zero
using LinearAlgebra, SparseArrays
import LinearAlgebra: dot, norm, normalize!
#import SparseArrays: nnz
using SimilaritySearch
using SimilaritySearch.Dist: NormAngle, NormCosine, Angle, Cosine
import SimilaritySearch: evaluate
export centroid, evaluate, NormAngle, NormCosine, Angle, Cosine, l1norm, l1normalize!

#const Dict{Ti<:Union{Integer,Symbol,String},Tv<:Number} = Dict{Ti,Tv}
#const Dict{Ti,Tv} where Ti where Tv = Dict{Ti<:Union{Integer,Symbol,String},Tv<:Number}
#const Dict{Ti,Tv<:Number} = Dict{Ti,Tv}
#nnz(Dict::Dict) = length(Dict)

function Base.findmax(voc::Dict{I,F}) where {I,F}
    m = typemin(F)
    maxkey = "typemax(I)"
    for (k, v) in voc
        if v >= m
            maxkey = k
            m = v
        end
    end

    (m, maxkey)
end

Base.argmax(voc::Dict) = last(findmax(voc))
Base.maximum(voc::Dict) = maximum(values(voc))

function Base.findmin(voc::Dict{I,F}) where {I,F}
    m = typemax(F)
    mk = typemax(I)
    for (k, v) in voc
        if v <= m
            mk = k
            m = v
        end
    end

    (m, mk)
end

Base.argmin(voc::Dict) = last(findmin(voc))
Base.minimum(voc::Dict) = first(findmin(voc))

"""
    normalize!(bow::Dict)

Inplace normalization of `bow`
"""
function normalize!(bow::Dict)
    s = one(valtype(bow)) / norm(bow)
    for (k, v) in bow
        bow[k] = v * s
    end

    bow
end

# function normalize!(bow::Dict)
#     Tv = valtype(bow)
#     s = 1.0 / norm(bow)
#     for (k, v) in bow
#         bow[k] = round(Tv, v * s)
#     end
# 
#     bow
# end

function normalize!(matrix::AbstractVector{<:Dict})
    for bow in matrix
        normalize!(bow)
    end

    matrix
end

function l1norm(v::AbstractVector{T}) where T
    s = zero(valtype(T))
    @inbounds @simd for i in eachindex(v)
        s += v[i]
    end

    s
end

function l1normalize!(v::AbstractVector{T}) where T
    invl1 = one(valtype(T)) / l1norm(v)

    @inbounds @simd for i in eachindex(v)
        v[i] = v[i] * invl1
    end

    v
end

function l1norm(V::Dict)
    s = zero(valtype(V))

    for (_, v) in V
        s += v[i]
    end

    s
end

function l1normalize!(V::Dict)
    invl1 = one(valtype(V)) / l1norm(V)

    for (k, v) in V
        v[k] = v * invl1
    end

    v
end

"""
    dot(a::Dict, b::Dict)

Computes the dot product for two Dict vectors
"""
function dot(a::Dict, b::Dict)
    if length(b) < length(a)
        a, b = b, a  # a must be the smallest bow
    end
    Tv = valtype(a)
    s = zero(Tv)
    for (k, v) in a
        w = convert(Tv, get(b, k, 0))
        s += v * w
    end

    s
end

"""
    norm(a::Dict)

Computes a normalized Dict vector
"""
function norm(a::Dict)
    s = zero(valtype(a))
    for (_, w) in a
        s = muladd(w, w, s)
    end

    sqrt(s)
end

"""
    zero(::Type{Dict{Ti,Tv}}) where {Ti,Tv}

Creates an empty Dict vector
"""
function zero(::Type{Dict{Ti,Tv}}) where {Ti,Tv}
    Dict{Ti,Tv}()
end

## inplace sum
"""
    add!(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
    add!(a::Dict{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}
    add!(a::Dict{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}

Updates `a` to the sum of `a+b`
"""
function add!(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
    for (k, w) in b
        if w != 0
            a[k] = get(a, k, zero(Tv)) + w
        end
    end

    a
end

function add!(a::Dict{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}
    for (k, w) in zip(b.nzind, b.nzval)
        if w != 0
            a[k] = get(a, k, zero(Tv)) + w
        end
    end

    a
end

function add!(a::Dict{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}
    k, w = b
    a[k] = get(a, k, zero(Tv)) + w
    a
end

"""
    Base.sum(col::AbstractVector{<:Dict})

Computes the sum of the given list of vectors
"""
function Base.sum(col::AbstractVector{<:Dict})
    v = copy(col[1])
    for i in 2:length(col)
        add!(v, col[i])
    end

    v
end


"""
    centroid(cluster::AbstractVector{<:Dict})

Computes a centroid of the given list of Dict vectors
"""
function centroid(cluster::AbstractVector{<:Dict})
    normalize!(sum(cluster))
end

"""
    +(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
    +(a::Dict, b::Pair)

Computes the sum of `a` and `b`
"""
function +(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
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

function +(a::Dict, b::Pair)
    c = copy(a)
    add!(c, b)
end

## definitions for substraction

"""
    -(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}

Substracts of `b` of `a`
"""
function -(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
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
    *(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
    *(a::Dict{K, V}, b::F) where K where {V<:Real} where {F<:Real}

Computes the element-wise product of a and b
"""
function *(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
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

function *(a::Dict{K,V}, b::F) where K where {V<:Real} where {F<:Real}
    c = copy(a)
    for (k, v) in a
        c[k] = convert(V, v * b)
    end

    c
end

function *(b::F, a::Dict) where {F<:Real}
    a * b
end

"""
    /(a::Dict{K, V}, b::F) where K where {V<:Real} where {F<:Real}

Computes the element-wise division of a and b
"""
function /(a::Dict{K,V}, b::F) where K where {V<:Real} where {F<:Real}
    a * (1.0 / b)
end


"""
    evaluate(::NormCosine, a::Dict, b::Dict)::Float64

Computes the cosine distance between two Dict sparse vectors

It supposes that bags are normalized (see `normalize!` function)

"""
function evaluate(::NormCosine, a::Dict, b::Dict)::Float64
    1.0 - dot(a, b)
end

"""
    evaluate(::Cosine, a::Dict, b::Dict)::Float64

Computes the cosine distance between two Dict sparse vectors

"""
function evaluate(::Cosine, a::Dict, b::Dict)::Float64
    1.0 - full_cosine(a, b)
end

const π_2 = π / 2

"""
    evaluate(::NormAngle, a::Dict, b::Dict)::Float64

Computes the angle  between two Dict sparse vectors

It supposes that all bags are normalized (see `normalize!` function)

"""
function evaluate(::NormAngle, a::Dict, b::Dict)::Float64
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
    evaluate(::Angle, a::Dict, b::Dict)::Float64

Computes the angle between two Dict sparse vectors

"""
function evaluate(::Angle, a::Dict, b::Dict)::Float64
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

function full_cosine(a::Dict, b::Dict)::Float64
    return dot(a, b) / (norm(a) * norm(b))
end
