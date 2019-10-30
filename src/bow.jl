import Base: +, -, *, /, ==, transpose, zero
import LinearAlgebra: dot, norm, normalize!
import SimilaritySearch: cosine_distance, angle_distance
export DBOW, BOW, compute_bow # add!

const DBOW{Ti,Tv<:Real} = Dict{Ti,Tv}
#const BOW = Dict{Symbol,Int}
const BOW = DBOW{Symbol,Int}

"""
    compute_bow(tokenlist::AbstractVector{Symbol}, voc::BOW)::Tuple{BOW,Float64}

Updates a BOW using the given list of tokens
"""
function compute_bow(tokenlist::AbstractVector{Symbol}, voc::BOW)
    maxfreq = 0
    for sym in tokenlist
        m = get(voc, sym, 0) + 1
        voc[sym] = m
        maxfreq = max(m, maxfreq)
    end

    voc, maxfreq
end

# these are needed to call `compute_bow` for symbol's list but also for simplicity of the API
compute_bow(tokenlist::AbstractVector{Symbol}) = compute_bow(tokenlist, BOW())
## 
"""
    normalize!(bow::DBOW)

Inplace normalization of `bow`
"""
function normalize!(bow::DBOW{Ti,Tv}) where {Ti,Tv<:AbstractFloat}
    s = 1.0 / norm(bow)
    for (k, v) in bow
        bow[k] = convert(Tv, v * s)
    end

    bow
end

function normalize!(bow::DBOW{Ti,Tv}) where {Ti,Tv<:Integer}
    s = 1.0 / norm(bow)
    for (k, v) in bow
        bow[k] = round(T, v * s)
    end

    bow
end

function normalize!(matrix::AbstractVector{DBOW})
    for bow in matrix
        normalize!(bow)
    end
end

function dot(a::DBOW, b::DBOW)
    if length(b) < length(a)
        a, b = b, a  # a must be the smallest bow
    end
    
    s = 0.0
    for (k, v) in a
        w = get(b, k, 0.0)::Float64
        s += v * w
    end

    s
end

function norm(a::DBOW)::Float64
    s = 0.0
    for w in values(a)
        s += w * w
    end

    sqrt(s)
end

function zero(::Type{DBOW{Ti,Tv}}) where {Ti,Tv<:Real}
    DBOW{Ti,Tv}()
end


## inplace sum
function add!(a::DBOW{Ti,Tv}, b::DBOW{Ti,Tv}) where {Ti,Tv<:Real}
    for (k, w) in b
        if w != 0
            a[k] = get(a, k, zero(Tv)) + w
        end
    end

    a
end

function add!(a::DBOW{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}
    for (k, w) in zip(b.nzind, b.nzval)
        if w != 0
            a[k] = get(a, k, zero(Tv)) + w
        end
    end

    a
end

function add!(a::DBOW{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}
    k, w = b
    a[k] = get(a, k, zero(Tv)) + w
    a
end

function +(a::DBOW{Ti,Tv}, b::DBOW{Ti,Tv}) where {Ti,Tv<:Real}
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

function +(a::DBOW, b::Pair)
    c = copy(a)
    add!(c, b)
end

## definitions for substraction
function -(a::DBOW{Ti,Tv}, b::DBOW{Ti,Tv}) where {Ti,Tv<:Real}
    c = copy(a)
    for (k, w) in b
        if w != 0
            c[k] = get(c, k, zero(Tv)) - w 
        end
    end

    c
end

## definitions for product

function *(a::DBOW{Ti,Tv}, b::DBOW{Ti,Tv}) where {Ti,Tv<:Real}
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

function *(a::DBOW, b::F) where F<:Real
    c = copy(a)
    for (k, v) in a
        c[k] = convert(F, v * b)
    end

    c
end

function *(b::F, a::DBOW) where F<:Real
    a * b
end

function /(a::DBOW, b::F) where F<:Real
    c = copy(a)
    for (k, v) in a
        c[k] = convert(F, v / b)
    end

    c
end

"""
    cosine_distance(a::DBOW, b::DBOW)::Float64

Computes the cosine_distance between two weighted bags

It supposes that bags are normalized (see `normalize!` function)

"""
function cosine_distance(a::DBOW, b::DBOW)::Float64
    return 1.0 - dot(a, b)
end

const π_2 = π / 2
"""
    angle_distance(a::DBOW, b::DBOW)::Float64

Computes the angle  between two weighted bags

It supposes that all bags are normalized (see `normalize!` function)

"""
function angle_distance(a::DBOW, b::DBOW)::Float64
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

function cosine(a::DBOW, b::DBOW)::Float64
    return dot(a, b) # * a.invnorm * b.invnorm # it is already normalized
end