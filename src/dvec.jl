# This file is a part of TextSearch.jl

import Base: +, -, *, /, ==, zero
using LinearAlgebra, SparseArrays
import LinearAlgebra: dot, norm, normalize!
#import SparseArrays: nnz
using SimilaritySearch
using SimilaritySearch.Dist: NormAngle, NormCosine, Angle, Cosine
import SimilaritySearch: evaluate
export centroid, evaluate, NormAngle, NormCosine, Angle, Cosine, l1norm, l1normalize!, sparsedot

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

# Example

```julia
julia> using LinearAlgebra

julia> a = Dict{UInt32,Float32}(1 => 3.0, 2 => 4.0);

julia> normalize!(a)
Dict{UInt32, Float32}(0x00000002 => 0.8, 0x00000001 => 0.6)
```
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

# Example

```julia
julia> using LinearAlgebra

julia> dot(Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8), Dict{UInt32,Float32}(2 => 1.0))
0.8f0
```
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

# Example

```julia
julia> using LinearAlgebra

julia> norm(Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8))
1.0f0
```
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

# Example

```julia
julia> a = Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8);

julia> TextSearch.add!(a, Dict{UInt32,Float32}(2 => 0.5))
Dict{UInt32, Float32}(0x00000002 => 1.3, 0x00000001 => 0.6)
```
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

Computes the sum of the given list of vectors.

Implemented as "concatenate every (key, value) pair from every input dict, sort once
by key, then combine consecutive equal keys in a single linear pass" rather than
folding with repeated `add!` calls — the latter rebuilds/rehashes a growing
accumulator on every step (effectively quadratic in the number of vectors for large
collections), while sorting once is `O(N log N)` in the total number of entries
regardless of how many vectors they came from. See
[sadit/TextSearch.jl#25](https://github.com/sadit/TextSearch.jl/issues/25) for the
benchmarks behind this choice.
"""
function Base.sum(col::AbstractVector{<:Dict})
    Ti, Tv = keytype(eltype(col)), valtype(eltype(col))
    total = sum(length, col)
    all_ind = Vector{Ti}(undef, total)
    all_val = Vector{Tv}(undef, total)
    p = 1
    for d in col
        for (k, v) in d
            all_ind[p] = k
            all_val[p] = v
            p += 1
        end
    end

    perm = sortperm(all_ind)
    permute!(all_ind, perm)
    permute!(all_val, perm)

    out = Dict{Ti,Tv}()
    sizehint!(out, total)
    i = 1
    @inbounds while i <= total
        j = i
        s = all_val[i]
        while j < total && all_ind[j+1] == all_ind[i]
            j += 1
            s += all_val[j]
        end
        out[all_ind[i]] = s
        i = j + 1
    end

    out
end


"""
    centroid(cluster::AbstractVector{<:Dict})

Computes a centroid of the given list of Dict vectors

# Example

```julia
julia> centroid([Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8), Dict{UInt32,Float32}(2 => 1.0)])
Dict{UInt32, Float32}(0x00000002 => 0.9486834, 0x00000001 => 0.3162278)
```
"""
function centroid(cluster::AbstractVector{<:Dict})
    normalize!(sum(cluster))
end

"""
    +(a::Dict{Ti,Tv}, b::Dict{Ti,Tv}) where {Ti,Tv<:Real}
    +(a::Dict, b::Pair)

Computes the sum of `a` and `b`

# Example

```julia
julia> Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8) + Dict{UInt32,Float32}(2 => 1.0)
Dict{UInt32, Float32}(0x00000002 => 1.8, 0x00000001 => 0.6)
```
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

# Example

```julia
julia> Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8) - Dict{UInt32,Float32}(2 => 1.0)
Dict{UInt32, Float32}(0x00000002 => -0.19999999, 0x00000001 => 0.6)
```
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

# Example

```julia
julia> Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8) * Dict{UInt32,Float32}(2 => 1.0)
Dict{UInt32, Float32}(0x00000002 => 0.8)

julia> Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8) * 2.0
Dict{UInt32, Float32}(0x00000002 => 1.6, 0x00000001 => 1.2)
```
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

# Example

```julia
julia> Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8) / 2.0
Dict{UInt32, Float32}(0x00000002 => 0.4, 0x00000001 => 0.3)
```
"""
function /(a::Dict{K,V}, b::F) where K where {V<:Real} where {F<:Real}
    a * (1.0 / b)
end


"""
    evaluate(::NormCosine, a::Dict, b::Dict)::Float64

Computes the cosine distance between two Dict sparse vectors

It supposes that bags are normalized (see `normalize!` function)

# Example

```julia
julia> evaluate(NormCosine(), Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8), Dict{UInt32,Float32}(2 => 1.0))
0.19999998807907104
```
"""
function evaluate(::NormCosine, a::Dict, b::Dict)::Float64
    1.0 - dot(a, b)
end

"""
    evaluate(::Cosine, a::Dict, b::Dict)::Float64

Computes the cosine distance between two Dict sparse vectors (unlike `NormCosine`,
`a`/`b` need not be pre-normalized).

# Example

```julia
julia> evaluate(Cosine(), Dict{UInt32,Float32}(1 => 3.0, 2 => 4.0), Dict{UInt32,Float32}(2 => 1.0))
0.19999998807907104
```
"""
function evaluate(::Cosine, a::Dict, b::Dict)::Float64
    1.0 - full_cosine(a, b)
end

const π_2 = π / 2

"""
    evaluate(::NormAngle, a::Dict, b::Dict)::Float64

Computes the angle  between two Dict sparse vectors

It supposes that all bags are normalized (see `normalize!` function)

# Example

```julia
julia> evaluate(NormAngle(), Dict{UInt32,Float32}(1 => 0.6, 2 => 0.8), Dict{UInt32,Float32}(2 => 1.0))
0.6435011029243469
```
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

Computes the angle between two Dict sparse vectors (unlike `NormAngle`, `a`/`b`
need not be pre-normalized).

# Example

```julia
julia> evaluate(Angle(), Dict{UInt32,Float32}(1 => 3.0, 2 => 4.0), Dict{UInt32,Float32}(2 => 1.0))
0.6435010889250692
```
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

#####
##
## SparseVector support
##
## `dot`, `norm`, `normalize!`, `+`, `-`, `*` (scalar and elementwise via `.*`), and
## `/` (scalar) already work correctly and efficiently for `SparseVector` out of the
## box via `SparseArrays`/`LinearAlgebra` — no need to redefine them here. What's
## missing is (1) a faster dot product for the very asymmetric "short query vs long
## document" case that dominates similarity search, and (2) a fast way to sum/centroid
## many `SparseVector`s at once. See
## [sadit/TextSearch.jl#25](https://github.com/sadit/TextSearch.jl/issues/25) for the
## benchmarks and design behind both.
##
#####

@inline function _gallop_lower_bound(B::Vector{Ti}, start::Int, n::Int, x::Ti) where Ti
    (start > n || B[start] >= x) && return start
    lo = start
    step = 1
    hi = start + step
    @inbounds while hi <= n && B[hi] < x
        lo = hi
        step += step
        hi = lo + step
    end
    hi = min(hi, n + 1)
    lo += 1
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if B[mid] < x
            lo = mid + 1
        else
            hi = mid
        end
    end
    lo
end

function _dot_linear(ai::AbstractVector{Ti}, av::AbstractVector{Tv}, bi::AbstractVector{Ti}, bv::AbstractVector{Tv}) where {Ti,Tv}
    na, nb = length(ai), length(bi)
    s = zero(Tv)
    i = j = 1
    @inbounds while i <= na && j <= nb
        xa, xb = ai[i], bi[j]
        if xa == xb
            s += av[i] * bv[j]
            i += 1; j += 1
        elseif xa < xb
            i += 1
        else
            j += 1
        end
    end

    s
end

function _dot_gallop(ai::AbstractVector{Ti}, av::AbstractVector{Tv}, bi::Vector{Ti}, bv::AbstractVector{Tv}) where {Ti,Tv}
    # ai/av MUST be the smaller side
    na, nb = length(ai), length(bi)
    s = zero(Tv)
    pos = 1
    @inbounds for i in 1:na
        pos > nb && break
        x = ai[i]
        pos = _gallop_lower_bound(bi, pos, nb, x)
        if pos <= nb && bi[pos] == x
            s += av[i] * bv[pos]
        end
    end

    s
end

"""
    sparsedot(a::SparseVector, b::SparseVector; small_threshold::Int=30, ratio_threshold::Float64=3.0)

Adaptive dot product between two `SparseVector`s:
- both sides have fewer than `small_threshold` stored entries, or their sizes are
  within `ratio_threshold` of each other: a plain linear merge (the same algorithm
  `LinearAlgebra.dot` already uses for `SparseVector`).
- otherwise (one side much larger than the other — e.g. a short query against a long
  document): a Hwang-Lin/galloping merge — for each stored entry of the smaller side,
  an exponential ("galloping") search with memory of the last found position locates
  its match in the larger side in `O(log gap)` instead of a full linear scan.

This is deliberately *not* a method of `LinearAlgebra.dot(::SparseVector,::SparseVector)`:
`SparseArrays` already owns that method (a plain merge), and shadowing it package-wide
would be a much more aggressive form of type piracy than TextSearch's existing `Dict`
overloads of `dot`/`normalize!` (neither `SparseVector` nor `dot` belong to TextSearch,
whereas extending `dot` for `Dict` doesn't collide with any other package's definitions).
[`evaluate`](@ref) for `SparseVector` uses `sparsedot` internally.

# Example

```julia
julia> using SparseArrays

julia> sparsedot(sparsevec(UInt32[1, 2], Float32[0.6, 0.8], 10), sparsevec(UInt32[2], Float32[1.0], 10))
0.8f0
```
"""
function sparsedot(a::SparseVector{Tv,Ti}, b::SparseVector{Tv,Ti};
        small_threshold::Int=30, ratio_threshold::Float64=3.0) where {Tv,Ti}
    ai, av, bi, bv = a.nzind, a.nzval, b.nzind, b.nzval
    na, nb = length(ai), length(bi)
    (na == 0 || nb == 0) && return zero(Tv)
    lo, hi = na <= nb ? (na, nb) : (nb, na)
    if hi < small_threshold || hi / lo <= ratio_threshold
        return _dot_linear(ai, av, bi, bv)
    end

    na <= nb ? _dot_gallop(ai, av, bi, bv) : _dot_gallop(bi, bv, ai, av)
end

"""
    evaluate(::NormCosine, a::SparseVector, b::SparseVector)::Float64
    evaluate(::Cosine, a::SparseVector, b::SparseVector)::Float64
    evaluate(::NormAngle, a::SparseVector, b::SparseVector)::Float64
    evaluate(::Angle, a::SparseVector, b::SparseVector)::Float64

`SparseVector` counterparts of the `Dict`-based distance functions above, using
[`sparsedot`](@ref) instead of the plain-merge `dot` for the underlying inner
product. `NormCosine`/`NormAngle` assume `a`/`b` are already normalized;
`Cosine`/`Angle` do not.

# Example

```julia
julia> using SparseArrays

julia> evaluate(NormCosine(), sparsevec(UInt32[1, 2], Float32[0.6, 0.8], 10), sparsevec(UInt32[2], Float32[1.0], 10))
0.19999998807907104
```
"""
function evaluate(::NormCosine, a::SparseVector, b::SparseVector)::Float64
    1.0 - sparsedot(a, b)
end

function evaluate(::Cosine, a::SparseVector, b::SparseVector)::Float64
    1.0 - sparsedot(a, b) / (norm(a) * norm(b))
end

function evaluate(::NormAngle, a::SparseVector, b::SparseVector)::Float64
    d = sparsedot(a, b)

    if d <= -1.0
        π
    elseif d >= 1.0
        0.0
    elseif d == 0
        π_2
    else
        acos(d)
    end
end

function evaluate(::Angle, a::SparseVector, b::SparseVector)::Float64
    d = sparsedot(a, b) / (norm(a) * norm(b))

    if d <= -1.0
        π
    elseif d >= 1.0
        0.0
    elseif d == 0
        π_2
    else
        acos(d)
    end
end

"""
    Base.sum(cluster::AbstractVector{<:SparseVector})

`SparseVector` counterpart of `sum(::AbstractVector{<:Dict})`: concatenate every
`(index, value)` pair from every input vector, sort once by index, then combine
consecutive equal indices in a single linear pass. Beat every alternative tried
(naive `+`-folding, pairwise tree merging, a k-way heap merge) at every cluster size
benchmarked, and — unlike a dense-accumulator approach — its cost does not depend on
the vectors' dimension, only on their total number of stored entries. See
[sadit/TextSearch.jl#25](https://github.com/sadit/TextSearch.jl/issues/25).

All vectors in `cluster` must have the same dimension (`length`).
"""
function Base.sum(cluster::AbstractVector{<:SparseVector{Tv,Ti}}) where {Tv,Ti}
    n = length(cluster[1])
    total = sum(nnz, cluster)
    all_ind = Vector{Ti}(undef, total)
    all_val = Vector{Tv}(undef, total)
    p = 1
    for v in cluster
        @assert length(v) == n "all vectors in `cluster` must share the same dimension"
        m = nnz(v)
        copyto!(all_ind, p, v.nzind, 1, m)
        copyto!(all_val, p, v.nzval, 1, m)
        p += m
    end

    perm = sortperm(all_ind)
    permute!(all_ind, perm)
    permute!(all_val, perm)

    out_ind = Vector{Ti}(); sizehint!(out_ind, total)
    out_val = Vector{Tv}(); sizehint!(out_val, total)
    i = 1
    @inbounds while i <= total
        j = i
        s = all_val[i]
        while j < total && all_ind[j+1] == all_ind[i]
            j += 1
            s += all_val[j]
        end
        push!(out_ind, all_ind[i])
        push!(out_val, s)
        i = j + 1
    end

    SparseVector(n, out_ind, out_val)
end

"""
    centroid(cluster::AbstractVector{<:SparseVector})

Centroid (normalized sum) of a cluster of `SparseVector`s. See
[`sum(::AbstractVector{<:SparseVector})`](@ref sum) for the algorithm.
"""
centroid(cluster::AbstractVector{<:SparseVector}) = normalize!(sum(cluster))
