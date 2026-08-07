# This file is a part of TextSearch.jl

import Base: +, -, *, /, ==, zero
using LinearAlgebra, SparseArrays
import LinearAlgebra: dot, norm, normalize!
#import SparseArrays: nnz
using SimilaritySearch
using SimilaritySearch.Dist: NormAngle, NormCosine, Angle, Cosine
import SimilaritySearch.Dist: evaluate
export centroid, sparsedot

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
        pos = doublingsearch(bi, x, pos, nb)
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
