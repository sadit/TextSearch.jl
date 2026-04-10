# This file is a part of TextSearch.jl

import SparseArrays: sparsevec, sparse

export dvec, sparse_coo, sparse

"""
    dvec(x::AbstractSparseVector)

Converts an sparse vector into a dict-based sparse vector
"""
function dvec(x::AbstractSparseVector)
    Dict{eltype(x.nzind),eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

"""
    sparsevec(vec::Dict{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}

Creates a sparse vector from a Dict-based sparse vector
"""
function sparsevec(vec::Dict{Ti,Tv}, m::Integer=0) where {Ti<:Integer,Tv<:Number}
    I = Ti[]
    F = Tv[]

    for (t, weight) in vec
        if t > 0
            push!(I, t)
            push!(F, weight)
        end
    end

    if m == 0
        sparsevec(I, F)
    else
        sparsevec(I, F, m)
    end
end



"""
    sparse(cols::AbstractVector{<:Dict}, m=0; minweight=1e-9) 
    sparse_coo(cols::AbstractVector{<:Dict}, minweight=1e-9)

Creates a sparse matrix from an array of Dict sparse vectors.
"""
function sparse_coo(cols::AbstractVector{S}, minweight=1e-9) where {S<:Dict}
    I = keytype(S)[]
    J = keytype(S)[]
    F = valtype(S)[]

    n = length(cols)
    n == 0 && return I, J, F

    let n = n * length(cols[1])
        sizehint!(I, n)
        sizehint!(J, n)
        sizehint!(F, n)
    end

    for (j, c) in enumerate(cols)
        for (term, weight) in c
            if term > 0 && weight >= minweight
                push!(I, term)
                push!(J, j)
                push!(F, weight)
            end
        end
    end

    I, J, F
end

function sparse(cols::AbstractVector{<:Dict}, m=0; minweight=1e-9)
    I, J, F = sparse_coo(cols, minweight)
    if m == 0
        sparse(I, J, F)
    else
        sparse(I, J, F, m, length(cols))
    end
end
