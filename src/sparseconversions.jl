# This file is a part of TextSearch.jl

import SparseArrays: sparsevec, sparse

export dvec, sparse_coo, sparse

"""
    dvec(x::AbstractSparseVector)

Converts an sparse vector into a DVEC sparse vector
"""
function dvec(x::AbstractSparseVector)
    DVEC{eltype(x.nzind), eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

"""
    sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}

Creates a sparse vector from a DVEC sparse vector
"""
function sparsevec(vec::DVEC{Ti,Tv}, m::Integer=0) where {Ti<:Integer,Tv<:Number}
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
    sparse(cols::AbstractVector{S}, m=0; minweight=1e-9) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}
    sparse_coo(cols::AbstractVector{S}, minweight=1e-9) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}

Creates a sparse matrix from an array of DVEC sparse vectors.
"""
function sparse_coo(cols::AbstractVector{DVEC{Ti,Tv}}, minweight=1e-9) where {Ti<:Integer,Tv<:Number}
    I = Ti[]
    J = Ti[]
    F = Tv[]

    for j in eachindex(cols)
        for (term, weight) in cols[j]
            if term > 0 && weight >= minweight
                push!(I, term)
                push!(J, j)
                push!(F, weight)
            end
        end
    end

    I, J, F
end

function sparse(cols::AbstractVector{DVEC{Ti,Tv}}, m=0; minweight=1e-9) where {Ti<:Integer,Tv<:Number}
    I, J, F = sparse_coo(cols, minweight)
    if m == 0
        sparse(I, J, F)
    else
        sparse(I, J, F, m, length(cols))
    end
end
