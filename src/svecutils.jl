# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export dvec, bow
import SparseArrays: sparsevec, sparse

"""
    dvec(x::AbstractSparseVector)

Converts an sparse vector into a DVEC sparse vector
"""
function dvec(x::AbstractSparseVector)
    #DVEC{Symbol,Float64}(model.id2token[x.nzind[i]] => x.nzval[i] for i in nonzeroinds(x))
    DVEC{eltype(x.nzind), eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

sparse2dvec(x) = dvec

"""
    sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}

Creates a sparse vector from a´ DVEC sparse vector
"""
function sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}
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
    sparse(cols::AbstractVector{S}, m=0) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}

Creates a sparse matrix from an array of DVEC sparse vectors.
"""
function sparse(cols::AbstractVector{S}, m=0) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}
    I = Ti[]
    J = Ti[]
    F = Tv[]

    for j in eachindex(cols)
        for (term, weight) in cols[j]
            if term > 0
                push!(I, term)
                push!(J, j)
                push!(F, weight)
            end
        end
    end
    if m == 0
        sparse(I, J, F)
    else
        sparse(I, J, F, m, length(cols))
    end
end

"""
    bow(model::TextModel, x::AbstractSparseVector)
    bow(model::TextModel, x::DVEC{Ti,Tv}) where {Ti<:Integer,Tv<:Number}
    
Creates a bag of words using the sparse vector `x` and the text model `model`
"""
function bow(model::TextModel, x::AbstractSparseVector)
    DVEC{Symbol,eltype{x.nzval}}(model.id2token[t] => v for (t, v) in zip(x.nzind, x.nzval))
end

function bow(model::TextModel, x::DVEC{Ti,Tv}) where {Ti<:Integer,Tv<:Number}
    DVEC{Symbol,Tv}(model.id2token[t] => v for (t, v) in x)
end

"""
    dvec(model::TextModel, x::DVEC{Symbol,Tv}, Ti=Int) where Tv<:Number

Creates a DVEC sparse vector from a bag of words sparse vector (i.e., with type DVEC{Symbol,Tv}),
using the text model `model`
"""
function dvec(model::TextModel, x::DVEC{Symbol,Tv}, Ti=Int) where Tv<:Number
    DVEC{Ti,Tv}(model.tokens[t].id => v for (t, v) in x)
end
