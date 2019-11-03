
export dvec, bow
import SparseArrays: sparsevec, sparse


function dvec(x::AbstractSparseVector)
    #DVEC{Symbol,Float64}(model.id2token[x.nzind[i]] => x.nzval[i] for i in nonzeroinds(x))
    DVEC{eltype(x.nzind), eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

sparse2dvec(x) = dvec

function sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}
    I = Ti[]
    F = Tv[]

    for (t, weight) in vec
        push!(I, t)
        push!(F, weight)
    end

    if m == 0
        sparsevec(I, F)
    else
        sparsevec(I, F, m)
    end
end

function sparse(cols::AbstractVector{S}, m=0) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}
    I = Ti[]
    J = Ti[]
    F = Tv[]

    for j in eachindex(cols)
        for (t, weight) in cols[j]
            push!(I, t)
            push!(J, j)
            push!(F, weight)
        end
    end
    if m == 0
        sparse(I, J, F)
    else
        sparse(m, length(cols), I, J, F)
    end
end

function bow(model::Model, x::AbstractSparseVector)
    DVEC{Symbol,eltype{x.nzval}}(model.id2token[t] => v for (t, v) in zip(x.nzind, x.nzval))
end

function bow(model::Model, x::DVEC{Ti,Tv}) where {Ti<:Integer,Tv<:Number}
    DVEC{Symbol,Tv}(model.id2token[t] => v for (t, v) in x)
end

function dvec(model::Model, x::DVEC{Symbol,Tv}, Ti=Int) where Tv<:Number
    DVEC{Ti,Tv}(model.tokens[t].id => v for (t, v) in zip(x.nzind, x.nzval))
end

function sparsevec(model::VectorModel, bow::DVEC{Symbol,Tv}, Ti=Int) where {Tv<:Number}
    I = Ti[]
    F = Tv[]

    for (sym, weight) in bow
        idfreq = get(model.tokens, sym, nothing)
        if idfreq === nothing
            continue
        end

        push!(I, idfreq.id)
        push!(F, weight)
    end

    sparsevec(I, F, model.m)
end

function sparsevec(model::EntModel, bow::DVEC{Symbol,Tv}, Ti=Int) where {Tv<:Number}
    I = Ti[]
    F = Tv[]

    for (sym, weight) in bow
        idweight = get(model.tokens, sym, nothing)
        if idweight === nothing
            continue
        end

        push!(I, idweight.id)
        push!(F, weight)
    end

    sparsevec(I, F, model.m)
end