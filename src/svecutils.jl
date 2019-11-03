
export dvec, sparse2bow, bow2dvec, dvec2sparse, bow2sparse, dvec2bow
import SparseArrays: sparsevec


function dvec(x::AbstractSparseVector)
    #DVEC{Symbol,Float64}(model.id2token[x.nzind[i]] => x.nzval[i] for i in nonzeroinds(x))
    DVEC{eltype(x.nzind), eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

sparse2dvec(x) = dvec

function dvec2sparse(vec::DVEC{Ti,Tv}, m=0) where {Ti,Tv}
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

function dvec2sparse(cols::AbstractVector{S}) where S<:DVEC{Ti,Tv} where {Ti,Tv}
    I = Ti[]
    J = Ti[]
    F = Tv[]

    for j in eachindex(cols)
        for (t, weight) in vec
            push!(I, t)
            push!(J, j)
            push!(F, weight)
        end
    end

    sparsevec(I, J, F)
end

function sparse2bow(model::Model, x::AbstractSparseVector)
    DVEC{Symbol,eltype{x.nzval}}(model.id2token[t] => v for (t, v) in zip(x.nzind, x.nzval))
end

function dvec2bow(model::Model, x::DVEC{Ti,Tv}) where {Ti,Tv}
    DVEC{Symbol,Tv}(model.id2token[t] => v for (t, v) in x)
end

function bow2dvec(model::Model, x::DVEC{Symbol,Tv}, Ti=Int) where Tv<:Real
    DVEC{Ti,Tv}(model.tokens[t].id => v for (t, v) in zip(x.nzind, x.nzval))
end

function bow2sparse(model::VectorModel, bow::DVEC{Symbol,Tv}, Ti=Int) where Tv
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

function bow2sparse(model::EntModel, bow::DVEC{Symbol,Tv}, Ti=Int) where Tv
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