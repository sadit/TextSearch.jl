
export dvec, sparse2bow, bow2dvec
import SparseArrays: sparsevec

function sparse2bow(model::Model, x::AbstractSparseVector)
    DVEC{Symbol,eltype{x.nzval}}(model.id2token[t] => v for (t, v) in zip(x.nzind, x.nzval))
end

function sparse2bow(model::Model, x::DVEC{Ti,Tv}) where {Ti,Tv}
    DVEC{Symbol,Tv}(model.id2token[t] => v for (t, v) in x)
end

function bow2dvec(model::Model, x::DVEC{Symbol,Tv}) where Tv<:Real
    DVEC{Int,Tv}(model.tokens[t].id => v for (t, v) in zip(x.nzind, x.nzval))
end

function dvec(x::AbstractSparseVector)
    #DVEC{Symbol,Float64}(model.id2token[x.nzind[i]] => x.nzval[i] for i in nonzeroinds(x))
    DVEC{eltype(x.nzind), eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

sparse2dvec(x) = dvec

function bow2sparse(model::VectorModel, bow::DVEC{Symbol,Tv}) where Tv
    I = Int[]
    F = Float64[]

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

function bow2sparse(model::EntModel, bow::DVEC{Symbol,Tv}) where Tv
    I = Int[]
    F = Float64[]

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