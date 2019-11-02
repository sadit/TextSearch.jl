
export dvec
import SparseArrays: sparsevec

function dvec(model::Model, x::AbstractSparseVector)
    #DVEC{Symbol,Float64}(model.id2token[x.nzind[i]] => x.nzval[i] for i in nonzeroinds(x))
    DVEC{Symbol,Float64}(model.id2token[t] => v for (t, v) in zip(x.nzind, x.nzval))
end

function dvec(x::AbstractSparseVector)
    #DVEC{Symbol,Float64}(model.id2token[x.nzind[i]] => x.nzval[i] for i in nonzeroinds(x))
    DVEC(t => v for (t, v) in zip(x.nzind, x.nzval))
end


function sparsevec(model::VectorModel, bow::DVEC{Symbol,Tv}) where Tv
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

function sparsevec(model::EntModel, bow::DVEC{Symbol,Tv}) where Tv
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