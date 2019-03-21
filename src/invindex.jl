
mutable struct InvIndex
    lists::Dict{Int, Vector{SparseVectorEntry}}
    n::Int
    InvIndex() = new(Dict{Int, Vector{SparseVectorEntry}}(), 0)
end

function push!(index::InvIndex, bow::SparseVector)
    index.n += 1
    objID = index.n
    for t in bow
        if hashkey(invindex, t.id)
            push!(index.lists[t.id], SparseVectorEntry(objID, t.weight))
        else
            index.lists[t.id] = [SparseVectorEntry(objID, t.weight)]
        end
    end
end

function prune_lists(invindex::InvIndex, k)
    I = InvIndex()
    I.n = invidex.n
    for (t, list) in invindex.lists
        I.lists[t] = l = copy(list)
        sort!(l, by=x -> x.weight)
        if length(list) > k
            resize!(l, k)
        end
    end

    # normalizing prunned vectors
    D = zeros(Float64, I.n)
    for (t, list) in I.lists
        @inbounds for p in list
            D[p.id] += p.weight * p.weight
        end
    end

    for i in 1:length(D)
        if D[i] == 0.0
            D[i] = 1.0
        else
            D[i] = 1.0 / D[i]
        end
    end

    for (t, list) in I.lists
        for p in list
            p.weight *= D[p.id]
        end
    end

    I
end

function search(invindex::InvIndex, q::SparseVector, res::KnnResult)
    D = Dict{Int, Float64}()
    # normalize!(q) # we expect a normalized q 
    for p in q.tokens
        invindex.lists[p.id]
            D[p.id] = get(D, p.id, 0.0) + p.weight
        end
    end

    for (id, weight) in D
        push!(res, id, weight)
    end

    res
end