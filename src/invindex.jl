import Base: push!
import SimilaritySearch: search
using SimilaritySearch
export InvIndex, prune, search

"""
Inverted index search structure

"""
mutable struct InvIndex
    lists::Dict{Symbol, Vector{SparseVectorEntry}}
    n::Int
    InvIndex() = new(Dict{Symbol, Vector{SparseVectorEntry}}(), 0)
end

function save(f::IO, invindex::InvIndex)
    write(f, invindex.n)
    write(f, length(invindex.lists))
    for plist in invindex.lists
        write(f, length(plist))
        for p in plist
            write(f, p.id, p.weight)
        end
    end
end

"""
    push!(index::InvIndex, bow::Dict{Symbol,Float64})

Inserts a weighted bag of words (BOW) into the index.

See [compute_bow](@ref) to compute a BOW from a text
"""
function push!(index::InvIndex, bow::Dict{Symbol,Float64})
    index.n += 1
    objID = index.n
    for (sym, weight) in bow
        if haskey(index.lists, sym)
            push!(index.lists[sym], SparseVectorEntry(objID, weight))
        else
            index.lists[sym] = [SparseVectorEntry(objID, weight)]
        end
    end
end


"""
    prune(invindex::InvIndex, k)

Creates a new inverted index using the given `invindex` keeping at most `k`
entries for each posting list; it keeps those entries with more wight values.

"""
function prune(invindex::InvIndex, k)
    I = InvIndex()
    I.n = invindex.n
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

"""
    search(invindex::InvIndex, q::Dict{Symbol, R}, res::KnnResult) where R <: Real

Seaches for the k-nearest neighbors of `q` inside the index `invindex`. The number of nearest
neighbors is specified in `res`; it is also used to collect the results. Returns the object `res`.

"""
function search(invindex::InvIndex, q::Dict{Symbol, R}, res::KnnResult) where R <: Real
    D = Dict{Int, Float64}()
    # normalize!(q) # we expect a normalized q 
    for (sym, weight) in q
        for e in invindex.lists[sym]
            D[e.id] = get(D, e.id, 0.0) + weight * e.weight
        end
    end

    for (id, weight) in D
        push!(res, id, 1.0 - weight)
    end

    res
end