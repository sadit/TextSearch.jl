# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: push!, length
import SimilaritySearch: search
import SparseArrays: nonzeroinds, nonzeros
export InvIndex, prune, search
using SimilaritySearch

const PostList = Vector{IdWeight}

"""
Inverted index search structure

"""
mutable struct InvIndex <: Index
    lists::Dict{Int,PostList}
    n::Int
    InvIndex() = new(Dict{Int,PostList}(), 0)
    InvIndex(lists, n) = new(lists, n)
end

# useful constant for searching
const EMPTY_POSTING_LIST = PostList()

"""
    update!(a::InvIndex, b::InvIndex)

Updates inverted index `a` with `b`.
"""
function update!(a::InvIndex, b::InvIndex)
    for (sym, list) in b.lists
        if haskey(a.lists, sym)
            append!(a.lists[sym], list)
        else
            a.lists[sym] = list
        end
    end

    a.n += b.n
    a
end

"""
    push!(index::InvIndex, p::Pair{Int,SVEC})

Inserts a weighted bag of words (BOW) into the index.

See [compute_bow](@ref) to compute a BOW from a text
"""
function push!(index::InvIndex, p::Pair{Int,SVEC})
    index.n += 1
    for (id, weight) in p.second
        if haskey(index.lists, id)
            push!(index.lists[id], IdWeight(p.first, weight))
        else
            index.lists[id] = [IdWeight(p.first, weight)]
        end
    end
end

function fit(::Type{InvIndex}, db::AbstractVector{SVEC})
    invindex = InvIndex()
    for (i, v) in enumerate(db)
        push!(invindex, i => v)
    end

    invindex
end

"""
    optimize!(invindex::InvIndex; keep_k=3000, store_large_lists=true)

Resizes large posting lists to maintain at most `keep_k` entries (most weighted ones).
If `store_large_lists` is true, then these lists are not deleted, but instead they are
stored under the negative of its key.
"""
function optimize!(invindex::InvIndex; keep_k=3000, store_large_lists=true)
    for (t, list) in invindex.lists
        if t < 0
            # ignore negative keys, negative keys store large lists
            if !store_large_lists
                delete!(invindex.lists, t)
            end

            continue
        end

        if length(list) > keep_k
            if store_large_lists
                invindex.lists[-t] = deepcopy(list)
            end
            sort!(list, by=x -> x.weight, rev=true)
            resize!(list, keep_k)
            sort!(list, by=x -> x.id)
        end
    end
end

"""
    prune(invindex::InvIndex, k)

Creates a new inverted index using the given `invindex` discarding many entries with low weight.
It keeps at most `k` entries for each posting list; it keeps those entries with more wight values.
"""
function prune(invindex::InvIndex, k)
    I = InvIndex()
    I.n = invindex.n
    sizehint!(I.lists, length(invindex.lists))

    for (t, list) in invindex.lists
        I.lists[t] = l = deepcopy(list)
        if length(list) > k
            sort!(l, by=x -> -x.weight)
            resize!(l, k)
            sort!(l, by=x -> x.id)         
        end
    end

    # normalizing prunned vectors
    _norm_pruned!(I)
end

function _norm_pruned!(I::InvIndex)
    D = SVEC()
    
    for (t, list) in I.lists
        for p in list
            D[p.id] = get(D, p.id, 0.0) + p.weight * p.weight
        end
    end

    for (k, v) in D
        D[k] = 1.0 / sqrt(v)
    end

    for k in keys(I.lists)
        I.lists[k] = l = [IdWeight(p.id, p.weight * D[p.id]) for p in I.lists[k]]
    end

    I
end

"""
    search(invindex::InvIndex, dist::Function, q::Dict{Symbol, R}, res::KnnResult) where R <: Real

Seaches for the k-nearest neighbors of `q` inside the index `invindex`. The number of nearest
neighbors is specified in `res`; it is also used to collect the results. Returns the object `res`.
If `dist` is set to `angle_distance` then the angle is reported; otherwise the
`cosine_distance` (i.e., 1 - cos) is computed.
"""

function search(invindex::InvIndex, dist::Function, q::SVEC, res::KnnResult; ignore_lists_larger_than::Int=10_000, small_query_k=3)
    if length(q) <= small_query_k
        search_with_intersection(invindex, dist, q, res; ignore_lists_larger_than=ignore_lists_larger_than)
    else
        search_with_union(invindex, dist, q, res; ignore_lists_larger_than=ignore_lists_larger_than)
    end
end

include("invindexio.jl")
include("intersection.jl")
