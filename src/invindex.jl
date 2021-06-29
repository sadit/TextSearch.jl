# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: push!, length
import SimilaritySearch: search, optimize!
import SparseArrays: nonzeroinds, nonzeros
export InvIndex, search, optimize!
using SimilaritySearch

const PostList = Vector{IdWeight}

mutable struct InvIndex <: AbstractSearchContext
    lists::Dict{UInt64,PostList}
    n::Int
    res::KnnResult
end

Base.show(io::IO, invindex::InvIndex) = print(io, "{InvIndex vocsize=$(length(invindex.lists)), n=$(invindex.n)}")

InvIndex() = InvIndex(Dict{Int,PostList}(), 0, KnnResult(10))
InvIndex(lists, n; ksearch=10) = InvIndex(lists, n, KnnResult(ksearch))

"""
    InvIndex(db::AbstractVector{SVEC})

Creates an inverted index search structure

"""

function InvIndex(db::AbstractVector{SVEC})
    invindex = InvIndex()
    for (i, v) in enumerate(db)
        push!(invindex, i => v)
    end

    invindex
end

Base.copy(invindex::InvIndex; lists=invindex.lists, n=invindex.n, res=KnnResult(maxlength(invindex.res))) =
    InvIndex(lists, n, res)

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

Inserts a weighted vector into the index.

"""
function push!(index::InvIndex, p::Pair{I,SVEC}) where {I<:Integer}
    index.n += 1
    for (id, weight) in p.second
        if haskey(index.lists, id)
            push!(index.lists[id], IdWeight(p.first, weight))
        else
            index.lists[id] = [IdWeight(p.first, weight)]
        end
    end
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
    prune(invindex::InvIndex, keeptop::Integer)
    prune(invindex::InvIndex, keeptop_ratio::AbstractFloat)

Creates a new inverted index using the given `invindex`; the idea is to discard document entries with low weight.
It keeps at most `keeptop` entries for each posting list.
"""
function prune(invindex::InvIndex, keeptop::Integer)
    I = InvIndex()
    I.n = invindex.n
    sizehint!(I.lists, length(invindex.lists))

    for (t, list) in invindex.lists
        I.lists[t] = l = deepcopy(list)
        if length(list) > keeptop
            sort!(l, by=x -> -x.weight)
            resize!(l, keeptop)
            sort!(l, by=x -> x.id)         
        end
    end

    # normalizing prunned vectors
    _norm_pruned!(I)
end

prune(invindex::InvIndex, ratio::AbstractFloat) = prune(invindex, max(1, ceil(Int, ratio * invindex.n)))

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
    search(invindex::InvIndex, q::SVEC, res::KnnResult; ignore_lists_larger_than::Int=typemax(Int))


Seaches for the k-nearest neighbors of `q` inside the index `invindex`. The number of nearest
neighbors is specified in `res`; it is also used to collect the results. Returns the object `res`.
"""

function search(invindex::InvIndex, q::SVEC, res::KnnResult; ignore_lists_larger_than::Int=typemax(Int))
    search_with_union(invindex, q, res, ignore_lists_larger_than=ignore_lists_larger_than)
end

function search(invindex::InvIndex, q::SVEC, ksearch::Integer; ignore_lists_larger_than::Int=typemax(Int))
    empty!(invindex.res, ksearch)
    search(invindex, q, invindex.res; ignore_lists_larger_than=ignore_lists_larger_than)
end

include("setsearch.jl")
