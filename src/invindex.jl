import Base: push!
import SimilaritySearch: search
export InvIndex, prune, search
using SimilaritySearch

mutable struct PostingListType{F, I <: Integer} <: AbstractSparseVector{F, I}
    nzind::Vector{I}
    nzval::Vector{F}
    n::Int
end

const PostingList = PostingListType{Float64,Int}
const EMPTY_POSTING_LIST = PostingList(Int[], Float64[], 0)

function push!(list::PostingListType{F, I}, p::Pair{I,F}) where I <: Integer where F <: Real
    if length(list.nzind) > 0 && list.nzind[end] >= p[1]
        error("ERROR: push! for PostingListType accepts only identifiers in increasing order -> $(list.nzind[end]) >= $(p[1])")
    end
    push!(list.nzind, p[1])
    push!(list.nzval, p[2])
    if p[1] > list.n
        list.n = p[1]
    end

    list
end

"""
mutable struct InvIndex{F, I <: Integer} <: Index

Inverted index search structure
"""
mutable struct InvIndex <: Index
    lists::Dict{Int, PostingList}
    n::Int
    function InvIndex()
        lists = Dict{Int, PostingList}()
        new(lists, 0)
    end

    #InvIndex(lists, n) = new(lists, n)
end


"""
    push!(index::InvIndex, vec::AbstractSparseVector)

Inserts a weighted bag of words into the index.
see [vectorize](@ref)
"""
function push!(index::InvIndex, vec::AbstractSparseVector)
    index.n += 1
    objID = index.n
    for (term, weight) in zip(vec.nzind, vec.nzval)
        lst = get(index.lists, term, nothing)
        if lst === nothing
            index.lists[term] = PostingList([objID], [weight], index.n)
        else
            push!(lst, objID => weight)
        end
    end
end

function fit(::Type{InvIndex}, db::AbstractVector{S}) where S <: AbstractSparseVector
    index = InvIndex()
    for vec in db
        push!(index, vec)
    end

    fix_size!(index)
end

function fix_size!(index::InvIndex)
    for list in values(index.lists)
        list.n = index.n
    end

    index
end

"""
    prune(invindex::InvIndex, k)

Creates a new inverted index using the given `invindex` discarding many entries with low weight.
It keeps at most `k` entries for each posting list; it keeps those entries with more wight values.
"""
function prune(index::InvIndex, k)
    I = InvIndex()
    I.n = index.n
    sizehint!(I.lists, length(index.lists))
    P = zeros(Int, I.n)  # TODO compute maximum posting-list's length
    for (t, list) in index.lists
        I.lists[t] = l = deepcopy(list)
        m = length(l.nzval)
        if m > k
            p = @view P[1:m]
            sortperm!(p, l.nzval, rev=true)
            permute!(l.nzind, p)
            permute!(l.nzval, p)
            resize!(l.nzind, k)
            resize!(l.nzval, k)
            p = @view P[1:k]
            sortperm!(p, l.nzind)
            sortperm!(p, l.nzval)
        end
    end

    # normalizing prunned vectors
    _norm_prunned!(I)
end

function _norm_prunned!(index::InvIndex)
    D = Dict{Int,Float64}() 
    
    for (t, list) in index.lists
        for (id, weight) in zip(list.nzind, list.nzval)
            D[id] = get(D, id, 0.0) + weight^2
        end
    end

    for (k, v) in D
        D[k] = 1.0 / sqrt(v)
    end

    for (t, list) in index.lists
        for i in eachindex(list.nzind)
            list.nzval[i] == D[list.nzind[i]]
        end
    end

    index
end

"""
    search(invindex::InvIndex, dist::Function, q::AbstractSparseVector, res::KnnResult) where R <: Real

Seaches for the k-nearest neighbors of `q` inside the index `invindex`. The number of nearest
neighbors is specified in `res`; it is also used to collect the results. Returns the object `res`.
If `dist` is set to `angle_distance` then the angle is reported; otherwise the
`cosine_distance` (i.e., 1 - cos) is computed.
"""
function search(invindex::InvIndex, dist::Function, q::AbstractSparseVector, res::KnnResult)
    D = Dict{Int, Float64}()
    # normalize!(q) # we expect a normalized q

    for (t, weight) in zip(q.nzind, q.nzval)
        lst = get(invindex.lists, t, EMPTY_POSTING_LIST)
        if lst.n > 0
            for (i, w) in zip(lst.nzind, lst.nzval)
                D[i] = get(D, i, 0.0) + weight * w
            end
        end
    end

    for (i, w) in D
        if dist == angle_distance
            w = max(-1.0, w)
            w = min(1.0, w)
            w = acos(w)
            push!(res, i, w)
        else
            push!(res, i, 1.0 - w)  # cosine distance
        end
    end

    res
end
