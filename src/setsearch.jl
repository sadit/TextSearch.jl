# This file is a part of TextSearch.jl
# License is Apac

export search_with_intersection, search_with_union, search_with_one_error, intersection
"""
find_insert_position(arr::AbstractVector, value)

Finds the insert position of `value` inside `arr`.
Note: it returns `0` for lower limit.
"""
function find_insert_position(v, arr::AbstractVector, by::Function)::Int
    n = length(arr)
    sp = 1
    ep = n
    while sp != ep
        imedian = ceil(Int, (ep+sp) / 2)
        median = by(arr[imedian])
        if v < median
            ep = imedian-1
        elseif median < v
            sp = imedian
        else
            return imedian
        end
    end
    
    sp == 1 && v < by(arr[sp]) ? 0 : sp
end

"""
    baezayates(first::AbstractVector, byfirst::Function, second::AbstractVector{T}, bysecond::Function, output::AbstractVector{T}) where T

Computes the intersection between first and second ordered lists using the Baeza-Yates algorithm [cite]; elements are mapped to a comparable value using `byfirst` and `bysecond` functions, for `first` and `second` lists.
The matched objects of `second` are stored in `output`.
"""
function baezayates(first::AbstractVector, byfirst::Function, second::AbstractVector{T}, bysecond::Function, output::AbstractVector{T}) where T
    m = length(first)
    n = length(second)
    if m == 0 || n == 0
        return output
    end
    imedian = ceil(Int, m / 2)
    median = byfirst(first[imedian])
    pos = find_insert_position(median, second, bysecond)
    pos_matches = pos > 0 && median == bysecond(second[pos])
    _first = @view first[1:imedian-1]
    left_pos = pos_matches ? pos - 1 : pos
    _second = @view second[1:left_pos]
    
    # @show "======" imedian median pos _first _second
    length(_first) > 0 && length(_second) > 0 && baezayates(_first, byfirst, _second, bysecond, output)
    
    if pos == 0
        pos += 1
    elseif median == bysecond(second[pos])
        push!(output, second[pos])
        #callback(first[imedian], second[pos])
        pos += 1
    end
    
    _first = @view first[imedian+1:m]
    _second = @view second[pos:n]
    length(_first) > 0 && length(_second) > 0 && baezayates(_first, byfirst, _second, bysecond, output)
    
    output
end

"""
    _svs(T::Type, sets::AbstractVector, by::Function)

Computes the intersection of the ordered lists in `sets` using the by::Function
to extract a comparable for elements in each list
"""
function _svs(T::Type, sets::AbstractVector, by::Function)    
    sort!(sets, by=p->length(p), rev=true)
    res = baezayates(pop!(sets), by, pop!(sets), by, T[])
    push!(sets, res)

    while length(sets) > 1
        res = baezayates(pop!(sets), by, pop!(sets), by, T[])
        push!(sets, res)
    end

    sets[1]
end

"""
    intersection(sets::AbstractVector{S}, by::Function=identity) where {S<:AbstractVector}

Computes the intersection of sets represented by ordered arrays `lists` using the by::Function
to extract a comparable for elements in each list
"""
function intersection(sets::AbstractVector{S}, by::Function=identity) where
        {S<:AbstractVector}
    n = length(sets)
    T = eltype(eltype(sets))

    if n == 0
        T[]
    elseif n == 1
        sets[1]
    else
        _svs(T, sets, by)
    end
end

_get_id(x) = x.id

function _append_to_result(D::Dict, res::KnnResult)
    for (i, w) in D
        push!(res, i, 1.0 - w)  # cosine distance
    end

    res
end

function search_with_intersection(invindex::InvIndex, q::SVEC, res::KnnResult; ignore_lists_larger_than::Int=10_000)
    # normalize!(q) # we expect a normalized q
    L = PostList[]
    for (id, weight) in q
        list = get(invindex.lists, -id, EMPTY_POSTING_LIST)
        if length(list) == 0
            list = get(invindex.lists, id, EMPTY_POSTING_LIST)
        end
        if length(list) > 0 && length(list) < ignore_lists_larger_than
            push!(L, list)
        end
    end

    I = intersection(L, _get_id)
    D = SVEC()
    output = PostList()
    for (sym, weight) in q
        list = get(invindex.lists, sym, EMPTY_POSTING_LIST)
        if length(list) > 0 && length(list) < ignore_lists_larger_than
            empty!(output)
            for e in baezayates(I, _get_id, list, _get_id, output)
                D[e.id] = get(D, e.id, 0.0) + weight * e.weight
            end
        end
    end

    _append_to_result(D, res)
end

function search_with_union(invindex::InvIndex, q::SVEC, res::KnnResult; ignore_lists_larger_than::Int=10_000)
    D = SVEC()
    # normalize!(q) # we expect a normalized q 
    for (sym, weight) in q
        lst = get(invindex.lists, sym, EMPTY_POSTING_LIST)
        if length(lst) > 0 && length(lst) < ignore_lists_larger_than
            for e in lst
                D[e.id] = get(D, e.id, 0.0) + weight * e.weight
            end
        end
    end

    _append_to_result(D, res)
end

function search_with_one_error(invindex::InvIndex, q::SVEC, res::KnnResult; ignore_lists_larger_than::Int=10_000)
    D = Dict{Int,Float64}(p.id => p.dist for p in res)

    for (term, weight) in collect(q)
        delete!(q, term)
        empty!(res)
        
        search_with_intersection(invindex, q, res; ignore_lists_larger_than=ignore_lists_larger_than)
        for p in res
            D[p.id] = min(p.dist, get(D, p.id, typemax(Float64)))
        end

        q[term] = weight
    end

    empty!(res)
    for (id, dist) in D
        push!(res, id, dist)
    end

    res
end
