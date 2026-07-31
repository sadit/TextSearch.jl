# This file is part of TextSearch.jl

export BM25InvertedFile, search, filter_lists!, append_items!, push_item!, InvertedFileContext

import SimilaritySearch: search, append_items!, push_item!, database, distance

using Intersections
using StatsBase


"""
    BM25InvertedFile{AdjType<:AbstractAdjList} <: AbstractInvertedFile

An inverted-file index (built on top of `InvertedFiles.jl`) that answers approximate/exact
top-k queries ranked by BM25 relevance. Build it with [`BM25InvertedFile(voc)`](@ref
BM25InvertedFile), populate it with [`append_items!`](@ref)/[`push_item!`](@ref), and
query it with `search` (from `SimilaritySearch.jl`).

# Fields
- `voc`: the [`Vocabulary`](@ref) shared by every indexed document (also used to
  tokenize/encode query text).
- `bm25`: the [`BM25`](@ref) scorer used to rank matches.
- `adj`: the adjacency list of posting lists (one per token id), mapping each token to
  the documents containing it and their term frequency.
- `doclens`: number of tokens per indexed document.
"""
struct BM25InvertedFile{AdjType<:AbstractAdjList} <: AbstractInvertedFile
    voc::Vocabulary
    bm25::BM25
    adj::AdjType
    doclens::Vector{Int32}  ## number of tokens per document
end

function Base.show(io::IO, invfile::BM25InvertedFile; prefix="", indent="  ")
    println(io, prefix, "BM25InvertedFile:")
    prefix = indent * prefix
    println(io, prefix, "length: ", length(invfile))
    println(io, prefix, "adj: ", typeof(invfile.adj))
    show(io, invfile.voc; prefix, indent)
    show(io, invfile.bm25; prefix, indent)
end

Base.length(invfile::BM25InvertedFile) = length(invfile.doclens)
database(invfile::BM25InvertedFile) = error("database() is not accesible in BM25InvertedFile")
distance(::BM25InvertedFile) = error("BM25InvertedFile is not a metric index")

"""
    BM25InvertedFile(voc::Vocabulary; k1=1.2f0, b=0.75f0, δ=1f0)

Creates an empty [`BM25InvertedFile`](@ref), fitting its [`BM25`](@ref) scorer from `voc`
(see [`BM25(voc)`](@ref BM25) for `k1`/`b`/`δ`). Populate it with
[`append_items!`](@ref)/[`push_item!`](@ref).
"""
function BM25InvertedFile(voc::Vocabulary;  k1=1.2f0, b=0.75f0, δ=1f0)
    bm25 = BM25(voc; k1, b, δ)

    BM25InvertedFile(
        voc,
        bm25,
        AdjList(IdIntWeight, vocsize(voc)),
        Vector{Int32}(undef, 0),
    )
end

"""
    filter_lists!(
        idx::BM25InvertedFile;
        list_min_length_for_checking::Int=96,
        list_max_allowed_length::Int=1024,
        doc_min_freq::Int=1,
        doc_max_freq::Int=128,
        always_sort::Bool=false
    )

Prunes each posting list of `idx` in place, once it is already populated. Lists shorter
than `list_min_length_for_checking` are left untouched (optionally sorted by document id
when `always_sort=true`). Longer lists are filtered to entries whose term frequency lies
in `[doc_min_freq, doc_max_freq]`, then truncated to the `list_max_allowed_length`
highest-frequency entries — this both discards overly rare/common (likely noisy) postings
and bounds the cost of scanning very long lists at query time. Returns `idx`.
"""
function filter_lists!(
        idx::BM25InvertedFile;
        list_min_length_for_checking::Int=96,
        list_max_allowed_length::Int=1024,
        doc_min_freq::Int=1,
        doc_max_freq::Int=128,
        always_sort::Bool=false
    )
    adj = idx.adj
    @assert adj isa AdjList
    buff = IdIntWeight[]
    sizehint!(buff, list_max_allowed_length)

    for i in eachindex(adj)
        L = neighbors(adj, i)
        n = length(L)
        n == 0 && continue
        if n < list_min_length_for_checking
            always_sort && sort!(L, by=p->p.id)
            continue
        end
        empty!(buff)
        for item in L
            if doc_min_freq <= item.weight <= doc_max_freq
                push!(buff, item)
            end
        end

        sort!(buff, by=p->p.weight, rev=true)
        if length(buff) > list_max_allowed_length
            resize!(buff, list_max_allowed_length)
        end

        sort!(buff, by=p->p.id)
        resize!(L, length(buff))
        L .= buff
    end

    idx
end

"""
    append_items!(idx::BM25InvertedFile, ctx::InvertedFileContext, corpus; kwargs...)

Adds every document in `corpus` to `idx`, computing each one's bag of words under
`idx.voc` first. `corpus` can hold raw text (`AbstractString`), already-tokenized
[`TokenizedText`](@ref), or pre-tokenized string vectors; a corpus of already-computed
[`BOW`](@ref)s is accepted directly by the generic `SimilaritySearch.append_items!`
method without going through this conversion. See also [`push_item!`](@ref).
"""
function append_items!(idx::BM25InvertedFile, ctx::InvertedFileContext, corpus::AbstractVector{T}; kwargs...) where {T<:AbstractString}
    append_items!(idx, ctx, VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end

function append_items!(idx::BM25InvertedFile, ctx::InvertedFileContext, corpus::AbstractVector{T}; kwargs...) where {T<:TokenizedText}
    append_items!(idx, ctx, VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end

function append_items!(idx::BM25InvertedFile, ctx::InvertedFileContext, corpus::AbstractVector{T}; kwargs...) where {T<:AbstractVector{<:AbstractString}}
    append_items!(idx, ctx, VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end

"""
    push_item!(idx::BM25InvertedFile, ctx::InvertedFileContext, doc)

Adds a single document `doc` to `idx`, computing its bag of words under `idx.voc` first.
`doc` can be raw text (`AbstractString`), already-tokenized [`TokenizedText`](@ref), or a
pre-tokenized string vector; an already-computed [`BOW`](@ref) is accepted directly by the
generic `SimilaritySearch.push_item!` method without going through this conversion.
See also [`append_items!`](@ref).
"""
function push_item!(idx::BM25InvertedFile, ctx::InvertedFileContext, doc::T) where {T<:Union{AbstractString,AbstractVector,TokenizedText}}
    push_item!(idx, ctx, bagofwords(idx.voc, doc))
end

function SimilaritySearch.push_item!(idx::BM25InvertedFile, ctx::InvertedFileContext, obj; docID::UInt32=length(idx) + 1, tol::Float64=1e-6)
    len = bm25_internal_push_object!(idx, ctx, docID, obj, tol)
    for (tokenID, _) in sparseiterator(obj)
        N = neighbors(idx.adj, tokenID)
        N === nothing && continue
        sort!(N)
    end

    push!(idx.doclens, len)
    !isnothing(idx.db) && push_item!(idx.db, obj)
    LOG(ctx.logger, :push_item!, idx, ctx, docID, docID)
    idx
end

function bm25_internal_push_object!(idx::BM25InvertedFile, ctx::InvertedFileContext, docID::Integer, obj, tol::Float64)
    len = 0
    @inbounds for (tokenID, freq) in InvertedFiles.sparseiterator(obj)  # obj is a BOW-like struct
        freq < tol && continue
        len += freq
        SimilaritySearch.add!(idx.adj, tokenID, (IdIntWeight(docID, freq),))
    end

    len
end

function InvertedFiles.parallel_append!(idx::BM25InvertedFile, ctx::InvertedFileContext, db::AbstractDatabase, startID::Int, n::Int, tol::Float64)
    resize!(idx.doclens, startID + n)
    minbatch = getminbatch(n)

    @batch minbatch = minbatch per = thread for i in 1:n
        docID = i + startID
        idx.doclens[docID] = bm25_internal_push_object!(idx, ctx, docID, db[i], tol)
    end

    @batch minbatch = minbatch per = thread for i in 1:length(idx.adj)
        N = neighbors(idx.adj, i)
        N === nothing && continue
        sort!(N, by=p -> p.id)
    end

    idx
end