# This file is part of TextSearch.jl

export BM25InvertedFile, search, append_items!, push_item!, InvertedFileContext

import SimilaritySearch: search, append_items!, push_item!, database, distance

using StatsBase


"""
    BM25InvertedFile{AdjType<:AbstractAdjList,DbType<:AbstractDatabase} <: AbstractInvertedFile

An inverted-file index (built on top of `SimilaritySearch.InvertedFiles`) that answers
approximate/exact top-k queries ranked by BM25 relevance. Build it with
[`BM25InvertedFile(voc)`](@ref BM25InvertedFile), populate it with
[`append_items!`](@ref)/[`push_item!`](@ref), and query it with `search` (from
`SimilaritySearch.jl`).

Follows the same design as `SimilaritySearch.InvertedFiles.InvertedFile`: `adj` only ever
stores plain document ids (`AdjType`'s element type is `UInt32`, exactly like the generic
`InvertedFile`); every other per-document detail needed to score a match -- term
frequencies -- is fetched from `db` instead of being duplicated into the posting lists.

# Fields
- `voc`: the [`Vocabulary`](@ref) shared by every indexed document (also used to
  tokenize/encode query text).
- `bm25`: the [`BM25Scorer`](@ref) used to rank matches.
- `adj`: the adjacency list of posting lists (one per token id), mapping each token to
  the ids of the documents containing it.
- `doclens`: number of tokens per indexed document.
- `db`: each indexed document's term-frequency vector, one `SparseVecView` (token id =>
  `UInt32` frequency) per document, always populated by [`push_item!`](@ref)/
  [`append_items!`](@ref).

# Example

```julia
julia> using SimilaritySearch

julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> invfile = BM25InvertedFile(voc);

julia> ctx = InvertedFileContext();

julia> append_items!(invfile, ctx, corpus);

julia> length(invfile)
3

julia> res = knnqueue(KnnSorted, 2);

julia> search(invfile, ctx, "hello", res);

julia> collect(IdView(res))
UInt32[0x00000001, 0x00000002]
```
"""
struct BM25InvertedFile{AdjType<:AbstractAdjList,DbType<:AbstractDatabase} <: AbstractInvertedFile
    voc::Vocabulary
    bm25::BM25Scorer
    adj::AdjType
    doclens::Vector{Int32}  ## number of tokens per document
    db::DbType              ## per-document term-frequency vectors
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
database(invfile::BM25InvertedFile) = invfile.db
distance(::BM25InvertedFile) = error("BM25InvertedFile is not a metric index")

# `BM25InvertedFile`'s posting lists only ever store plain document ids (`AdjList{UInt32}`,
# like the generic `InvertedFile`), so `getcontainer` and its `PostingList{Vector{UInt32}}`
# come straight from `SimilaritySearch.InvertedFiles` with no local overrides. `dist` is the
# one thing `select_posting_lists`'s generic implementation needs that `BM25InvertedFile`
# doesn't have (BM25 isn't a metric index, see `distance` above), so it gets this small
# override instead -- otherwise identical to the generic one, just `identiterator(q)` instead
# of `identiterator(idx.dist, q)`.
function select_posting_lists(idx::BM25InvertedFile, ctx::InvertedFileContext, q)
    Q = getcontainer(idx, ctx)
    @inbounds for tokenID in identiterator(q)
        tokenID == 0 && continue
        N = neighbors(idx.adj, tokenID)
        N === nothing && continue
        if length(N) > 0
            L = PostingList(N, convert(UInt32, tokenID))
            push!(Q, L)
        end
    end

    Q
end

"""
    BM25InvertedFile(voc::Vocabulary; k1=1.2f0, b=0.75f0, δ=1f0)

Creates an empty [`BM25InvertedFile`](@ref), fitting its [`BM25Scorer`](@ref) from `voc`
(see [`BM25Scorer(voc)`](@ref BM25Scorer) for `k1`/`b`/`δ`). Populate it with
[`append_items!`](@ref)/[`push_item!`](@ref).

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> invfile = BM25InvertedFile(voc);

julia> length(invfile)
0
```
"""
function BM25InvertedFile(voc::Vocabulary;  k1=1.2f0, b=0.75f0, δ=1f0)
    bm25 = BM25Scorer(voc; k1, b, δ)

    BM25InvertedFile(
        voc,
        bm25,
        resize!(AdjList(UInt32), vocsize(voc)),
        Vector{Int32}(undef, 0),
        VectorDatabase(SparseVecView{Vector{Int32},Vector{UInt32}}[]),
    )
end

"""
    append_items!(idx::BM25InvertedFile, ctx::InvertedFileContext, corpus; kwargs...)

Adds every document in `corpus` to `idx`, computing each one's bag of words under
`idx.voc` first. `corpus` can hold raw text (`AbstractString`), already-tokenized
[`TokenizedText`](@ref), or pre-tokenized string vectors; a corpus of already-computed
[`BOW`](@ref)s is accepted directly by the generic `SimilaritySearch.append_items!`
method without going through this conversion. See also [`push_item!`](@ref).

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> invfile = BM25InvertedFile(voc);

julia> ctx = InvertedFileContext();

julia> append_items!(invfile, ctx, ["hello world", "hello there"]);

julia> length(invfile)
2
```
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

# Example

```julia
julia> corpus = ["hello world", "hello there"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> invfile = BM25InvertedFile(voc);

julia> ctx = InvertedFileContext();

julia> append_items!(invfile, ctx, corpus);

julia> push_item!(invfile, ctx, "hello again");

julia> length(invfile)
3
```
"""
function push_item!(idx::BM25InvertedFile, ctx::InvertedFileContext, doc::T) where {T<:Union{AbstractString,AbstractVector,TokenizedText}}
    push_item!(idx, ctx, bagofwords(idx.voc, doc))
end

function SimilaritySearch.push_item!(idx::BM25InvertedFile, ctx::InvertedFileContext, obj; docID::UInt32=UInt32(length(idx) + 1), tol::Float64=1e-6)
    len, docvec = bm25_internal_push_object!(idx, docID, obj, tol)
    for tokenID in identiterator(obj)
        N = neighbors(idx.adj, tokenID)
        N === nothing && continue
        sort!(N)
    end

    push!(idx.doclens, len)
    push_item!(idx.db, docvec)
    LOG(ctx.logger, :push_item!, idx, ctx, docID, docID)
    idx
end

"""
    bm25_internal_push_object!(idx, docID, obj, tol) -> (doclen, docvec)

Registers `obj` (a pair `(tokenID, freq)` iterable) into `idx.adj` under `docID` and
builds its `SparseVecView` term-frequency representation (`docvec`, to be stored at `docID`
in `idx.db` by the caller). Returns `obj`'s total token count (`doclen`) and `docvec`.
"""
function bm25_internal_push_object!(idx::BM25InvertedFile, docID::Integer, obj, tol::Float64)
    tokenids = Int32[]
    freqs = UInt32[]
    len = 0

    @inbounds for (tokenID, freq) in pairiterator(obj)  # obj is a BOW-like struct
        freq < tol && continue
        len += freq
        push!(tokenids, convert(Int32, tokenID))
        push!(freqs, convert(UInt32, freq))
        SimilaritySearch.add!(idx.adj, tokenID, (convert(UInt32, docID),))
    end

    if !issorted(tokenids)
        perm = sortperm(tokenids)
        permute!(tokenids, perm)
        permute!(freqs, perm)
    end

    len, SparseVecView(vocsize(idx.voc), tokenids, freqs)
end

function _parallel_append!(idx::BM25InvertedFile, ctx::InvertedFileContext, db::AbstractDatabase, startID::Int, n::Int, tol::Float64)
    resize!(idx.doclens, startID + n)
    resize!(idx.db.vecs, startID + n)
    minbatch = getminbatch(n)

    @BATCHES minbatch for i in 1:n
        docID = i + startID
        len, docvec = bm25_internal_push_object!(idx, docID, db[i], tol)
        idx.doclens[docID] = len
        idx.db[docID] = docvec
    end

    @BATCHES minbatch for i in 1:length(idx.adj)
        N = neighbors(idx.adj, i)
        N === nothing && continue
        sort!(N)
    end

    idx
end
