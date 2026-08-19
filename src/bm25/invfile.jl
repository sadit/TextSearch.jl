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
  [`append_items!`](@ref); may hold more documents than have actually been indexed -- see `len`.
- `len`: number of documents already indexed (postings built); may be less than
  `length(database(idx))` if `db` was grown directly (e.g. `push_item!(database(idx), docvec)`)
  without a following `index!` call to catch up. Growing `db` directly with pre-computed
  `SparseVecView`s and then calling `index!(invfile, ctx)` is supported and builds postings from
  the already-stored vectors; the raw-text/BOW-taking `append_items!`/`push_item!` methods remain
  fused (encode+store+register in one pass) for efficiency.

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
    len::Ref{Int64}         ## number of documents already indexed (postings built)
end

function Base.show(io::IO, invfile::BM25InvertedFile; prefix="", indent="  ")
    println(io, prefix, "BM25InvertedFile:")
    prefix = indent * prefix
    println(io, prefix, "length: ", length(invfile))
    println(io, prefix, "adj: ", typeof(invfile.adj))
    show(io, invfile.voc; prefix, indent)
    show(io, invfile.bm25; prefix, indent)
end

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
        Ref(Int64(0)),
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
    idx.len[] += 1
    LOG(ctx.logger, :add!, idx, ctx, docID, docID)
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

function _bm25_prepare_grow!(idx::BM25InvertedFile, new_size::Integer)
    resize!(idx.doclens, new_size)
    resize!(idx.db.vecs, new_size)
end

"""
    _bm25_fused_index_and_grow!(idx, ctx, items, startID, n, tol=1e-6)

Encodes each raw/BOW document in `items[1:n]` into its `SparseVecView` term-frequency
representation, registers its postings into `idx.adj`, and stores the vector into `idx.db` --
all in a single pass (this is the "fused" behavior `append_items!`/`push_item!` use for raw-text/
BOW input, kept for efficiency: unlike the generic `InvertedFile`, `BM25InvertedFile`'s `db`
element type is *derived* from its input, so there is no cheap way to grow `db` ahead of
indexing for this path -- see [`_index_block!`](@ref) for the decoupled path instead, used when
`db` is grown directly with already-encoded `SparseVecView`s).
"""
function _bm25_fused_index_and_grow!(idx::BM25InvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, startID::Int, n::Int, tol::Float64=1e-6)
    _bm25_prepare_grow!(idx, startID + n)
    minbatch = getminbatch(n)

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:n
        docID = i + startID
        len, docvec = bm25_internal_push_object!(idx, docID, items[i], tol)
        idx.doclens[docID] = len
        idx.db[docID] = docvec
    end

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:length(idx.adj)
        N = neighbors(idx.adj, i)
        N === nothing && continue
        sort!(N)
    end

    idx
end

function append_items!(idx::BM25InvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, n=length(items); tol::Float64=1e-6)
    startID = length(idx)
    _bm25_fused_index_and_grow!(idx, ctx, items, startID, n, tol)
    idx.len[] = startID + n
    n > 0 && LOG(ctx.logger, :add!, idx, ctx, startID + 1, length(idx))
    idx
end

"""
    bm25_register_postings!(idx::BM25InvertedFile, docID::Integer, docvec) -> doclen

Registers an already-encoded document vector `docvec` (anything `pairiterator`-compatible,
typically the `SparseVecView` already stored at `idx.db[docID]`) into `idx.adj` under `docID`,
and returns its token count (doclen). This is the postings-only half of
[`bm25_internal_push_object!`](@ref) -- it does not parse/build a `SparseVecView`, since `docvec`
is assumed to already be one (e.g. read back from `idx.db` by [`_index_block!`](@ref)).
"""
function bm25_register_postings!(idx::BM25InvertedFile, docID::Integer, docvec)
    len = 0
    @inbounds for (tokenID, freq) in pairiterator(docvec)
        len += freq
        SimilaritySearch.add!(idx.adj, tokenID, (convert(UInt32, docID),))
    end
    len
end

"""
    _index_block!(idx::BM25InvertedFile, ctx::InvertedFileContext, sp::Int, n::Int)

Decoupled indexing path: builds postings and `doclens` for `idx.db[sp:n]`, reading the
already-encoded `SparseVecView`s directly out of `db` (no re-tokenization). Used by
[`index!`](@ref) when `db` was grown directly (e.g. `push_item!(database(idx), docvec)`) rather
than through the fused `append_items!`/`push_item!` entry points.
"""
function _index_block!(idx::BM25InvertedFile, ctx::InvertedFileContext, sp::Int, n::Int)
    resize!(idx.doclens, n)
    minbatch = getminbatch(n - sp + 1)

    @BATCHES minbatch scheduler=ctx.scheduler for docID in sp:n
        idx.doclens[docID] = bm25_register_postings!(idx, docID, idx.db[docID])
    end

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:length(idx.adj)
        N = neighbors(idx.adj, i)
        N === nothing && continue
        sort!(N)
    end

    idx
end
