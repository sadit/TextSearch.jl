# This file is part of TextSearch.jl

struct BM25InvFileOutput{InvFileType<:BM25InvertedFile,QType<:SparseVectorLike}
    idx::InvFileType
    query::QType
    res::KnnSorted
end

# `m`/`L[2:m]` aren't needed anymore: `bm25score` recomputes the query-doc intersection
# itself (see src/bm25/scorer.jl) instead of scoring just the `m` posting lists the outer
# merge already matched. Simpler, at the cost of redoing that intersection per candidate
# instead of reusing the merge's own -- see the discussion that led to this trade-off.
function onmatch!(output::BM25InvFileOutput, L::T, P, m::Int) where T
    @inbounds docID = L[1][P[1]]
    idx = output.idx
    S = -bm25score(idx.bm25, idx.voc, output.query, idx.db[docID])
    push_item!(output.res, IdDist(docID, S))
end

"""
    bm25_query_vector(idx::BM25InvertedFile, q)

Converts `q` (a bag of words -- `BOW`/`Dict`, or anything else [`sparseiterator`](@ref)
accepts) into a `SparseVector`, for [`bm25score`](@ref) to use in [`onmatch!`](@ref).
Passes `q` through unchanged if it's already a `SparseVectorLike`.
"""
bm25_query_vector(idx::BM25InvertedFile, q::SparseVectorLike) = q

function bm25_query_vector(idx::BM25InvertedFile, q)
    I = Int32[]
    F = Int32[]
    for (tokenID, freq) in sparseiterator(q)
        push!(I, convert(Int32, tokenID))
        push!(F, convert(Int32, freq))
    end

    sparsevec(I, F, vocsize(idx.voc))
end

"""
    search(idx::BM25InvertedFile, ctx::InvertedFileContext, qtext, res::AbstractKnn)
    search(accept_posting_list::Function, idx::BM25InvertedFile, ctx::InvertedFileContext, qtext, res::AbstractKnn; t::Int=1)

Solves a top-k query over `idx` for `qtext` (raw text, [`TokenizedText`](@ref), or an
already-computed bag of words), accumulating matches into `res` (an `AbstractKnn`, e.g.
`KnnResult`/`KnnSorted`). Documents are ranked by BM25 score (stored internally as a
negative value in `res`, so lower "distance" still means better match, consistent with
`SimilaritySearch.jl`'s convention).

The `accept_posting_list` variant additionally receives a predicate called with each
query token's posting list before scanning it, letting the caller skip lists (e.g. to
implement stopword-like filtering at query time); it defaults to accepting every list.
Returns `res`.

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> invfile = BM25InvertedFile(voc);

julia> ctx = InvertedFileContext();

julia> append_items!(invfile, ctx, corpus);

julia> res = knnqueue(KnnSorted, 2);

julia> search(invfile, ctx, "hello", res) do lst
           true
       end;

julia> collect(IdView(res))
UInt32[0x00000001, 0x00000002]
```
"""
function SimilaritySearch.search(accept_posting_list::Function, idx::BM25InvertedFile, ctx::InvertedFileContext, qtext::T, res::AbstractKnn) where {T<:Union{AbstractString,TokenizedText}}
    q = bagofwords(idx.voc, qtext)
    search(accept_posting_list, idx, ctx, q, res)
end

function SimilaritySearch.search(accept_posting_list::Function, idx::BM25InvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn; t::Int=1)
  Q = select_posting_lists(accept_posting_list, idx, ctx, q)
  length(Q) == 0 && return res
  P = getpositions(length(Q), ctx)
  query_vec = bm25_query_vector(idx, q)

  costevals = xmerge!(BM25InvFileOutput(idx, query_vec, res), Q, P; t)
  SimilaritySearch.add_distance_evaluations!(ctx, costevals)
  res
end

function SimilaritySearch.search(idx::BM25InvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn)
  search(idx, ctx, q, res) do lst
    true
  end
end
