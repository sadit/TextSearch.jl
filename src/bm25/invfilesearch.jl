# This file is part of TextSearch.jl

struct BM25InvFileOutput{InvFileType<:BM25InvertedFile}
    idx::InvFileType
    res::KnnSorted
end

function Intersections.onmatch!(output::BM25InvFileOutput, L::T, P, m::Int) where T
		@inbounds docID = L[1].list[P[1]].id
    idx = output.idx
		doclen = idx.doclens[docID]
		S = 0f0
		@inbounds @simd for i in 1:m
			freq = L[i].list[P[i]].weight
			tokndocs = ndocs(idx.voc, L[i].tokenID)
			s = tokenscore(idx.bm25, tokndocs, doclen, freq)
			# @show i, docID, idx.voc[L[i].tokenID], s, tokndocs, doclen, freq
			S -= s
		end

    push_item!(output.res, IdDist(docID, S))
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

  costevals = xmerge!(BM25InvFileOutput(idx, res), Q, P; t)
  SimilaritySearch.add_distance_evaluations!(ctx, costevals)
  res
end

function SimilaritySearch.search(idx::BM25InvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn)
  search(idx, ctx, q, res) do lst
    true
  end
end
