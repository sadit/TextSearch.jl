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
  search(accept_posting_list::Function, idx::BM25InvertedFile, ctx::InvertedFileContext, qtext::AbstractString, res::AbstractKnn
  search(idx::BM25InvertedFile, ctx::InvertedFileContext, qtext::AbstractString, res::AbstractKnn

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(docID, dist)`
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
  SimilaritySearch.add_distance_evaluations!(res, costevals)
  res
end

function SimilaritySearch.search(idx::BM25InvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn)
  search(idx, ctx, q, res) do lst
    true
  end
end
