# This file is part of TextSearch.jl

using SimilaritySearch: getpools
using InvertedFiles: getcachepositions

struct BM25InvFileOutput{InvFileType<:BM25InvertedFile}
    idx::InvFileType
    res::KnnResult
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

    push_item!(output.res, IdWeight(docID, S))
end

"""
  search(accept_posting_list::Function, idx::BM25InvertedFile, qtext::AbstractString, res::KnnResult; pools=getpools(idx))
  search(idx::BM25InvertedFile, qtext::AbstractString, res::KnnResult; pools=getpools(idx))

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(docID, dist)`
"""
function SimilaritySearch.search(accept_posting_list::Function, idx::BM25InvertedFile, qtext::AbstractString, res::KnnResult; pools=getpools(idx))
    q = vectorize(idx.voc, idx.textconfig, qtext)
    search(accept_posting_list, idx, q, res; pools)
end

function SimilaritySearch.search(accept_posting_list::Function, idx::BM25InvertedFile, q, res::KnnResult; pools=getpools(idx), t::Int=1)
  Q = select_posting_lists(accept_posting_list, idx, q; pools)
  if length(Q) == 0
      return SearchResult(res, 0)
  end
  P = getcachepositions(length(Q))
  cost = xmerge!(BM25InvFileOutput(idx, res), Q, P; t)
  SearchResult(res, cost)
end

function SimilaritySearch.search(idx::BM25InvertedFile, q, res::KnnResult; pools=getpools(idx))
  search(idx, q, res; pools) do lst
    true
  end
end
