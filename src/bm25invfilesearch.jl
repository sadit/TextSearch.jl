# This file is part of TextSearch.jl

function SimilaritySearch.search(idx::BM25InvertedFile, qtext::AbstractString, res::KnnResult; pools=nothing)
	v = vectorize(idx.voc, idx.textconfig, qtext)
	search(idx, v, res)
end

"""
search(callback::Function, idx::BM25InvertedFile, Q, P; t=1)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(docID, dist)`

# Arguments:
- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
- `P`: a vector of starting positions in Q (initial state as ones)
"""
function InvertedFiles.search_invfile(callback::Function, idx::BM25InvertedFile, Q, P_, t)
	#@show "----------------"
	umerge(Q, P_; t) do L, P, m
		@inbounds docID = L[1].list[P[1]].id
		doclen = idx.doclens[docID]
		S = 0f0
		@inbounds @simd for i in 1:m
			freq = L[i].list[P[i]].weight
			tokndocs = ndocs(idx.voc, L[i].tokenID)
			s = tokenscore(idx.bm25, tokndocs, doclen, freq)
			#@show i, docID, idx.voc[L[i].tokenID], s, tokndocs, doclen, freq
			S -= s
		end

		callback(docID, S)
	end
end
