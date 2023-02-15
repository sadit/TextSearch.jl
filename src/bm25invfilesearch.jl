# This file is part of TextSearch.jl

function push_posting_list!(Q, idx::WeightedInvertedFile, tokenID, freq)
	@inbounds p = PostingList(idx.lists[tokenID], idx.freqs[tokenID], tokenID, convert(Float32, freq))
	push!(Q, p)
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
function search(callback::Function, idx::BM25InvertedFile, Q, P_, t)
	umerge(Q, P_; t) do L, P, m
		@inbounds docID = L[1].I[P[1]]
		doclen = idx.doclens[docID]
		s = 0f0
		@inbounds @simd for i in 1:m
			freq = L[i].W[P[i]]
			tokndocs = ndocs(idx.voc, L[i].tokenID)
			s -= bm25tokenscore(idx.bm25, tokndocs, doclen, freq)
		end

		callback(docID, s)
	end
end
