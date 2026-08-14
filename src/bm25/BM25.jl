# This file is a part of TextSearch.jl

module BM25

using SimilaritySearch
using SimilaritySearch: getminbatch
using StatsBase

using SimilaritySearch.Intersections
import SimilaritySearch.Intersections: onmatch!
using SimilaritySearch.InvertedFiles
import SimilaritySearch.InvertedFiles: getpositions, getcontainer, identiterator, PostingList, _parallel_append!, internal_parallel_prepare_append!
using SimilaritySearch.Special.Sparse: SparseVecView, SparseVectorLike
using SparseArrays: sparsevec, SparseVector
using ..TextSearch: Vocabulary, trainsize, avgdoclen, ndocs, vocsize, bagofwords, bagofwords_corpus, TokenizedText, BOW

pairiterator(d::Dict) = d
pairiterator(d::SparseVecView) = zip(d.I, d.F)
pairiterator(d::SparseVector) = zip(SparseArrays.nonzeroinds(d), SparseArrays.nonzeros(d))
pairiterator(d) = d

include("scorer.jl")
include("invfile.jl")
include("invfilesearch.jl")

end
