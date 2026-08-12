# This file is a part of TextSearch.jl

module BM25

using SimilaritySearch
using SimilaritySearch: getminbatch
using StatsBase

using SimilaritySearch.Intersections
import SimilaritySearch.Intersections: onmatch!
using SimilaritySearch.InvertedFiles
import SimilaritySearch.InvertedFiles: getpositions, getcontainer, sparseiterator, PostingList, parallel_append!
using SimilaritySearch.Special.Sparse: SparseVecView, SparseVectorLike
using SparseArrays: sparsevec
using ..TextSearch: Vocabulary, trainsize, avgdoclen, ndocs, vocsize, bagofwords, bagofwords_corpus, TokenizedText, BOW

include("scorer.jl")
include("invfile.jl")
include("invfilesearch.jl")

end
