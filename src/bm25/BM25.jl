# This file is a part of TextSearch.jl

module BM25

using SimilaritySearch
using SimilaritySearch: getminbatch
using StatsBase

using ..Intersections
using ..InvertedFiles
using ..InvertedFiles: getcontext, getpositions, InvertedFileContext, IdWeight, IdIntWeight
using ..TextSearch: Vocabulary, trainsize, avgdoclen, ndocs, vocsize, bagofwords, bagofwords_corpus, TokenizedText, BOW

include("scorer.jl")
include("invfile.jl")
include("invfilesearch.jl")

end
