# This file is part of TextSearch.jl

module FullText

using SimilaritySearch
using SimilaritySearch: getminbatch
using SimilaritySearch.Intersections
using SimilaritySearch.InvertedFiles
using SimilaritySearch.Special.Sparse: SparseVecView, SparseVectorLike
using SparseArrays: SparseVector, sparsevec
using StatsBase
using LinearAlgebra: normalize!

using ..TextSearch: Vocabulary, VectorModel, TextConfig, TokenizedText, BOW,
                    vocsize, vectorize, vectorize_corpus, bagofwords, bagofwords_corpus,
                    LocalWeighting, GlobalWeighting, TfWeighting, IdfWeighting,
                    expand_synonyms!

using ..BM25
using ..BM25: BM25InvertedFile, BM25Scorer, bm25score, tokenscore

export TextInvertedFile, BM25InvertedFile, BM25Scorer, bm25score, tokenscore,
       search, append_items!, push_item!

include("textinvfile.jl")

end
