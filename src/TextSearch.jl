# This file is a part of TextSearch.jl

module TextSearch

import Base: broadcastable
import StatsBase: fit, predict
import SimilaritySearch: search, append_items!, push_item!, database, distance
using Accessors
using SimilaritySearch, LinearAlgebra, SparseArrays
using SimilaritySearch: getminbatch

using ProgressMeter

include("idweight.jl")

using SimilaritySearch.Intersections
using SimilaritySearch.InvertedFiles

"""
    WeightedInvertedFile(vocsize::Integer)

Creates an empty [`InvertedFile`](@ref) indexed by cosine dissimilarity (`Dist.NormCosine()`),
suitable for weighted sparse vectors (e.g. `SparseVector`s produced by [`vectorize`](@ref)).
"""
WeightedInvertedFile(vocsize::Integer) = InvertedFile(vocsize, Dist.NormCosine())

"""
    BinaryInvertedFile(vocsize::Integer, dist=Dist.Sets.Jaccard())

Creates an empty [`InvertedFile`](@ref) indexed by a set distance (e.g. `Dist.Sets.Jaccard()`,
`Dist.Sets.Dice()`, `Dist.Sets.Intersection()`, `Dist.Sets.CosineSet()`), suitable for
set/token-membership objects (sets or sorted vectors of integer ids).
"""
BinaryInvertedFile(vocsize::Integer, dist=Dist.Sets.Jaccard()) = InvertedFile(vocsize, dist)

export WeightedInvertedFile, BinaryInvertedFile, AbstractInvertedFile, InvertedFile, SortedIntSet,
       InvertedFileContext, getcontext, set_distance_evaluate, search_invfile, select_posting_lists

include("tokenizer/Tokenizer.jl")
using .Tokenizer
using .Tokenizer: borrowtokenizedtext, tokenizerbuffer, tokenize_paragraphs, tokenize_sentences
export TextConfig, NormalizationConfig, TokenizationConfig, TokenizedText, tokenize, tokenize_corpus, unigrams,
       normalize_text, isemoji, tokenize_paragraphs, tokenize_sentences,
       AbstractTokenTransformation, IdentityTokenTransformation, IgnoreStopwords, ChainTransformation, SnowballTokenTransformation, transform,
       AbstractTokenGenerator, UnigramGenerator, NWordGenerator,
       needs_unigrams, tokentag, generate!, flush_token!, alltokengenerators

include("dvec.jl")

"""
    BOW = Dict{UInt32,Int32}

A bag of words: a sparse `token id => occurrence count` mapping for a single document,
as produced by [`bagofwords`](@ref)/[`bagofwords!`](@ref).

# Example

```julia
julia> BOW(0x00000001 => 2, 0x00000002 => 1)
Dict{UInt32, Int32}(0x00000002 => 1, 0x00000001 => 2)
```
"""
const BOW = Dict{UInt32,Int32}

export BOW

function __init__()
    for _ in 1:2*Threads.nthreads()+4
        put!(BOW_CACHES, sizehint!(BOW(), 128))
        put!(VECTORIZE_CACHES, VectorizeBuffer())
    end
end

include("voc.jl")
include("updatevoc.jl")
include("bow.jl")
include("sparseconversions.jl")
include("vmodel.jl")
include("emodel.jl")

include("bm25/BM25.jl")
using .BM25
export BM25Scorer, bm25score, tokenscore, BM25InvertedFile, search, append_items!, push_item!

include("fulltext/FullText.jl")
using .FullText
export FullText, TextInvertedFile

include("lsi.jl")
using .LSI: LatentSemanticIndexing, LSIModel, indim, outdim, wordvectors, synonyms
export LSI, LatentSemanticIndexing, LSIModel, indim, outdim, wordvectors, synonyms

include("randomindexing.jl")
using .RI: RandomIndexing, RIModel, BitSketch, bitsketch, bitsketch_corpus
export RI, RandomIndexing, RIModel, BitSketch, bitsketch, bitsketch_corpus

include("deprecated.jl")

end
