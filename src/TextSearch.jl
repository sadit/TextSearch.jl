# This file is a part of TextSearch.jl

module TextSearch

import Base: broadcastable
import StatsBase: fit, predict
import SimilaritySearch: search, append_items!, push_item!, database, distance
using Accessors
using SimilaritySearch, LinearAlgebra, SparseArrays
using SimilaritySearch: getminbatch

using ProgressMeter

using SimilaritySearch: Intersections, InvertedFiles
using SimilaritySearch.InvertedFiles: getcontext, getpositions, InvertedFileContext, IdWeight, IdIntWeight
export Intersections, InvertedFiles

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

"""
    SVEC = Dict{UInt32,Float32}

A lightweight `token id => weight` mapping. Note that [`vectorize`](@ref)/[`vectorize!`](@ref)
actually return a `SparseVector{Float32,Int32}` (from `SparseArrays.jl`), not a `SVEC` —
arithmetic (`+`, `-`, `*`, `/`), norms, and distances (see [`normalize!`](@ref),
[`dot`](@ref), [`norm`](@ref), [`centroid`](@ref)) are defined for `SparseVector`, not for
`SVEC`/[`BOW`](@ref).

# Example

```julia
julia> SVEC(0x00000001 => 0.5f0)
Dict{UInt32, Float32}(0x00000001 => 0.5)
```
"""
const SVEC = Dict{UInt32,Float32}

export SVEC, BOW

function __init__()
    for _ in 1:2*Threads.nthreads()+4
        put!(BOW_CACHES, sizehint!(BOW(), 128))
        put!(VECTORIZE_CACHES, VectorizeBuffer())
    end
end

include("voc.jl")
include("updatevoc.jl")
include("bow.jl")
include("vmodel.jl")
include("emodel.jl")

include("bm25/BM25.jl")
using .BM25
export BM25Scorer, bm25score, tokenscore, BM25InvertedFile, filter_lists!, search, append_items!, push_item!

include("deprecated.jl")

end
