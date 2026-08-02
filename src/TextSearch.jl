# This file is a part of TextSearch.jl

module TextSearch

import Base: broadcastable
import StatsBase: fit, predict
using Accessors
using SimilaritySearch, LinearAlgebra, SparseArrays
using SimilaritySearch: getminbatch

using Polyester, ProgressMeter

include("intersections/Intersections.jl")
using .Intersections

include("invertedfiles/InvertedFiles.jl")
using .InvertedFiles
using .InvertedFiles: getcontext, getpositions, InvertedFileContext, IdWeight, IdIntWeight

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

A weighted sparse vector: a `token id => weight` mapping, as produced by
[`vectorize`](@ref)/[`vectorize!`](@ref). Arithmetic (`+`, `-`, `*`, `/`), norms, and
distances for `SVEC`/`BOW`-like `Dict`s are defined in `dvec.jl` (see [`normalize!`](@ref),
[`dot`](@ref), [`norm`](@ref), [`centroid`](@ref)).

# Example

```julia
julia> SVEC(0x00000001 => 0.5f0)
Dict{UInt32, Float32}(0x00000001 => 0.5)
```
"""
const SVEC = Dict{UInt32,Float32}

export SVEC, BOW


struct TextSearchBuffer
    normtext::Vector{Char}
    tokens::Vector{String}
    unigrams::Vector{String}
    io::IOBuffer
    bow::BOW
    vec::SVEC

    function TextSearchBuffer(n=128)
        normtext = Char[]
        tokens = UInt64[]
        unigrams = String[]
        io = IOBuffer()
        bow = BOW()
        vec = SVEC()

        sizehint!(normtext, n)
        sizehint!(tokens, n)
        sizehint!(unigrams, n)
        sizehint!(bow, n)
        sizehint!(vec, n)

        new(normtext, tokens, unigrams, io, bow, vec)
    end
end

const TEXT_SEARCH_CACHES = Channel{TextSearchBuffer}(Inf)

function Base.empty!(buff::TextSearchBuffer; normtext::Bool=true, tokens::Bool=true, unigrams::Bool=true, bow::Bool=true, vec::Bool=true)
    normtext && empty!(buff.normtext)
    tokens && empty!(buff.tokens)
    unigrams && empty!(buff.unigrams)
    bow && empty!(buff.bow)
    vec && empty!(buff.vec)
end

function __init__()
    for _ in 1:2*Threads.nthreads()+4
        put!(TEXT_SEARCH_CACHES, TextSearchBuffer())
    end
end

@inline function textbuffer(f)
    buff = take!(TEXT_SEARCH_CACHES)
    try
        f(buff)
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

include("tokentrans.jl")
include("textconfig.jl")
include("normalize.jl")
include("tokenize.jl")
include("voc.jl")
include("updatevoc.jl")
include("tokcorpus.jl")
include("bow.jl")
include("sparseconversions.jl")
include("vmodel.jl")
include("emodel.jl")
include("bm25.jl")
include("bm25invfile.jl")
include("bm25invfilesearch.jl")
include("approxvoc.jl")
include("deprecated.jl")

end
