# This file is a part of TextSearch.jl

module TextSearch

import Base: broadcastable
import StatsBase: fit, predict
using SimilaritySearch, InvertedFiles, LinearAlgebra, SparseArrays
using SimilaritySearch: getminbatch
using Polyester

include("dvec.jl")

const BOW = DVEC{UInt32,Int32}

struct TextSearchBuffer
    normtext::Vector{Char}
    tokens::Vector{String}
    unigrams::Vector{String}
    io::IOBuffer
    bow::BOW

    function TextSearchBuffer(n=128)
        normtext = Char[]
        tokens = UInt64[]
        unigrams = String[]
        bow = BOW()
        sizehint!(normtext, n)
        sizehint!(tokens, n)
        sizehint!(unigrams, n)
        sizehint!(bow, n)

        new(normtext, tokens, unigrams, IOBuffer(), bow)
    end
end

const CACHES = Channel{TextSearchBuffer}(Inf)

function Base.empty!(buff::TextSearchBuffer)
    empty!(buff.normtext)
    empty!(buff.tokens)
    empty!(buff.unigrams)
    empty!(buff.bow)
end

function __init__()
    for _ in 1:Threads.nthreads()
        put!(CACHES, TextSearchBuffer())
    end
end

include("textconfig.jl")
include("normalize.jl")
include("tokenize.jl")
include("voc.jl")
include("bow.jl")
include("sparseconversions.jl")
include("vmodel.jl")
include("emodel.jl")

end
