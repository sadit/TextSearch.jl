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

function Base.empty!(buff::TextSearchBuffer)
    empty!(buff.normtext)
    empty!(buff.tokens)
    empty!(buff.unigrams)
    empty!(buff.bow)
    empty!(buff.vec)
end

function __init__()
    for _ in 1:Threads.nthreads()
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

include("textconfig.jl")
include("normalize.jl")
include("tokenize.jl")
include("voc.jl")
include("bow.jl")
include("sparseconversions.jl")
include("vmodel.jl")
include("emodel.jl")
include("bm25.jl")
include("bm25invfile.jl")
include("bm25invfilesearch.jl")
include("langmodel.jl")
include("corpuslangmodel.jl")
include("io.jl")

end
