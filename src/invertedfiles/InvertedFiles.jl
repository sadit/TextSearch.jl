# This file is part of InvertedFiles.jl

module InvertedFiles
using ..Intersections
import SimilaritySearch:
    search, index!, append_items!, push_item!
using SimilaritySearch
using SimilaritySearch: Dist, AbstractContext, getminbatch

using Base.Threads: SpinLock

export InvertedFileContext, getcontext
include("idweight.jl")
include("sortedintset.jl")
include("plists.jl")

struct InvertedFileContext{A,B,C,D} <: AbstractContext
    logger::AbstractLog
    minbatch::Int
    parallel_block::Int
    maxbatches::Int32
    batchid::Int32
    costdist::Vector{Int}
    costblk::Vector{Int}
    positions::A
    cont_u32::B
    cont_iw::C
    cont_iiw::D
    knns::Matrix{IdWeight}
end

#=function InvertedFileContext(
        ctx::InvertedFileContext{KnnType};
        logger = ctx.loggger,                 
        minbatch = ctx.minbatch,
        parallel_block = ctx.parallel_block,
        positions=ctx.positions,
        cont_u32=ctx.cont_u32,
        cont_iw=ctx.cont_iw,
        cont_iiw=ctx.cont_iiw,
        knns=ctx.knns
    ) where KnnType
    InvertedFileContext{KnnType}(ctx, logger, minbatch, parallel_block, positions, cont_u32, cont_iw, cont_iiw, knns) 
end=#

function InvertedFileContext(;
        logger = InformativeLog(dt=1.0),
        minbatch = 0,
        parallel_block = 256,
        maxbatches::Integer = 8Threads.nthreads(),
        batchid::Integer = 1,
        costdist::Vector{Int} = zeros(Int, maxbatches),
        costblk::Vector{Int} = zeros(Int, maxbatches),
        positions = [Vector{UInt32}(undef, 32) for _ in 1:Threads.maxthreadid()],
        cont_u32 = [Vector{PostingList{Vector{UInt32}}}(undef, 32) for _ in 1:Threads.maxthreadid()],
        cont_iw = [Vector{PostingList{Vector{IdWeight}}}(undef, 32) for _ in 1:Threads.maxthreadid()],
        cont_iiw = [Vector{PostingList{Vector{IdIntWeight}}}(undef, 32) for _ in 1:Threads.maxthreadid()],
        knns = zeros(IdWeight, 64, Threads.maxthreadid())
    )

    InvertedFileContext(logger, minbatch, parallel_block, convert(Int32, maxbatches), convert(Int32, batchid),
                         costdist, costblk, positions, cont_u32, cont_iw, cont_iiw, knns)
end

SimilaritySearch.knnqueue(::InvertedFileContext, arg) = knnqueue(KnnSorted, arg)

include("invfile.jl")
include("winvfile.jl")
include("binvfile.jl")
include("invfilesearch.jl")
include("winvfilesearch.jl")
include("binvfilesearch.jl")

DEFAULT_CACHE_INVFILES = Ref(InvertedFileContext())

function __init__()
    n = Threads.nthreads()
    DEFAULT_CACHE_INVFILES[] = InvertedFileContext()
end

getcontext(invfile::AbstractInvertedFile) = DEFAULT_CACHE_INVFILES[]

end
