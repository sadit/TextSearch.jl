# This file is part of TextSearch.jl

module RI

using Random, LinearAlgebra, SparseArrays, StatsBase
using ProgressMeter
using SimilaritySearch
using SimilaritySearch: @BATCHES, getminbatch, MatrixDatabase, AbstractDatabase
using SimilaritySearch.Special.Sparse: SparseVecView, SparseVectorLike
using SimilaritySearch.Projections: RandomProjections, gaussian, qr, packsigns!, packsigns
using SimilaritySearch.ScalarQuant: SQu8, SQgu8
import SimilaritySearch.Projections: bitsketch

using ..TextSearch: TextModel, VectorModel, Vocabulary, TextConfig,
                    GlobalWeighting, LocalWeighting, IdfWeighting, TfWeighting,
                    VECTORIZE_CACHES, VectorizeBuffer
import ..TextSearch: vectorize, vectorize!, vectorize_corpus, vocsize, gettrainsize, indim, outdim

export RandomIndexing, RIModel, indim, outdim, vocsize, gettrainsize,
       vectorize, vectorize!, vectorize_corpus, bitsketch, bitsketch_corpus, BitSketch

"""
    BitSketch

Type tag used to request binary SimHash-style bit sketches when calling [`vectorize`](@ref)
or [`vectorize_corpus`](@ref).
"""
struct BitSketch end

"""
    RandomIndexing{M<:AbstractMatrix{Float32}, VM<:VectorModel} <: TextModel

Random Indexing (RI) model that projects sparse vector representations produced
by a [`VectorModel`](@ref) into a lower-dimensional dense semantic space via random projections
([`SimilaritySearch.Projections`](@ref)), with default output dimension `maxoutdim=1024`.

# Fields
- `model`: The underlying [`VectorModel`](@ref) used to tokenize and weight text.
- `P`: Projection matrix of size `(k, m)` where `k = outdim` and `m = indim = vocsize(model)`.
- `k`: Output dimension (`k = maxoutdim`).
- `maxoutdim`: Target embedding dimension (default: 1024).
- `method`: Random projection method used (`:gaussian`, `:qr`, `:sparse_random`).
"""
struct RandomIndexing{M<:AbstractMatrix{Float32}, VM<:VectorModel} <: TextModel
    model::VM
    P::M
    k::Int
    maxoutdim::Int
    method::Symbol
end

const RIModel = RandomIndexing

"""
    RandomIndexing(model::VectorModel, corpus=nothing;
                   maxoutdim::Integer=1024,
                   method::Symbol=:gaussian,
                   rng::AbstractRNG=Random.default_rng())

Constructs a [`RandomIndexing`](@ref) model from a [`VectorModel`](@ref).

# Arguments
- `model`: The vocabulary and weighting model.
- `corpus`: Optional corpus parameter (ignored during construction, provided for API symmetry with LSI).

# Keyword Arguments
- `maxoutdim`: Target projection dimension (default: `1024`).
- `method`: Random projection algorithm:
  - `:gaussian` (default): Gaussian random projection matrix with unit-norm columns.
  - `:qr`: Orthonormal random projection matrix via QR factorization.
  - `:sparse_random` (or `:ternary`): Sparse ternary random projection (±1 with sparse support).
- `rng`: Random number generator (default: `Random.default_rng()`).
"""
function RandomIndexing(
    model::VectorModel,
    corpus=nothing;
    maxoutdim::Integer=1024,
    method::Symbol=:gaussian,
    rng::AbstractRNG=Random.default_rng()
)
    m = vocsize(model)
    m > 0 || throw(ArgumentError("model vocabulary is empty (vocsize = 0)"))
    k = Int(maxoutdim)
    k > 0 || throw(ArgumentError("Output dimension maxoutdim must be positive, got $k"))

    P = if method === :gaussian
        rp = gaussian(rng, Float32, m, k)
        Matrix{Float32}(transpose(rp.map))
    elseif method === :qr
        Pmat = zeros(Float32, k, m)
        offset = 0
        while offset < k
            bs = min(m, k - offset)
            Q, _ = LinearAlgebra.qr(rand(rng, Float32, m, m))
            Qmat = Matrix(Q)
            Pmat[offset+1:offset+bs, :] .= transpose(Qmat[:, 1:bs])
            offset += bs
        end
        Pmat
    elseif method === :sparse_random || method === :ternary
        # Sparse ternary random matrix (Achlioptas / classical RI)
        s = max(2, round(Int, sqrt(k)))
        val = 1f0 / sqrt(Float32(s))
        Pmat = zeros(Float32, k, m)
        for j in 1:m
            idx = randperm(rng, k)[1:s]
            for i in idx
                Pmat[i, j] = rand(rng, Bool) ? val : -val
            end
        end
        Pmat
    else
        throw(ArgumentError("Unknown projection method :$method (allowed: :gaussian, :qr, :sparse_random)"))
    end

    RandomIndexing(model, P, k, k, method)
end

"""
    RandomIndexing(config::TextConfig, corpus;
                   maxoutdim::Integer=1024,
                   method::Symbol=:gaussian,
                   gw::GlobalWeighting=IdfWeighting(),
                   lw::LocalWeighting=TfWeighting(),
                   minfreq::Integer=1,
                   maxfreq::Integer=0,
                   verbose::Bool=true,
                   rng::AbstractRNG=Random.default_rng())

Convenience constructor: builds a [`Vocabulary`](@ref) and [`VectorModel`](@ref) from `corpus`
using `config`, and then creates a [`RandomIndexing`](@ref) model.
"""
function RandomIndexing(
    config::TextConfig,
    corpus;
    maxoutdim::Integer=1024,
    method::Symbol=:gaussian,
    gw::GlobalWeighting=IdfWeighting(),
    lw::LocalWeighting=TfWeighting(),
    verbose::Bool=true,
    rng::AbstractRNG=Random.default_rng()
)
    voc = Vocabulary(config, corpus; verbose)
    model = VectorModel(gw, lw, voc)
    RandomIndexing(model; maxoutdim, method, rng)
end

"""
    RandomIndexing(corpus;
                   config::TextConfig=TextConfig(),
                   maxoutdim::Integer=1024,
                   method::Symbol=:gaussian,
                   gw::GlobalWeighting=IdfWeighting(),
                   lw::LocalWeighting=TfWeighting(),
                   verbose::Bool=true,
                   rng::AbstractRNG=Random.default_rng())

Convenience constructor: creates a [`RandomIndexing`](@ref) model directly from raw `corpus`
using default text configuration.
"""
function RandomIndexing(
    corpus;
    config::TextConfig=TextConfig(),
    maxoutdim::Integer=1024,
    method::Symbol=:gaussian,
    gw::GlobalWeighting=IdfWeighting(),
    lw::LocalWeighting=TfWeighting(),
    verbose::Bool=true,
    rng::AbstractRNG=Random.default_rng()
)
    RandomIndexing(config, corpus; maxoutdim, method, gw, lw, verbose, rng)
end

# In-place dense projection from sparse coordinates
@inline function _project_sparse!(out::AbstractVector{Float32}, P::AbstractMatrix{Float32}, nzind::AbstractVector{<:Integer}, nzval::AbstractVector{Float32})
    fill!(out, 0f0)
    k = size(P, 1)
    @inbounds for (i, term) in enumerate(nzind)
        val = nzval[i]
        @simd for r in 1:k
            out[r] += P[r, term] * val
        end
    end
    out
end

function _normalize_dense!(out::AbstractVector{Float32})
    nrm = norm(out)
    if nrm > 0f0
        inv_nrm = 1f0 / nrm
        @simd for i in eachindex(out)
            out[i] *= inv_nrm
        end
    end
    out
end

# ====================================================================
# Dense Vectorization (Float32)
# ====================================================================

"""
    vectorize!(out::AbstractVector{Float32}, ri::RandomIndexing, vec::SparseVectorLike; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    vectorize!(out::AbstractVector{Float32}, ri::RandomIndexing, text; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)

Projects a document (sparse vector or raw text) into the lower-dimensional dense Random Indexing space in-place into `out`.
"""
function vectorize!(out::AbstractVector{Float32}, ri::RandomIndexing, vec::SparseVectorLike; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    length(out) == outdim(ri) || throw(DimensionMismatch("out vector length $(length(out)) must equal outdim(ri) $(outdim(ri))"))
    _project_sparse!(out, ri.P, vec.nzind, vec.nzval)
    normalize && _normalize_dense!(out)
    out
end

function vectorize!(out::AbstractVector{Float32}, ri::RandomIndexing, text; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    length(out) == outdim(ri) || throw(DimensionMismatch("out vector length $(length(out)) must equal outdim(ri) $(outdim(ri))"))
    buff = take!(VECTORIZE_CACHES)
    try
        svec = vectorize!(buff, ri.model, text; normalize=false, minweight, isnormalized)
        _project_sparse!(out, ri.P, svec.nzind, svec.nzval)
        normalize && _normalize_dense!(out)
    finally
        put!(VECTORIZE_CACHES, buff)
    end
    out
end

"""
    vectorize(ri::RandomIndexing, text_or_sparsevec; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false) -> Vector{Float32}

Projects a raw text or sparse vector into the dense Random Indexing space, returning a `Vector{Float32}` of length `outdim(ri)`.
"""
function vectorize(ri::RandomIndexing, text_or_sparsevec; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    out = Vector{Float32}(undef, outdim(ri))
    vectorize!(out, ri, text_or_sparsevec; normalize, minweight, isnormalized)
    out
end

"""
    vectorize_corpus(ri::RandomIndexing, corpus;
                     normalize::Bool=true,
                     minweight::Real=1e-6,
                     isnormalized::Bool=false,
                     verbose::Bool=true) -> MatrixDatabase{Matrix{Float32}}

Vectorizes every document in `corpus` into the dense Random Indexing space in parallel across threads via `@BATCHES`,
returning a `MatrixDatabase` of size `(outdim(ri), length(corpus))` ready for dense similarity search.
"""
function vectorize_corpus(ri::RandomIndexing, corpus; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false, verbose::Bool=true)
    corpus = collect(corpus)
    n = length(corpus)
    k = outdim(ri)
    O = Matrix{Float32}(undef, k, n)
    minbatch = getminbatch(n)
    prog = Progress(n; dt=1, enabled=verbose, desc="vectorizing corpus with Random Indexing")

    @BATCHES minbatch for i in 1:n
        vectorize!(view(O, :, i), ri, corpus[i]; normalize, minweight, isnormalized)
        next!(prog)
    end

    MatrixDatabase(O)
end

# ====================================================================
# Quantization Pipelines (SQu8, SQgu8)
# ====================================================================

"""
    vectorize(m::Module, ri::RandomIndexing, text_or_sparsevec; kwargs...)

Projects a document into the Random Indexing space and quantizes it using `SQu8` or `SQgu8`.
"""
function vectorize(m::Module, ri::RandomIndexing, text_or_sparsevec; kwargs...)
    if m === SQu8
        normalize = get(kwargs, :normalize, true)
        minweight = get(kwargs, :minweight, 1e-6)
        isnormalized = get(kwargs, :isnormalized, false)
        v = vectorize(ri, text_or_sparsevec; normalize, minweight, isnormalized)
        SQu8.SQu8Vec(v)
    elseif m === SQgu8
        minmax = get(kwargs, :minmax, (-1f0, 1f0))
        normalize = get(kwargs, :normalize, true)
        minweight = get(kwargs, :minweight, 1e-6)
        isnormalized = get(kwargs, :isnormalized, false)
        v = vectorize(ri, text_or_sparsevec; normalize, minweight, isnormalized)
        SQgu8.quantize(v; minmax)
    else
        throw(ArgumentError("Unsupported quantization module $m (expected SQu8 or SQgu8)"))
    end
end

vectorize(::Type{<:SQu8.SQu8Vec}, ri::RandomIndexing, text_or_sparsevec; kwargs...) = vectorize(SQu8, ri, text_or_sparsevec; kwargs...)

"""
    vectorize_corpus(m::Module, ri::RandomIndexing, corpus; kwargs...)

Projects an entire corpus with Random Indexing and quantizes it with `SQu8` or `SQgu8`.
"""
function vectorize_corpus(m::Module, ri::RandomIndexing, corpus; kwargs...)
    if m === SQu8
        normalize = get(kwargs, :normalize, true)
        minweight = get(kwargs, :minweight, 1e-6)
        isnormalized = get(kwargs, :isnormalized, false)
        verbose = get(kwargs, :verbose, true)
        db = vectorize_corpus(ri, corpus; normalize, minweight, isnormalized, verbose)
        SQu8.quantize(db.matrix)
    elseif m === SQgu8
        minmax = get(kwargs, :minmax, nothing)
        quant = get(kwargs, :quant, [0.025, 0.975])
        samplesize = get(kwargs, :samplesize, 0)
        normalize = get(kwargs, :normalize, true)
        minweight = get(kwargs, :minweight, 1e-6)
        isnormalized = get(kwargs, :isnormalized, false)
        verbose = get(kwargs, :verbose, true)
        db = vectorize_corpus(ri, corpus; normalize, minweight, isnormalized, verbose)
        Q = SQgu8.quantize(db.matrix; minmax, quant, samplesize)
        MatrixDatabase(Q)
    else
        throw(ArgumentError("Unsupported quantization module $m (expected SQu8 or SQgu8)"))
    end
end

vectorize_corpus(::Type{<:SQu8.SQu8Database}, ri::RandomIndexing, corpus; kwargs...) = vectorize_corpus(SQu8, ri, corpus; kwargs...)

# ====================================================================
# BitSketches (SimHash / Sign-packing Pipeline)
# ====================================================================

"""
    bitsketch(ri::RandomIndexing, doc; minweight::Real=1e-6, isnormalized::Bool=false) -> Vector{UInt64}
    vectorize(::Union{Type{BitSketch}, BitSketch}, ri::RandomIndexing, doc; minweight::Real=1e-6, isnormalized::Bool=false) -> Vector{UInt64}

Computes a SimHash-style binary bit sketch (packed into `UInt64` words) from a document projected by [`RandomIndexing`](@ref).
"""
function bitsketch(ri::RandomIndexing, doc::Union{AbstractString, SparseVectorLike}; minweight::Real=1e-6, isnormalized::Bool=false)
    v = vectorize(ri, doc; normalize=false, minweight, isnormalized)
    packsigns(v)
end

function bitsketch(ri::RandomIndexing, corpus; minweight::Real=1e-6, isnormalized::Bool=false, verbose::Bool=true)
    db = vectorize_corpus(ri, corpus; normalize=false, minweight, isnormalized, verbose)
    B = packsigns(db.matrix)
    MatrixDatabase(B)
end

const bitsketch_corpus = bitsketch

function vectorize(::Union{Type{BitSketch}, BitSketch}, ri::RandomIndexing, doc; minweight::Real=1e-6, isnormalized::Bool=false)
    bitsketch(ri, doc; minweight, isnormalized)
end

function vectorize_corpus(::Union{Type{BitSketch}, BitSketch}, ri::RandomIndexing, corpus; minweight::Real=1e-6, isnormalized::Bool=false, verbose::Bool=true)
    bitsketch(ri, corpus; minweight, isnormalized, verbose)
end

# ====================================================================
# Model Properties and Display
# ====================================================================

indim(ri::RandomIndexing) = vocsize(ri.model)
outdim(ri::RandomIndexing) = ri.k
vocsize(ri::RandomIndexing) = vocsize(ri.model)
gettrainsize(ri::RandomIndexing) = gettrainsize(ri.model)

function Base.show(io::IO, ri::RandomIndexing)
    println(io, "RandomIndexing: (indim=$(indim(ri)) -> outdim=$(outdim(ri)))")
    println(io, "  method: :$(ri.method)")
    print(io, "  model: ")
    show(io, ri.model)
end

end
