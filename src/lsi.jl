# This file is part of TextSearch.jl

module LSI

using LinearAlgebra, SparseArrays
using ProgressMeter
using SimilaritySearch
using SimilaritySearch: @BATCHES, getminbatch, MatrixDatabase, AbstractDatabase, ParallelExhaustiveSearch
using SimilaritySearch.Special.Sparse: SparseVecView, SparseVectorLike

using ..TextSearch: TextModel, VectorModel, Vocabulary, TextConfig, token,
                    GlobalWeighting, LocalWeighting, IdfWeighting, TfWeighting,
                    VECTORIZE_CACHES, VectorizeBuffer
import ..TextSearch: vectorize, vectorize!, vectorize_corpus, vocsize, trainsize

export LatentSemanticIndexing, LSIModel, indim, outdim, vocsize, trainsize,
       vectorize, vectorize!, vectorize_corpus, wordvectors, synonyms

"""
    LatentSemanticIndexing{M<:AbstractMatrix{Float32}, VM<:VectorModel} <: TextModel

Latent Semantic Indexing (LSI) model that projects sparse vector representations produced
by a [`VectorModel`](@ref) into a lower-dimensional dense semantic space via Truncated
Singular Value Decomposition (SVD).

# Fields
- `model`: The underlying [`VectorModel`](@ref) used to tokenize and weight text.
- `P`: Dense projection matrix of size `(k, m)` where `k = outdim` and `m = indim = vocsize(model)`.
- `s`: Vector of singular values of length `k`.
- `k`: Output dimension (`k <= maxoutdim`).
- `maxoutdim`: Requested maximum output dimension (default: 128).
- `scaling`: Scaling applied to singular vectors (`:none`, `:inv_singular_values`, `:singular_values`).
"""
struct LatentSemanticIndexing{M<:AbstractMatrix{Float32}, VM<:VectorModel} <: TextModel
    model::VM
    P::M
    s::Vector{Float32}
    k::Int
    maxoutdim::Int
    scaling::Symbol
end

const LSIModel = LatentSemanticIndexing

function _sparse_matrix(X, m::Integer)
    n = length(X)
    colptr = Vector{Int32}(undef, n + 1)
    colptr[1] = 1
    total_nnz = sum(x -> length(x.nzind), X)
    rowval = Vector{Int32}(undef, total_nnz)
    nzval = Vector{Float32}(undef, total_nnz)

    k = 0
    @inbounds for (j, x) in enumerate(X)
        nz = length(x.nzind)
        copyto!(rowval, k + 1, x.nzind, 1, nz)
        copyto!(nzval, k + 1, x.nzval, 1, nz)
        k += nz
        colptr[j + 1] = k + 1
    end

    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

"""
    LatentSemanticIndexing(model::VectorModel, corpus;
                           maxoutdim::Integer=128,
                           normalize::Bool=true,
                           minweight::Real=1e-6,
                           isnormalized::Bool=false,
                           verbose::Bool=true,
                           scaling::Symbol=:none)

Computes a Latent Semantic Indexing (LSI) projection matrix from `corpus` weighted by `model`.
`corpus` can be a collection of raw texts or pre-vectorized sparse vectors (`AbstractVector{<:SparseVectorLike}` or `AbstractDatabase`).

# Keyword Arguments
- `maxoutdim`: Target embedding dimension (default: `128`).
- `normalize`: Whether to L2-normalize vectors during intermediate vectorization (default: `true`).
- `minweight`: Threshold below which sparse vector weights are dropped (default: `1e-6`).
- `isnormalized`: Set to `true` if input texts are already normalized (default: `false`).
- `verbose`: Whether to display progress bar during corpus vectorization (default: `true`).
- `scaling`: Scaling factor applied to projection coordinates:
  - `:none` (default): standard orthogonal concept projection P = U_k^T.
  - `:inv_singular_values`: classical LSI document coordinate scaling P = Σ_k^{-1} U_k^T.
  - `:singular_values`: singular value weighted projection P = Σ_k U_k^T.
"""
function LatentSemanticIndexing(
    model::VectorModel,
    corpus;
    maxoutdim::Integer=128,
    normalize::Bool=true,
    minweight::Real=1e-6,
    isnormalized::Bool=false,
    verbose::Bool=true,
    scaling::Symbol=:none
)
    m = vocsize(model)
    m > 0 || throw(ArgumentError("model vocabulary is empty (vocsize = 0)"))

    A = if corpus isa SparseMatrixCSC
        corpus
    elseif corpus isa AbstractVector{<:SparseVectorLike} || (corpus isa AbstractDatabase && eltype(corpus) <: SparseVectorLike)
        _sparse_matrix(corpus, m)
    else
        X = vectorize_corpus(model, corpus; normalize, minweight, isnormalized, verbose)
        _sparse_matrix(X, m)
    end

    m_mat, n_mat = size(A)
    k = min(Int(maxoutdim), m_mat, n_mat)
    k > 0 || throw(ArgumentError("Output dimension k must be positive; got k=$k for matrix size $(size(A))"))

    # Compute truncated SVD using the smaller Gram matrix
    if m_mat <= n_mat
        C = Matrix(A * transpose(A))
        E = eigen(Symmetric(C))
        idx = sortperm(E.values, rev=true)[1:k]
        s = sqrt.(max.(0f0, E.values[idx]))
        U = E.vectors[:, idx]  # (m x k)
    else
        B = Matrix(transpose(A) * A)
        E = eigen(Symmetric(B))
        idx = sortperm(E.values, rev=true)[1:k]
        s = sqrt.(max.(0f0, E.values[idx]))
        V = E.vectors[:, idx]  # (n x k)
        inv_s = [si > 1e-12 ? 1f0 / si : 0f0 for si in s]
        U = Matrix(A * V) .* reshape(inv_s, 1, :)
    end

    P = Matrix{Float32}(transpose(U))

    if scaling === :inv_singular_values
        inv_s = [si > 1e-12 ? 1f0 / si : 0f0 for si in s]
        P = P .* reshape(inv_s, :, 1)
    elseif scaling === :singular_values
        P = P .* reshape(s, :, 1)
    elseif scaling !== :none
        throw(ArgumentError("Unknown scaling symbol: :$scaling (allowed: :none, :inv_singular_values, :singular_values)"))
    end

    LatentSemanticIndexing(model, P, s, k, Int(maxoutdim), scaling)
end

"""
    LatentSemanticIndexing(config::TextConfig, corpus;
                           gw::GlobalWeighting=IdfWeighting(),
                           lw::LocalWeighting=TfWeighting(),
                           maxoutdim::Integer=128,
                           normalize::Bool=true,
                           minweight::Real=1e-6,
                           isnormalized::Bool=false,
                           verbose::Bool=true,
                           scaling::Symbol=:none)

Convenience constructor that builds a [`Vocabulary`](@ref) and [`VectorModel`](@ref) from `config` and `corpus`,
then fits and returns a [`LatentSemanticIndexing`](@ref) model.
"""
function LatentSemanticIndexing(
    config::TextConfig,
    corpus;
    gw::GlobalWeighting=IdfWeighting(),
    lw::LocalWeighting=TfWeighting(),
    maxoutdim::Integer=128,
    normalize::Bool=true,
    minweight::Real=1e-6,
    isnormalized::Bool=false,
    verbose::Bool=true,
    scaling::Symbol=:none
)
    voc = Vocabulary(config, corpus; verbose)
    model = VectorModel(gw, lw, voc)
    LatentSemanticIndexing(model, corpus; maxoutdim, normalize, minweight, isnormalized, verbose, scaling)
end

"""
    LatentSemanticIndexing(corpus;
                           config::TextConfig=TextConfig(),
                           gw::GlobalWeighting=IdfWeighting(),
                           lw::LocalWeighting=TfWeighting(),
                           maxoutdim::Integer=128,
                           normalize::Bool=true,
                           minweight::Real=1e-6,
                           isnormalized::Bool=false,
                           verbose::Bool=true,
                           scaling::Symbol=:none)

Convenience constructor that builds an LSI model directly from a text `corpus` using default or provided `TextConfig`.
"""
function LatentSemanticIndexing(
    corpus;
    config::TextConfig=TextConfig(),
    gw::GlobalWeighting=IdfWeighting(),
    lw::LocalWeighting=TfWeighting(),
    maxoutdim::Integer=128,
    normalize::Bool=true,
    minweight::Real=1e-6,
    isnormalized::Bool=false,
    verbose::Bool=true,
    scaling::Symbol=:none
)
    LatentSemanticIndexing(config, corpus; gw, lw, maxoutdim, normalize, minweight, isnormalized, verbose, scaling)
end

function _project_sparse!(out::AbstractVector{Float32}, P::AbstractMatrix{Float32}, nzind::AbstractVector{<:Integer}, nzval::AbstractVector{<:Real})
    fill!(out, 0f0)
    k = length(out)
    m = size(P, 2)
    @inbounds for (t, val) in zip(nzind, nzval)
        (t > 0 && t <= m) || continue
        val32 = Float32(val)
        @simd for row in 1:k
            out[row] += P[row, t] * val32
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

"""
    vectorize!(out::AbstractVector{Float32}, lsi::LatentSemanticIndexing, vec::SparseVectorLike; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    vectorize!(out::AbstractVector{Float32}, lsi::LatentSemanticIndexing, text; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)

Projects a document (sparse vector or raw text) into the lower-dimensional dense LSI space in-place into `out`.
"""
function vectorize!(out::AbstractVector{Float32}, lsi::LatentSemanticIndexing, vec::SparseVectorLike; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    length(out) == outdim(lsi) || throw(DimensionMismatch("out vector length $(length(out)) must equal outdim(lsi) $(outdim(lsi))"))
    _project_sparse!(out, lsi.P, vec.nzind, vec.nzval)
    normalize && _normalize_dense!(out)
    out
end

function vectorize!(out::AbstractVector{Float32}, lsi::LatentSemanticIndexing, text; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    length(out) == outdim(lsi) || throw(DimensionMismatch("out vector length $(length(out)) must equal outdim(lsi) $(outdim(lsi))"))
    buff = take!(VECTORIZE_CACHES)
    try
        svec = vectorize!(buff, lsi.model, text; normalize=false, minweight, isnormalized)
        _project_sparse!(out, lsi.P, svec.nzind, svec.nzval)
        normalize && _normalize_dense!(out)
    finally
        put!(VECTORIZE_CACHES, buff)
    end
    out
end

"""
    vectorize(lsi::LatentSemanticIndexing, text_or_sparsevec; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)

Projects a raw text or sparse vector into the dense LSI space, returning a `Vector{Float32}` of length `outdim(lsi)`.
"""
function vectorize(lsi::LatentSemanticIndexing, text_or_sparsevec; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false)
    out = Vector{Float32}(undef, outdim(lsi))
    vectorize!(out, lsi, text_or_sparsevec; normalize, minweight, isnormalized)
    out
end

"""
    vectorize_corpus(lsi::LatentSemanticIndexing, corpus;
                     normalize::Bool=true,
                     minweight::Real=1e-6,
                     isnormalized::Bool=false,
                     verbose::Bool=true) -> MatrixDatabase{Matrix{Float32}}

Vectorizes every document in `corpus` into the dense LSI space in parallel across threads via `@BATCHES`,
returning a `MatrixDatabase` of size `(outdim(lsi), length(corpus))` ready for dense similarity search.
"""
function vectorize_corpus(lsi::LatentSemanticIndexing, corpus; normalize::Bool=true, minweight::Real=1e-6, isnormalized::Bool=false, verbose::Bool=true)
    corpus = collect(corpus)
    n = length(corpus)
    k = outdim(lsi)
    O = Matrix{Float32}(undef, k, n)
    minbatch = getminbatch(n)
    prog = Progress(n; dt=1, enabled=verbose, desc="vectorizing corpus with LSI")

    @BATCHES minbatch for i in 1:n
        vectorize!(view(O, :, i), lsi, corpus[i]; normalize, minweight, isnormalized)
        next!(prog)
    end

    MatrixDatabase(O)
end

"""
    wordvectors(lsi::LatentSemanticIndexing; normalize::Bool=true) -> MatrixDatabase{Matrix{Float32}}

Returns the LSI embedding of every vocabulary token, as a `(outdim(lsi), vocsize(lsi))` matrix
database -- column `t` is the embedding of `token(lsi.model, t)`. This is exactly `lsi.P`
(optionally column-normalized): a document's LSI vector (via [`vectorize`](@ref)/
[`vectorize_corpus`](@ref)) is a weighted sum of its tokens' columns of `lsi.P`, so these
per-token vectors live in the same projected space and are directly comparable to each other and
to document vectors (e.g. via `Dist.Cosine()`/`Dist.NormCosine()`). Set `normalize=false` to keep
the raw (`scaling`-adjusted) `lsi.P` columns instead of unit-normalizing them.

# Example
```julia
X = wordvectors(lsi)   # (outdim(lsi), vocsize(lsi)) MatrixDatabase
X[5]                   # the embedding of token(lsi.model, 5)
```
"""
function wordvectors(lsi::LatentSemanticIndexing; normalize::Bool=true)
    m = vocsize(lsi)
    O = Matrix{Float32}(undef, outdim(lsi), m)
    copyto!(O, lsi.P)
    if normalize
        minbatch = getminbatch(m)
        @BATCHES minbatch for t in 1:m
            _normalize_dense!(view(O, :, t))
        end
    end
    MatrixDatabase(O)
end

"""
    synonyms(voc::Vocabulary, wordvecs::AbstractDatabase, k::Integer=8;
             dist=Dist.Cosine(), verbose::Bool=true) -> Dict{String,Vector{Pair{String,Float32}}}

Builds an approximate synonym network from `voc`'s token embeddings in `wordvecs` (column
`t` = embedding of `token(voc, t)`, e.g. from [`wordvectors`](@ref) or an externally
supplied matrix): for every vocabulary token, finds its `k` nearest neighbors (by `dist`,
cosine by default, computed exactly via `ParallelExhaustiveSearch`) among all *other*
tokens' embeddings, using `SimilaritySearch.allknn`. Returns a `Dict` mapping each token to
a list of `neighbor_token => distance` pairs sorted by increasing distance (lower means
more similar); the token itself is always excluded from its own neighbor list.

For very large vocabularies, exhaustive all-pairs search may be too slow -- build your own
approximate index instead (e.g. `SearchGraph(dist, wordvecs)`, `index!`'d) and call
`SimilaritySearch.allknn` on it directly.

# Example
```julia
net = synonyms(voc, wordvectors(lsi), 5)
net["dog"]   # ["dogs" => 0.02, "puppy" => 0.11, ...]
```
"""
function synonyms(voc::Vocabulary, wordvecs::AbstractDatabase, k::Integer=8;
                   dist=Dist.Cosine(), verbose::Bool=true)
    k > 0 || throw(ArgumentError("k must be positive"))
    m = length(wordvecs)
    idx = ParallelExhaustiveSearch(dist, wordvecs)
    ctx = GenericContext()
    ids, dists = allknn(idx, ctx, min(k + 1, m); progress=Progress(m; dt=1, enabled=verbose, desc="synonyms allknn"))

    net = Dict{String,Vector{Pair{String,Float32}}}()
    for t in 1:m
        pairs = Pair{String,Float32}[]
        for j in 1:size(ids, 1)
            nb = ids[j, t]
            d = dists[j, t]
            # a token with a near-uniform document frequency (e.g. "the") can end up with
            # an all-zero embedding after LSI projection, making cosine distance to/from it
            # undefined (0/0 = NaN); such a token has no meaningful direction to compare, so
            # it gets no synonyms and is never anyone else's synonym, rather than poisoning
            # the network (and downstream JSON serialization, which rejects NaN) with NaN.
            (nb == 0 || nb == t || isnan(d)) && continue
            push!(pairs, token(voc, nb) => d)
            length(pairs) >= k && break
        end
        net[token(voc, t)] = pairs
    end

    net
end

"""
    synonyms(lsi::LatentSemanticIndexing, k::Integer=8;
             dist=Dist.Cosine(), normalize::Bool=true, verbose::Bool=true) -> Dict{String,Vector{Pair{String,Float32}}}

Builds an approximate synonym network from `lsi`'s vocabulary embeddings ([`wordvectors`](@ref));
see the `(voc, wordvecs, k)` method above for the underlying algorithm. `normalize` is
forwarded to [`wordvectors`](@ref) before searching.

# Example
```julia
net = synonyms(lsi, 5)
net["dog"]   # ["dogs" => 0.02, "puppy" => 0.11, ...]
```
"""
function synonyms(lsi::LatentSemanticIndexing, k::Integer=8;
                   dist=Dist.Cosine(), normalize::Bool=true, verbose::Bool=true)
    synonyms(lsi.model.voc, wordvectors(lsi; normalize), k; dist, verbose)
end

indim(lsi::LatentSemanticIndexing) = vocsize(lsi.model)
outdim(lsi::LatentSemanticIndexing) = lsi.k
vocsize(lsi::LatentSemanticIndexing) = vocsize(lsi.model)
trainsize(lsi::LatentSemanticIndexing) = trainsize(lsi.model)

function Base.show(io::IO, lsi::LatentSemanticIndexing)
    println(io, "LatentSemanticIndexing: (indim=$(indim(lsi)) -> outdim=$(outdim(lsi)))")
    println(io, "  scaling: :$(lsi.scaling)")
    println(io, "  top singular values: $(first(lsi.s, min(5, length(lsi.s))))")
    print(io, "  model: ")
    show(io, lsi.model)
end

end
