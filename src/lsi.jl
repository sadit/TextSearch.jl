# This file is part of TextSearch.jl

module LSI

using LinearAlgebra, SparseArrays, Random
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

"The Gram-matrix side past which `LatentSemanticIndexing`'s `factorization=:auto` uses randomized SVD."
const LSI_RANDOMIZED_THRESHOLD = 4096

"""
    _randomized_svd(A, k; oversampling=10, power_iterations=2, rng=Random.default_rng()) -> (U, s)

Truncated SVD of `A` (`m×n`, typically sparse) keeping the top `k` left singular vectors and
singular values, by the randomized range-finder method (Halko, Martinsson & Tropp).

This exists because the exact alternative does not scale: forming a Gram matrix and calling
`eigen` computes *all* `min(m,n)` eigenpairs, at `O(min(m,n)^3)` time and a dense
`min(m,n)^2` allocation, only to keep `k` of them -- for 25k documents that is a 2.5 GB
matrix and ~1.6e13 flops to extract 128 directions. Here the cost is instead a couple of
sparse products plus a QR and an SVD of a `(k+oversampling)`-row matrix, and no Gram matrix
is ever formed.

`power_iterations` sharpens accuracy when the spectrum decays slowly (term-document matrices
do); the basis is re-orthonormalized between iterations, which matters in `Float32` where
the powers would otherwise collapse toward the leading direction. `oversampling = 0` (the
default) widens the sketch to `2k`.

Accuracy is a real knob here, not a formality: these are measured on 4,000 Spanish Wikipedia
articles (66,538 tokens, `k=128`), comparing against the exact factorization -- "synonym
recall" is how much of the exact top-1/top-8 synonym network the resulting embeddings
reproduce.

| `oversampling` | `power_iterations` | singular-value error | synonym recall@1 / @8 |
|---:|---:|---:|---:|
| 10 | 2 | 4.2e-2 | 0.73 / 0.49 |
| 10 | 4 | 1.4e-2 | 0.90 / 0.67 |
| 64 | 4 | 5.7e-3 | 0.96 / 0.76 |
| 0 (=k) | 4 | 2.3e-3 | **0.99 / 0.85** *(default)* |
| 0 (=k) | 8 | 1.2e-4 | 1.00 / 0.93 |
| 2k | 8 | 1.0e-5 | 1.00 / 0.99 |

The textbook `p=10, q=2` is visibly not enough for this kind of matrix: it recovers the
singular *values* to a few percent while still getting a materially different subspace, and
the synonym network is built from the subspace.
"""
function _randomized_svd(A::AbstractMatrix, k::Integer;
                          oversampling::Integer=0, power_iterations::Integer=4,
                          rng=Random.default_rng())
    m, n = size(A)
    # oversampling = 0 means "as wide again as k": a term-document spectrum decays too
    # slowly for the textbook p=10, which loses the subspace (see the docstring's table)
    p = oversampling > 0 ? Int(oversampling) : k
    l = min(k + p, n)

    # thin Q of a QR: `Matrix(F.Q)` is not portably thin, so project the implicit Q
    thinq(Y) = (F = qr(Y); F.Q * Matrix{Float32}(I, size(Y, 1), size(Y, 2)))

    Y = A * randn(rng, Float32, n, l)          # m×l sample of A's range
    for _ in 1:power_iterations
        Y = thinq(Y)
        Y = A * (transpose(A) * Y)
    end

    Q = thinq(Y)                                # m×l orthonormal basis
    B = Matrix(transpose(Q) * A)                # l×n -- small, this is what gets factorized
    F = svd(B)
    kk = min(k, length(F.S))
    Q * F.U[:, 1:kk], F.S[1:kk]
end

"""
    LatentSemanticIndexing(model::VectorModel, corpus;
                           maxoutdim::Integer=128,
                           normalize::Bool=true,
                           minweight::Real=1e-6,
                           isnormalized::Bool=false,
                           verbose::Bool=true,
                           scaling::Symbol=:none,
                           factorization::Symbol=:auto,
                           oversampling::Integer=0,
                           power_iterations::Integer=4)

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
- `factorization`: how the truncated SVD is computed, which decides whether a large corpus
  is tractable at all:
  - `:auto` (default): `:randomized` once `min(vocsize, length(corpus))` exceeds
    [`LSI_RANDOMIZED_THRESHOLD`](@ref), `:full` below it.
  - `:randomized`: [`_randomized_svd`](@ref) -- a couple of sparse products plus a small
    dense SVD, never forming a Gram matrix.
  - `:full`: exact, via a dense Gram matrix and a complete `eigen`. Costs
    `O(min(m,n)^3)` time and `min(m,n)^2` memory *regardless of `maxoutdim`*, so it is only
    appropriate for small corpora.
- `oversampling`, `power_iterations`: accuracy knobs for `:randomized` (see
  [`_randomized_svd`](@ref)); ignored by `:full`.
"""
function LatentSemanticIndexing(
    model::VectorModel,
    corpus;
    maxoutdim::Integer=128,
    normalize::Bool=true,
    minweight::Real=1e-6,
    isnormalized::Bool=false,
    verbose::Bool=true,
    scaling::Symbol=:none,
    factorization::Symbol=:auto,
    oversampling::Integer=0,
    power_iterations::Integer=4
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

    gram_side = min(m_mat, n_mat)
    use_randomized = if factorization === :auto
        gram_side > LSI_RANDOMIZED_THRESHOLD
    elseif factorization === :randomized
        true
    elseif factorization === :full
        false
    else
        throw(ArgumentError("Unknown factorization: :$factorization (allowed: :auto, :randomized, :full)"))
    end

    U, s = if use_randomized
        _randomized_svd(A, k; oversampling, power_iterations)
    elseif m_mat <= n_mat
        # exact, via the smaller Gram matrix: dense and O(gram_side^3), see the note on
        # `factorization` in the docstring
        C = Matrix(A * transpose(A))
        E = eigen(Symmetric(C))
        idx = sortperm(E.values, rev=true)[1:k]
        E.vectors[:, idx], sqrt.(max.(0f0, E.values[idx]))  # (m x k)
    else
        B = Matrix(transpose(A) * A)
        E = eigen(Symmetric(B))
        idx = sortperm(E.values, rev=true)[1:k]
        sv = sqrt.(max.(0f0, E.values[idx]))
        V = E.vectors[:, idx]  # (n x k)
        inv_s = [si > 1e-12 ? 1f0 / si : 0f0 for si in sv]
        Matrix(A * V) .* reshape(inv_s, 1, :), sv
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
    scaling::Symbol=:none,
    factorization::Symbol=:auto,
    oversampling::Integer=0,
    power_iterations::Integer=4
)
    voc = Vocabulary(config, corpus; verbose)
    model = VectorModel(gw, lw, voc)
    LatentSemanticIndexing(model, corpus; maxoutdim, normalize, minweight, isnormalized, verbose,
                            scaling, factorization, oversampling, power_iterations)
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
    scaling::Symbol=:none,
    factorization::Symbol=:auto,
    oversampling::Integer=0,
    power_iterations::Integer=4
)
    LatentSemanticIndexing(config, corpus; gw, lw, maxoutdim, normalize, minweight, isnormalized,
                            verbose, scaling, factorization, oversampling, power_iterations)
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

"The vocabulary size past which [`synonyms`](@ref)' `approx=:auto` prefers an approximate index."
const SYNONYMS_APPROX_THRESHOLD = 4096

"""
    synonyms(voc::Vocabulary, wordvecs::AbstractDatabase, k::Integer=8;
             dist=Dist.Cosine(), verbose::Bool=true, approx=:auto,
             construction_recall::Real=0.97, search_recall::Real=0.9)
        -> Dict{String,Vector{Pair{String,Float32}}}

Builds a synonym network from `voc`'s token embeddings in `wordvecs` (column `t` =
embedding of `token(voc, t)`, e.g. from [`wordvectors`](@ref) or an externally supplied
matrix): for every vocabulary token, finds its `k` nearest neighbors (by `dist`, cosine by
default) among all *other* tokens' embeddings, via `SimilaritySearch.allknn`. Returns a
`Dict` mapping each token to a list of `neighbor_token => distance` pairs sorted by
increasing distance (lower means more similar); the token itself is always excluded from
its own neighbor list.

`approx` selects how the all-pairs search is done, and matters enormously on real
vocabularies -- an exhaustive search is O(vocabulary²):

- `:auto` (default): approximate when `length(wordvecs) > SYNONYMS_APPROX_THRESHOLD`,
  exhaustive below it (where exhaustive is already fast *and* exact, so there is nothing
  to gain from approximating).
- `true`: always approximate -- build a `SearchGraph`, autotuning construction to
  `MinRecall(construction_recall)` and then the search parameters to
  `MinRecall(search_recall)`.
- `false`: always exhaustive, via `ParallelExhaustiveSearch`. Exact, and unusably slow past
  a few tens of thousands of tokens.

# Example
```julia
net = synonyms(voc, wordvectors(lsi), 5)
net["dog"]   # ["dogs" => 0.02, "puppy" => 0.11, ...]
```
"""
function synonyms(voc::Vocabulary, wordvecs::AbstractDatabase, k::Integer=8;
                   dist=Dist.Cosine(), verbose::Bool=true, approx=:auto,
                   construction_recall::Real=0.97, search_recall::Real=0.9)
    k > 0 || throw(ArgumentError("k must be positive"))
    m = length(wordvecs)
    kk = min(k + 1, m)

    useapprox = approx === :auto ? m > SYNONYMS_APPROX_THRESHOLD :
                approx isa Bool ? approx :
                throw(ArgumentError("approx must be :auto, true, or false; got $(repr(approx))"))

    ids, dists = if useapprox
        G = SearchGraph(dist, wordvecs)
        gctx = SearchGraphContext(;
            hyperparameters_callback=OptimizeParameters(MinRecall(construction_recall)),
            verbose)
        index!(G, gctx)
        # tune for the same k `allknn` will ask for; optimizing at the default ksearch=10
        # and then querying at a different k leaves realized recall off target
        optimize_index!(G, gctx, MinRecall(search_recall); ksearch=kk)
        allknn(G, gctx, kk; progress=Progress(m; dt=1, enabled=verbose, desc="synonyms allknn (approx)"))
    else
        idx = ParallelExhaustiveSearch(dist, wordvecs)
        allknn(idx, GenericContext(), kk; progress=Progress(m; dt=1, enabled=verbose, desc="synonyms allknn (exact)"))
    end

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
             dist=Dist.Cosine(), normalize::Bool=true, verbose::Bool=true, approx=:auto,
             construction_recall::Real=0.97, search_recall::Real=0.9) -> Dict{String,Vector{Pair{String,Float32}}}

Builds a synonym network from `lsi`'s vocabulary embeddings ([`wordvectors`](@ref)); see
the `(voc, wordvecs, k)` method above for the underlying algorithm and for what `approx`/
`construction_recall`/`search_recall` control. `normalize` is forwarded to
[`wordvectors`](@ref) before searching.

# Example
```julia
net = synonyms(lsi, 5)
net["dog"]   # ["dogs" => 0.02, "puppy" => 0.11, ...]
```
"""
function synonyms(lsi::LatentSemanticIndexing, k::Integer=8;
                   dist=Dist.Cosine(), normalize::Bool=true, verbose::Bool=true, approx=:auto,
                   construction_recall::Real=0.97, search_recall::Real=0.9)
    synonyms(lsi.model.voc, wordvectors(lsi; normalize), k;
             dist, verbose, approx, construction_recall, search_recall)
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
