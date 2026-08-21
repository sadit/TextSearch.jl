# This file is part of TextSearch.jl

module LSI

using LinearAlgebra, SparseArrays
using Arpack: svds
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
    LSI_FULL_FACTORIZATION_MAX

Largest Gram-matrix side (`min(vocsize, ndocs)`) for which `factorization=:auto` still uses
the exact dense `:full` path. Above it, `:auto` switches to `:lanczos`: measured on Spanish
Wikipedia slices, `:full` wins below a couple of thousand documents (n=2000: 4.6s vs 14.8s)
and loses badly above (n=8000: 48.8s vs 11.5s), since its cost grows with the cube of this
side while ARPACK's is driven by the number of nonzeros.
"""
const LSI_FULL_FACTORIZATION_MAX = 3072

"""
    _lanczos_svd(A, k) -> Union{Nothing,Tuple}

Truncated SVD of `A` keeping the top `k` singular triplets via ARPACK's implicitly restarted
Lanczos iteration (`Arpack.svds`): exact to working precision (measured ~3e-7 relative error
on the singular values) while never forming a Gram matrix, which is what makes it both the
accurate and the fast choice at scale.

ARPACK's own iteration is sequential and it is not re-entrant (unsynchronized static state,
so it must not be called concurrently from multiple threads -- LSI factorizes one batch at a
time, so that is not a constraint here). It is not, however, serial in throughput: the heavy
work goes to BLAS, so on a multicore host it does use many cores (~17 of 64 measured), just
less effectively than a dense `eigen`, which is BLAS-3 rather than mostly BLAS-1/2.

Returns `nothing` when ARPACK cannot deliver `k` converged triplets -- either by failing to
converge or by throwing -- so the caller can fall back to the exact dense path rather than
abort a long fit.
"""
function _lanczos_svd(A::AbstractMatrix, k::Integer)
    # ARPACK needs strictly fewer singular values than the smaller dimension
    k < minimum(size(A)) || return nothing
    try
        r = svds(A; nsv=k)
        F, nconv = r[1], r[2]
        nconv < k && return nothing
        F.U, F.S
    catch err
        err isa InterruptException && rethrow()
        @warn "LSI: ARPACK/Lanczos factorization failed; falling back to the exact dense path, which may be much slower" exception=err
        nothing
    end
end

"""
    LatentSemanticIndexing(model::VectorModel, corpus;
                           maxoutdim::Integer=128,
                           normalize::Bool=true,
                           minweight::Real=1e-6,
                           isnormalized::Bool=false,
                           verbose::Bool=true,
                           scaling::Symbol=:none,
                           factorization::Symbol=:auto)

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
  - `:auto` (default): `:full` while `min(vocsize, length(corpus))` is at most
    [`LSI_FULL_FACTORIZATION_MAX`](@ref), `:lanczos` above it.
  - `:lanczos`: [`_lanczos_svd`](@ref) -- ARPACK's restarted Lanczos iteration. Exact to
    working precision and the fastest option at scale; falls back to `:full` if ARPACK
    fails to converge.
  - `:full`: exact, via a dense Gram matrix and a complete `eigen`. Costs
    `O(min(m,n)^3)` time and `min(m,n)^2` memory *regardless of `maxoutdim`* (it computes
    every eigenpair and keeps `maxoutdim` of them), so it is only appropriate for small
    corpora.

Both options are exact; the choice is purely about cost, so there is no accuracy knob to
tune here.
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
    resolved = if factorization === :auto
        gram_side <= LSI_FULL_FACTORIZATION_MAX ? :full : :lanczos
    elseif factorization in (:lanczos, :full)
        factorization
    else
        throw(ArgumentError("Unknown factorization: :$factorization (allowed: :auto, :lanczos, :full)"))
    end

    # `:lanczos` is exact but can fail to converge; falling back to the dense path keeps a
    # long fit alive (at a real cost in time) instead of losing it at the factorization step
    lanczos = resolved === :lanczos ? _lanczos_svd(A, k) : nothing
    resolved === :lanczos && lanczos === nothing && (resolved = :full)

    U, s = if resolved === :lanczos
        lanczos
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
)
    voc = Vocabulary(config, corpus; verbose)
    model = VectorModel(gw, lw, voc)
    LatentSemanticIndexing(model, corpus; maxoutdim, normalize, minweight, isnormalized, verbose,
                            scaling, factorization)
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
)
    LatentSemanticIndexing(config, corpus; gw, lw, maxoutdim, normalize, minweight, isnormalized,
                            verbose, scaling, factorization)
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
        -> (; synonyms::Dict{String,Vector{String}}, distances::Dict{String,Vector{Float32}})

Builds a synonym network from `voc`'s token embeddings in `wordvecs` (column `t` =
embedding of `token(voc, t)`, e.g. from [`wordvectors`](@ref) or an externally supplied
matrix): for every vocabulary token, finds its `k` nearest neighbors (by `dist`, cosine by
default) among all *other* tokens' embeddings, via `SimilaritySearch.allknn`. The token
itself is always excluded from its own neighbor list.

The two halves come back **separately**, as parallel per-token lists sorted by increasing
distance (lower means more similar): `synonyms[tok]` are the neighbor tokens in rank order,
and `distances[tok][i]` is the distance to `synonyms[tok][i]`. They are split because only
the ranking participates in the normal query-expansion path -- BM25 ignores the query side's
weights entirely, and the distances stop being distances in any single space as soon as a
network is merged or refitted. Keeping them apart lets a consumer (or a profile on disk)
carry the ranking alone, which is where nearly all of a network's size lives.

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
net.synonyms["dog"]   # ["dogs", "puppy", ...]
net.distances["dog"]  # [0.02, 0.11, ...]
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

    net = Dict{String,Vector{String}}()
    netdist = Dict{String,Vector{Float32}}()
    for t in 1:m
        words = String[]
        wdists = Float32[]
        for j in 1:size(ids, 1)
            nb = ids[j, t]
            d = dists[j, t]
            # a token with a near-uniform document frequency (e.g. "the") can end up with
            # an all-zero embedding after LSI projection, making cosine distance to/from it
            # undefined (0/0 = NaN); such a token has no meaningful direction to compare, so
            # it gets no synonyms and is never anyone else's synonym, rather than poisoning
            # the network (and downstream JSON serialization, which rejects NaN) with NaN.
            (nb == 0 || nb == t || isnan(d)) && continue
            push!(words, token(voc, nb))
            push!(wdists, d)
            length(words) >= k && break
        end
        tok = token(voc, t)
        net[tok] = words
        netdist[tok] = wdists
    end

    (; synonyms=net, distances=netdist)
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
