# This file is part of TextSearch.jl

module LSI

using LinearAlgebra, SparseArrays
using Arpack: svds
using ProgressMeter
using SimilaritySearch
using SimilaritySearch: @BATCHES, getminbatch, MatrixDatabase, AbstractDatabase, ParallelExhaustiveSearch
using SimilaritySearch.Special.Sparse: SparseVecView, SparseVectorLike

using ..TextSearch: TextModel, VectorModel, Vocabulary, TextConfig, gettoken,
                    GlobalWeighting, LocalWeighting, IdfWeighting, TfWeighting,
                    VECTORIZE_CACHES, VectorizeBuffer
import ..TextSearch: vectorize, vectorize!, vectorize_corpus, vocsize, gettrainsize

export LatentSemanticIndexing, LSIModel, indim, outdim, vocsize, gettrainsize,
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
database -- column `t` is the embedding of `gettoken(lsi.model, t)`. This is exactly `lsi.P`
(optionally column-normalized): a document's LSI vector (via [`vectorize`](@ref)/
[`vectorize_corpus`](@ref)) is a weighted sum of its tokens' columns of `lsi.P`, so these
per-token vectors live in the same projected space and are directly comparable to each other and
to document vectors (e.g. via `Dist.Cosine()`/`Dist.NormCosine()`). Set `normalize=false` to keep
the raw (`scaling`-adjusted) `lsi.P` columns instead of unit-normalizing them.

# Example
```julia
X = wordvectors(lsi)   # (outdim(lsi), vocsize(lsi)) MatrixDatabase
X[5]                   # the embedding of gettoken(lsi.model, 5)
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

# Four ways of deciding "where does this token's real neighborhood end?" were measured on
# 272,466 Spanish Wikipedia paragraphs (106,436 tokens, 256-dim LSI) and all four fail, for the
# same underlying reason. Recorded here because each one looks obviously right on paper.
#
# The failing population is a token whose neighbors are incoherent -- `comun` returning
# `aguilucho pintojo arrendajo cerceta` (bird names, from "nombre comun" in species articles),
# `bojan` returning `cahill musica carnatica`. The healthy comparison cases are `innodb` ->
# `mysql oracle sqlite mariadb interbase` and `planeta` -> `marte saturno neptuno urano jupiter`.
#
# 1. ABSOLUTE DISTANCE CEILING. No global scale exists: `innodb`, whose list is perfect, has the
#    farthest first neighbour of every token examined (0.501) -- farther than `boca` (0.387),
#    `comun` (0.426) and `seccion` (0.465), whose lists are junk. Any ceiling separating them
#    cuts the good one first.
# 2. KNEE / SECOND DIFFERENCE of the sorted distance curve. There is no knee: `planeta` runs
#    0.037 -> 0.179 over 24 neighbours in smooth increments, and its largest curvature falls at
#    position 9, in the middle of the correct list (it would drop `joviana ceres deimos
#    galileanos caronte`). In `innodb` the real good/junk boundary sits between `interbase`
#    (0.572) and `sidereo` (0.579), a gap of 0.007 -- indistinguishable from the 0.005 gaps
#    inside the good part.
# 3. LOCAL RADIUS from reverse votes (`bichromatic_metricjoin`, the `prune=:localradius` option
#    below). Correctly empties the polysemous fillers, but also empties `innodb`, `cistoscopia`
#    (-> `vesical cistitis biopsia`) and `aluminosilicato`, because the estimate needs the token
#    to be in someone else's top-k: a rare term close to `mysql` gets no voters, since `mysql`
#    has closer neighbours, so it falls to the global cutoff and is cut. It hits precisely the
#    rare-but-coherent technical vocabulary a synonym network is most useful for.
# 4. RECIPROCITY, and DOCUMENT FREQUENCY. Reciprocity measures frequency, not coherence:
#    `forma` and `tiene` (junk lists) score 1.00 while `innodb` and `destacamento` (good lists)
#    score 0.00 -- high-frequency tokens sit near the centre of the space and link mutually,
#    rare ones point at tokens with closer neighbours. Document frequency does not separate at
#    the bottom either: at 5-8 documents both `innodb`/`ruderal` (good) and
#    `bojan`/`declararan` (bad) live together, so a sample-size floor cuts both.
#    The raw LSI norm over sqrt(ndocs) does separate by ~2x in the mid and high bands
#    (`guardia` 0.00215 vs `boca` 0.00099) and not at all in the rare band (good 0.00040-0.00085
#    against bad 0.00049-0.00056) -- i.e. it works where it is not needed.
#
# 5. TYPING THE NETWORK by syntactic class -- "nouns with nouns, verbs with verbs" -- so that
#    `comun` (an adjective) can reach `popular` but not `aguilucho`, without disambiguating
#    senses at all. The class can be induced cheaply, with no tagger and no context model: the
#    function word PRECEDING a token is diagnostic, since in Spanish an adjective follows
#    `mas`/`muy`, a noun follows a determiner, and a verb follows `se`/`que`/`lo`/`le`. One
#    bigram pass restricted to (function word, token) pairs gives a small profile per token.
#
#    It works on the target case: cosine between profiles is 0.86 for `comun`/`popular` and 0.03
#    for `comun`/`aguilucho`, 0.01 for `comun`/`cigueniuela`. It does NOT implement "nouns with
#    nouns", because the profile separates common from proper nouns -- `planeta` follows a
#    determiner (0.96) and `marte` follows a preposition (0.91), as Spanish proper nouns take no
#    article -- so `planeta`/`marte` scores 0.058, *below* the 0.14 that must be cut to remove
#    `comun`/`aguilucho`. A hard same-class filter therefore deletes one of the best synonyms the
#    paragraph split produces.
#
#    Coarsening the anchors into groups (determiners, graders, copulas, clitics, prepositions)
#    trades one failure for another rather than fixing it: `tiene`/`hizo` improves from 0.53 to
#    0.92, and `nuevo`/`ciudad` breaks from 0.01 to 0.97, because the fine profile distinguished
#    `nuevo` (after `un`) from `ciudad` (after `la`) and the group merges both into DET.
#    `planeta`/`marte` stays at 0.058 either way. The preceding-word distribution encodes a
#    mixture of category, definiteness, proper-versus-common and construction type, and no fixed
#    grouping isolates the category, because the signal is not separated in the data.
#
#    Coverage bounds it further: only 40,132 of the 106,436 vocabulary tokens have 10 or more
#    anchor observations (38%), and the 62% without them are the rare tail where an incoherent
#    neighborhood does the most damage, since high idf amplifies it.
#
#    What the data does suggest, unmeasured beyond two pairs: filter only when the profiles are
#    incompatible AND the neighbour is far rarer. `comun`(1857 anchor observations) ->
#    `aguilucho`(9) is a ratio of 206x while `planeta`(1610) -> `marte`(578) is 2.8x, a much
#    wider margin than 0.14 against 0.058. That is two interacting heuristics and would need
#    measuring over many pairs before being believed.
#
# The common cause of 1-4 is that in 256 dimensions under cosine the distances concentrate: the RANKING
# carries information, the absolute values and their differences do not. What actually separates
# `innodb` from `bojan` is that `innodb`'s five documents are all about databases while
# `bojan`'s eight are about unrelated people -- context coherence, which no statistic already on
# hand encodes.
#
# So the noise is accepted rather than filtered, and the query-expansion weighting is what bounds
# it: `expand_synonyms!` appends `weight * weight_fn(rank)`, i.e. the weight of the token that
# produced the expansion. A high-frequency polysemous token has low idf and therefore contributes
# its noise weakly. The remaining exposure is a rare token with an incoherent neighbourhood, whose
# high idf amplifies it -- bounded in practice because such tokens are the bulk of a vocabulary
# by count but not of real queries.

"""
    _synonyms_localradius(voc, wordvecs, idx, ictx, kk, kcap, rank, q, mingroup)

Assembles a synonym network whose per-token neighbor count is decided by the data instead of by
a fixed `k`, via [`SimilaritySearch.bichromatic_metricjoin`](@ref) as a self-join.

A top-`k` network gives every token exactly `k` neighbors whether or not it has `k` real ones,
so a token in a sparse region of the embedding gets filler and a token in a dense one gets
truncated. The join instead estimates a cutoff radius *per token* from the reverse view of the
same search: every token that ranked `t` among its own closest `rank` candidates votes for `t`
with that distance, and `t`'s cutoff is the `q`-quantile of its voters. Tokens with fewer than
`mingroup` voters fall back to a pooled global cutoff.

Two consequences worth knowing before choosing this over `:topk`. The surviving pairs are those
where the *other* token found `t` in its own top-`kk`, so the network becomes mutual-ish rather
than a plain per-token top-`k`: a token that nobody's neighborhood reaches gets no synonyms even
if it has close ones of its own. And the output size is data-dependent, so `kcap` is applied only
as a ceiling to keep a pathologically dense token from carrying thousands of neighbors.
"""
function _synonyms_localradius(voc::Vocabulary, wordvecs::AbstractDatabase, idx, ictx,
                               kk::Integer, kcap::Integer, rank::Int, q::Float64, mingroup::Int)
    pairs = bichromatic_metricjoin(idx, ictx, wordvecs; k=kk, rank=rank, q=q,
                                   mingroup=mingroup, samedata=true)
    acc = Dict{Int,Vector{Tuple{Float32,Int}}}()
    for (a, b, d) in pairs
        isnan(d) && continue          # zero embeddings: no direction to compare, see `synonyms`
        push!(get!(() -> Tuple{Float32,Int}[], acc, Int(a)), (d, Int(b)))
    end

    net = Dict{String,Vector{String}}()
    netdist = Dict{String,Vector{Float32}}()
    for t in 1:length(wordvecs)
        tok = gettoken(voc, t)
        got = get(acc, t, nothing)
        if got === nothing
            net[tok] = String[]
            netdist[tok] = Float32[]
            continue
        end
        sort!(got)
        length(got) > kcap && resize!(got, kcap)
        net[tok] = [gettoken(voc, b) for (_, b) in got]
        netdist[tok] = [d for (d, _) in got]
    end

    (; synonyms=net, distances=netdist)
end

"""
    synonyms(voc::Vocabulary, wordvecs::AbstractDatabase, k::Integer=8;
             dist=Dist.Cosine(), verbose::Bool=true, approx=:auto,
             construction_recall::Real=0.97, search_recall::Real=0.9)
        -> (; synonyms::Dict{String,Vector{String}}, distances::Dict{String,Vector{Float32}})

Builds a synonym network from `voc`'s token embeddings in `wordvecs` (column `t` =
embedding of `gettoken(voc, t)`, e.g. from [`wordvectors`](@ref) or an externally supplied
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
                   construction_recall::Real=0.97, search_recall::Real=0.9,
                   prune::Symbol=:topk, join_rank::Integer=1, join_quantile::Real=0.9,
                   join_mingroup::Integer=8)
    k > 0 || throw(ArgumentError("k must be positive"))
    prune in (:topk, :localradius) ||
        throw(ArgumentError("prune must be :topk or :localradius; got $(repr(prune))"))
    m = length(wordvecs)
    kk = min(k + 1, m)

    useapprox = approx === :auto ? m > SYNONYMS_APPROX_THRESHOLD :
                approx isa Bool ? approx :
                throw(ArgumentError("approx must be :auto, true, or false; got $(repr(approx))"))

    idx, ictx = if useapprox
        G = SearchGraph(dist, wordvecs)
        gctx = SearchGraphContext(;
            hyperparameters_callback=OptimizeParameters(MinRecall(construction_recall)),
            verbose)
        index!(G, gctx)
        # tune for the same k `allknn` will ask for; optimizing at the default ksearch=10
        # and then querying at a different k leaves realized recall off target
        optimize_index!(G, gctx, MinRecall(search_recall); ksearch=kk)
        G, gctx
    else
        ParallelExhaustiveSearch(dist, wordvecs), GenericContext()
    end

    if prune === :localradius
        return _synonyms_localradius(voc, wordvecs, idx, ictx, kk, k,
                                     Int(join_rank), Float64(join_quantile), Int(join_mingroup))
    end

    ids, dists = allknn(idx, ictx, kk;
                        progress=Progress(m; dt=1, enabled=verbose,
                                          desc="synonyms allknn ($(useapprox ? "approx" : "exact"))"))

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
            push!(words, gettoken(voc, nb))
            push!(wdists, d)
            length(words) >= k && break
        end
        tok = gettoken(voc, t)
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
gettrainsize(lsi::LatentSemanticIndexing) = gettrainsize(lsi.model)

function Base.show(io::IO, lsi::LatentSemanticIndexing)
    println(io, "LatentSemanticIndexing: (indim=$(indim(lsi)) -> outdim=$(outdim(lsi)))")
    println(io, "  scaling: :$(lsi.scaling)")
    println(io, "  top singular values: $(first(lsi.s, min(5, length(lsi.s))))")
    print(io, "  model: ")
    show(io, lsi.model)
end

end
