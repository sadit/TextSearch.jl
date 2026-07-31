# This file is a part of TextSearch.jl

#####
export EntropyWeighting, NormalizedEntropy, SigmoidPenalizeFewSamples, CombineWeighting

"""
    CombineWeighting

Abstract type for the strategies that turn a per-token, per-class distribution and its
empirical entropy into a single global weight (via the internal `combine_weight`
function), used by [`EntropyWeighting`](@ref). Available strategies:
[`NormalizedEntropy`](@ref) and [`SigmoidPenalizeFewSamples`](@ref).
"""
abstract type CombineWeighting end

"""
    NormalizedEntropy()

A [`CombineWeighting`](@ref) that weights a token as `1 - entropy / maxent`: tokens that
discriminate well between classes (low entropy) get a weight close to `1`, tokens spread
uniformly across classes (entropy close to `maxent`) get a weight close to `0`.
"""
struct NormalizedEntropy <: CombineWeighting end
combine_weight(::NormalizedEntropy, model, tokenID, entropy, maxent)::Float32 = 1.0 - entropy / maxent
# the entropy scores the discrimination power of the term while log(m) weights
# the term w.r.t the available evidency. The current form tries to equalize the
# scales
struct PenalizeFewSamples <: CombineWeighting end
combine_weight(::PenalizeFewSamples, model, tokenID, entropy, maxent)::Float32 = (maxent - entropy) * log2(ndocs(model, tokenID))

"""
    SigmoidPenalizeFewSamples()

Like [`NormalizedEntropy`](@ref), but additionally down-weights tokens seen in very few
documents (low `ndocs`) via a sigmoid-like penalty on `log2(ndocs)`, so that rare tokens
don't get an unduly high weight just because they appear in a single class.
"""
struct SigmoidPenalizeFewSamples <: CombineWeighting end
combine_weight(::SigmoidPenalizeFewSamples, model, tokenID, entropy, maxent)::Float32 = (1 - entropy/maxent) * (1-1/(1+log2(ndocs(model, tokenID))))

"""
    EntropyWeighting()

A [`GlobalWeighting`](@ref) that scores each token by the empirical entropy of its
occurrences across document classes/labels, instead of the plain document frequency used
by [`IdfWeighting`](@ref) — tokens whose occurrences concentrate on few classes get a
higher weight than tokens spread uniformly across all classes. Since it needs document
labels, it is not built via the generic [`VectorModel(gw, lw, voc)`](@ref VectorModel)
constructor; use [`VectorModel(::EntropyWeighting, lw, voc, corpus, labels)`](@ref
VectorModel) instead.
"""
struct EntropyWeighting <: GlobalWeighting end

function entropy_(dist)::Float32
    e = 0f0
    ipop = 1f0/sum(dist)

    for x in dist
        p = x * ipop

        if p > 0.0
            e -= p * log2(p)
        end
    end

    e
end


"""
    VectorModel(ent::EntropyWeighting, lw::LocalWeighting, voc::Vocabulary, corpus::AbstractVector, labels::AbstractVector;
        mindocs=3,
        smooth=3,
        weights=:balance,
        comb::CombineWeighting=NormalizedEntropy(),
        minbatch=0, verbose=true
    )

Creates a [`VectorModel`](@ref) with [`EntropyWeighting`](@ref) as its global weighting
scheme. Unlike the generic [`VectorModel(gw, lw, voc)`](@ref VectorModel) constructor,
this one needs the actual `corpus` and a matching `labels` vector (one label per
document) to compute, for each token, its occurrence distribution across classes and the
resulting entropy-based weight.

- `mindocs`: tokens occurring in fewer than `mindocs` documents get weight `0`.
- `smooth`: additive (Laplace-like) smoothing applied to the per-class occurrence counts
  before computing entropy, to avoid zero counts.
- `weights`: how to reweight classes before computing entropy — `:balance` compensates
  for class-size imbalance, `:none` (or `nothing`) leaves classes unweighted, or pass an
  `AbstractVector` of per-class weights directly.
- `comb`: the [`CombineWeighting`](@ref) strategy combining entropy and evidence into the
  final per-token weight.
"""
function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, voc::Vocabulary, corpus::AbstractVector, labels::AbstractVector;
            mindocs=3,
            smooth=3,
            weights=:balance,
            comb::CombineWeighting=NormalizedEntropy(),
            minbatch=0, verbose=true
        )
    @assert length(labels) == length(corpus)
    n = length(labels)
    L = Dict(l => i for (i, l) in enumerate(sort!(unique(labels))))
    nclasses = length(L)
    D = Matrix{Float32}(undef, nclasses, vocsize(voc))
    D .= smooth
   
    @showprogress dt=1 enabled=verbose desc="label-distribution block" for block in Iterators.partition(1:n, 1024)
        C = bagofwords_corpus(voc, corpus[block]; verbose=false)

        for (i, j) in enumerate(block)
            code = L[labels[j]]
            for (tokenID, _) in C[i]
                D[code, tokenID] += 1 # occs/M # log2(1 + occs)
            end
        end
    end

    weights = _compute_weights(weights, D, nclasses)
    model = VectorModel(ent, lw, voc)
    weights
    _compute_entropy(comb, model, D, weights, mindocs)
    model
end

function _compute_weights(weights, D, nclasses)
    weights isa String && (weights = Symbol(weights))
    if weights === nothing || weights === :none
        return ones(Float32, nclasses)
    end
    weights isa AbstractVector && return weights
    if weights === :balance
        W = vec(sum(D, dims=2))
        W .= sum(W) ./ W
        return W
    end

    error("Unknown weights=$weights nclasses=$nclasses")
end

function _compute_entropy(comb, model, D, weights, mindocs)
    maxent = log2(length(weights))

    @inbounds for tokenID in eachindex(model)
        m = ndocs(model, tokenID)
        if m < mindocs
            model.weight[tokenID] = 0.0
        else
            dist = @view D[:, tokenID]
            dist .= dist .* weights
            model.weight[tokenID] = combine_weight(comb, model, tokenID, entropy_(dist), maxent)
        end
    end
end

@inline global_weighting(model::VectorModel{EntropyWeighting}, tokenID) = weight(model, tokenID)
