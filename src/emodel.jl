# This file is a part of TextSearch.jl

#####
using CategoricalArrays
export EntropyWeighting, categorical, NormalizedEntropy, SigmoidPenalizeFewSamples, CombineWeighting

abstract type CombineWeighting end
struct NormalizedEntropy <: CombineWeighting end
combine_weight(::NormalizedEntropy, model, tokenID, entropy, maxent)::Float32 = 1.0 - entropy / maxent 
# the entropy scores the discrimination power of the term while log(m) weights
# the term w.r.t the available evidency. The current form tries to equalize the
# scales
struct PenalizeFewSamples <: CombineWeighting end
combine_weight(::PenalizeFewSamples, model, tokenID, entropy, maxent)::Float32 = (maxent - entropy) * log2(ndocs(model, tokenID)) 

struct SigmoidPenalizeFewSamples <: CombineWeighting end
combine_weight(::SigmoidPenalizeFewSamples, model, tokenID, entropy, maxent)::Float32 = (1 - entropy/maxent) * (1-1/(1+log2(ndocs(model, tokenID))))

"""
    EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)

Entropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token
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

categorical_labels(labels::AbstractVector{<:CategoricalValue}) = labels
categorical_labels(labels::AbstractVector{T}) where {T<:Union{AbstractString,Integer,Symbol}} = categorical(labels) 
categorical_labels(labels::AbstractCategoricalVector) = labels

"""
    VectorModel(ent::EntropyWeighting, lw::LocalWeighting, corpus::BOW, labels;
        mindocs::Integer=1,
        smooth::Float64=0.0,
        weights=:balance
        comb::CombineWeighting=NormalizedEntropy(),
    )

Creates a vector model using the input corpus. 
"""
function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, voc::Vocabulary, corpus::AbstractVector, labels::AbstractVector;
            mindocs=3,
            smooth=3,
            weights=:balance,
            comb::CombineWeighting=NormalizedEntropy(),
            minbatch=0, verbose=true
        )
    labels = categorical(labels)
    @assert length(labels) == length(corpus)
    labels = categorical_labels(labels)
    n = length(labels)
    nclasses = length(levels(labels))
    D = Matrix{Float32}(undef, nclasses, vocsize(voc))
    D .= smooth
   
    @showprogress dt=1 enabled=verbose desc="label-distribution block" for block in Iterators.partition(1:n, 1024)
        C = bagofwords_corpus(voc, corpus[block]; verbose=false)

        for (i, j) in enumerate(block)
            code = levelcode(labels[j])
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
