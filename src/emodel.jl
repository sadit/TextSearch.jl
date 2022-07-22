# This file is a part of TextSearch.jl

#####
using CategoricalArrays
export EntropyWeighting, categorical

"""
    EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)

Entropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token
"""
struct EntropyWeighting <: GlobalWeighting end

function entropy(dist)
    e = 0.0
    ipop = 1/sum(dist)

    for x in dist
        p = x * ipop

        if p > 0.0
            e -= p * log2(p)
        end
    end

    e
end

"""
    VectorModel(ent::EntropyWeighting, lw::LocalWeighting, corpus::BOW, labels;
        mindocs::Integer=1,
        smooth::Float64=0.0,
        weights=:balance,
        lowerweight=0.0
    )

Creates a vector model using the input corpus. 
"""
function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, voc::Vocabulary, corpus::AbstractVector{BOW}, labels;
            mindocs=1,
            smooth::Float64=0.0,
            weights=:balance
        )
    nclasses = length(levels(labels))
    D = fill(smooth, nclasses, vocsize(voc))

    for (i, bow) in enumerate(corpus)
        code = levelcode(labels[i])
        for (tokenID, _) in bow
            D[code, tokenID] += 1 # occs/M # log2(1 + occs)
        end
    end

    weights = _compute_weights(weights, D, nclasses)
    model = VectorModel(ent, lw, voc)
    _compute_entropy(model, D, weights, nclasses, mindocs)
    model
end

function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, textconfig::TextConfig, corpus::AbstractVector, labels;
            mindocs=1,
            smooth::Float64=0.0,
            weights=:balance,
            minbatch=0
        )
    nclasses = length(levels(labels))
    corpus_tokens = tokenize_corpus(textconfig, corpus; minbatch)
    voc = Vocabulary(textconfig, corpus_tokens)
    D = fill(smooth, nclasses, length(voc))
    bow = BOW()
    
    for (i, tokens) in enumerate(corpus_tokens)
        empty!(bow)
        vectorize(voc, tokens, bow)
        
        code = levelcode(labels[i])
        for (tokenID, _) in bow
            D[code, tokenID] += 1 # occs/M # log2(1 + occs)
        end
    end

    weights = _compute_weights(weights, D, nclasses)
    model = VectorModel(ent, lw, voc)
    _compute_entropy(model, D, weights, nclasses, mindocs)
    model
end

function _compute_weights(weights, D, nclasses)
    weights isa String ? Symbol(weights) : weights

    if weights === :balance
        weights = vec(sum(D, dims=2))
        weights .= sum(weights) ./ weights
    elseif weights === :none
        weights = ones(Float64, nclasses)
    end

    weights
end

function _compute_entropy(model, D, weights, nclasses, mindocs)
    maxent = log2(nclasses)

    @inbounds for tokenID in eachindex(model)
        if ndocs(model, tokenID) < mindocs
            model.weight[tokenID] = 0.0
        else
            dist = @view D[:, tokenID]
            dist .= dist .* weights
            e = 1.0 - entropy(dist) / maxent
            model.weight[tokenID] = e
        end
    end
end

@inline global_weighting(model::VectorModel{EntropyWeighting}, tokenID) = weight(model, tokenID)
