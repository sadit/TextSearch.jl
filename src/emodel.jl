# This file is a part of TextSearch.jl

#####
using CategoricalArrays
export EntropyWeighting, categorical

"""
    EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)

Entropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token
"""
struct EntropyWeighting <: GlobalWeighting end

function entropy_(dist)
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

categorical_labels(labels::AbstractVector{<:CategoricalValue}) = labels
categorical_labels(labels::AbstractVector{T}) where {T<:Union{AbstractString,Integer,Symbol}} = categorical(labels) 
categorical_labels(labels::AbstractCategoricalVector) = labels

"""
    VectorModel(ent::EntropyWeighting, lw::LocalWeighting, corpus::BOW, labels;
        mindocs::Integer=1,
        smooth::Float64=0.0,
        weights=:balance,
        lowerweight=0.0
    )

Creates a vector model using the input corpus. 
"""
function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, voc::Vocabulary, corpus::AbstractVector, labels::AbstractVector;
            mindocs=1,
            smooth::Float64=0.0,
            weights=:balance,
            minbatch=0
        )
    @assert length(labels) == length(corpus)
    labels = categorical_labels(labels)
    n = length(labels)
    nclasses = length(levels(labels))
    D = fill(smooth, nclasses, vocsize(voc))
   
    for block in Iterators.partition(1:n, 10^5)
        C = bagofwords_corpus(voc, corpus[block])

        for i in block
            code = levelcode(labels[i])
            for (tokenID, _) in C[i]
                D[code, tokenID] += 1 # occs/M # log2(1 + occs)
            end
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
            e = 1.0 - entropy_(dist) / maxent
            model.weight[tokenID] = e
        end
    end
end

@inline global_weighting(model::VectorModel{EntropyWeighting}, tokenID) = weight(model, tokenID)
