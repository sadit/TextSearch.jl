# This file is a part of TextSearch.jl

#####
using CategoricalArrays
export EntropyWeighting

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
function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, voc::Vocabulary, corpus::AbstractVector{BOW}, labels::CategoricalArray;
            mindocs=1,
            smooth::Float64=0.0,
            weights=:balance
        )
    nclasses = length(levels(labels))
    D = fill(smooth, nclasses, length(voc))

    for i in eachindex(corpus)
        bow = corpus[i]
        code = labels.refs[i]
        
        for (tokenID, occs) in bow
            D[code, tokenID] += 1 # occs/M # log2(1 + occs)
        end
    end

    weights = _compute_weights(weights, D, nclasses)
    _compute_entropy(voc, D, weights, nclasses, mindocs)
    VectorModel(ent, lw, voc; mindocs)
end

function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, tok::Tokenizer, corpus::AbstractVector{<:AbstractString}, labels::CategoricalArray;
            bow=BOW(),
            mindocs=1,
            smooth::Float64=0.0,
            weights=:balance
        )
    nclasses = length(levels(labels))
    voc = Vocabulary(tok, corpus)
    D = fill(smooth, nclasses, length(voc))

    for i in eachindex(corpus)
        empty!(bow)
        vectorize(voc, tok, corpus[i], bow)
        
        code = labels.refs[i]
        for (tokenID, _) in bow
            D[code, tokenID] += 1 # occs/M # log2(1 + occs)
        end
    end

    weights = _compute_weights(weights, D, nclasses)
    _compute_entropy(voc, D, weights, nclasses, mindocs)
    VectorModel(ent, lw, voc; mindocs)
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

function _compute_entropy(V, D, weights, nclasses, mindocs)
    maxent = log2(nclasses)

    @inbounds for tokenID in eachindex(V.occs)
        if V.ndocs[tokenID] < mindocs
            V.weight[tokenID] = 0.0
        else
            dist = @view D[:, tokenID]
            dist .= dist .* weights
            e = 1.0 - entropy(dist) / maxent
            V.weight[tokenID] = e
        end
    end
end

global_weighting(m::VectorModel{EntropyWeighting}, tokenID) = @inbounds m.voc.weight[tokenID]
