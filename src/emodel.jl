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
function VectorModel(ent::EntropyWeighting, lw::LocalWeighting, corpus::AbstractVector{BOW}, labels=nothing;
        mindocs=1,
        smooth::Float64=0.0,
        weights=:balance
    )
    labels === nothing && error("EntropyWeighting requires labels as a categorical array to work")
    nclasses = length(levels(labels))
    D = Dict{UInt64,Vector{Float64}}()
    V = Vocabulary()
    
    for i in eachindex(corpus)
        vec = corpus[i]
        code = labels.refs[i]
        #M = sum(x for x in values(vec))
        for (t, occs) in vec
            s = get(V, t, UnknownTokenStats)
            V[t] = TokenStats(s.occs + occs, s.ndocs + 1, 0f0)
            dist = get(D, t, nothing)
            if dist === nothing
                D[t] = dist = fill(smooth, nclasses)
            end
            
            dist[code] += 1  # occs/M # log2(1 + occs)
        end
    end

    weights = weights isa String ? Symbol(weights) : weights

    if weights === :balance
        weights = sum(Vector{Float64}, values(D))
        weights .= sum(weights) ./ weights
    elseif weights === :none
        weights = ones(Float64, nclasses)
    end

    tokens, maxfreq = _create_vocabulary_with_entropy(V, D, weights, nclasses, mindocs)
    
    VectorModel(ent, lw, tokens, maxfreq, length(tokens), length(corpus))
end

function _create_vocabulary_with_entropy(V, D, weights, nclasses, mindocs)
    tokens = Vocabulary()
    tokenID = 0
    maxfreq = 0
    maxent = log2(nclasses)

    for (t, s) in V
        s.ndocs < mindocs && continue
        dist = D[t]
        dist .= dist .* weights
        e = 1.0 - entropy(dist) / maxent
        tokens[t] = TokenStats(s.occs, s.ndocs, e)
        maxfreq = max(maxfreq, s.occs)
    end

    tokens, maxfreq
end

global_weighting(::VectorModel{EntropyWeighting}, s::TokenStats) = s.weight
