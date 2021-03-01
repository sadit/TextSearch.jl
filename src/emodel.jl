# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

#####
using CategoricalArrays
export EntropyWeighting

"""
    EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)

Entropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token
"""
struct EntropyWeighting <: GlobalWeighting
    smooth::Float64
    lowerweight::Float64
    weights
end

EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance) = EntropyWeighting(smooth, lowerweight, weights)

function entropy(dist)
    popsize = sum(dist)
    e = 0.0
    pop = sum(dist)

    for x in dist
        p = x / pop

        if p > 0.0
            e -= p * log2(p)
        end
    end

    e
end

"""
    VectorModel(lw::LocalWeighting, ent::EntropyWeighting, corpus::BOW, labels; minocc::Integer=1)

Creates a vector model using the input corpus. 
"""
function VectorModel(lw::LocalWeighting, ent::EntropyWeighting, corpus::AbstractVector{BOW}, labels=nothing; minocc::Integer=1)
    labels === nothing && error("EntropyWeighting requires labels as a categorical array to work")

    tokens = Vocabulary()
    id2token = IdTokenMap()
    nclasses = length(levels(labels))
    smooth = Float64(ent.smooth)

    D = Dict{Symbol,Vector{Float64}}()
    V = Vocabulary()
    
    for i in eachindex(corpus)
        vec = corpus[i]
        for (t, occ) in vec
            s = get(V, t, UnknownTokenStats)
            V[t] = TokenStats(0, s.occs + occ, s.ndocs + 1, 0f0)
            dist = get(D, t, nothing)
            if dist === nothing
                dist = D[t] = fill(smooth, nclasses)
            end
            
            dist[labels.refs[i]] += 1
        end
    end

    if ent.weights === :balance
        weights = sum(Vector{Float64}, values(D))
        weights .= sum(weights) ./ weights
    elseif ent.weights === :none
        weights = ones(Float64, nclasses)
    else
        weights = ent.weights
    end

    tokens = Vocabulary()
    tokenID = 0
    maxfreq = 0
    maxent = log2(nclasses)

    lowerweight = ent.lowerweight

    for (t, s) in V
        s.occs < minocc && continue
        dist = D[t]
        dist .= dist .* weights

        e = 1.0 - entropy(dist) / maxent
        e < lowerweight && continue

        tokenID += 1
        id2token[tokenID] = t
        tokens[t] = TokenStats(tokenID, s.occs, s.ndocs, e)
        maxfreq = max(maxfreq, s.occs)
    end
    
    VectorModel(lw, ent, tokens, id2token, maxfreq, length(tokens), length(corpus))
end

global_weighting(::EntropyWeighting, s::TokenStats, m::TextModel) = s.weight
