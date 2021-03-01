# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

#####
using CategoricalArrays
export EntropyWeighting

struct EntropyWeighting <: GlobalWeighting end

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
    VectorModel(weighting::WeightingType, corpus::BOW; minocc::Integer=1)

Trains a vector model using the input corpus. 
"""
function VectorModel(local_weighting::LocalWeighting, global_weighting::EntropyWeighting, corpus::AbstractVector{BOW}, labels; minocc::Integer=1, lowerweight=0.0, smooth=0.0)
    tokens = Vocabulary()
    id2token = IdTokenMap()
    nclasses = length(levels(labels))
    smooth = Float64(smooth)

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

    weights = sum(Vector{Float64}, values(D))
    weights .= sum(weights) ./ weights
    tokens = Vocabulary()
    tokenID = 0
    maxfreq = 0
    maxent = log2(nclasses)

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
    
    VectorModel(local_weighting, global_weighting, tokens, id2token, maxfreq, length(tokens), length(corpus))
end

global_weighting(::EntropyWeighting, s::TokenStats, m::TextModel) = s.weight
