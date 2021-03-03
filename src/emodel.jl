# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

#####
using CategoricalArrays
export EntropyWeighting

"""
    EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)

Entropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token
"""
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
    VectorModel(lw::LocalWeighting, ent::EntropyWeighting, corpus::BOW, labels;
        minocc::Integer=1,
        smooth::Float64=0.0,
        weights=:balance,
        lowerweight=0.0
    )

Creates a vector model using the input corpus. 
"""
function VectorModel(lw::LocalWeighting, ent::EntropyWeighting, corpus::AbstractVector{BOW}, labels=nothing;
        minocc::Integer=1,
        smooth::Float64=0.0,
        weights=:balance,
        lowerweight=0.0
    )
    labels === nothing && error("EntropyWeighting requires labels as a categorical array to work")

    nclasses = length(levels(labels))

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

    if weights isa String
        weights = Symbol(weights)
    end

    if weights === :balance
        weights = sum(Vector{Float64}, values(D))
        weights .= sum(weights) ./ weights
    elseif weights === :none
        weights = ones(Float64, nclasses)
    end

    tokens, id2token, maxfreq = _create_vocabulary_with_entropy(V, D, weights, nclasses, minocc, lowerweight)
    
    VectorModel(lw, ent, tokens, id2token, maxfreq, length(tokens), length(corpus))
end

function _create_vocabulary_with_entropy(V, D, weights, nclasses, minocc, lowerweight)
    tokens = Vocabulary()
    id2token = IdTokenMap()

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

    tokens, id2token, maxfreq
end

global_weighting(::EntropyWeighting, s::TokenStats, m::TextModel) = s.weight
