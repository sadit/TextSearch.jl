# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: push!, append!
export EntModel, EntWeighting, EntFreqWeighting, EntTpWeighting, EntTfWeighting

struct EntWeighting <: WeightingType end
struct EntFreqWeighting <: WeightingType end
struct EntTpWeighting <: WeightingType end
struct EntTfWeighting <: WeightingType end

const WeightedVocabulary = Dict{Symbol,IdWeight}

mutable struct EntModel{W_<:WeightingType} <: TextModel
    weighting::W_
    tokens::WeightedVocabulary
    id2token::IdTokenMap
    m::Int
    n::Int
end

StructTypes.StructType(::Type{<:EntModel}) = StructTypes.Struct()
Base.show(io::IO, model::EntModel) = print(io, "{EntModel weighthing=$(model.weighting), vocsize=$(model.m)}")


Base.copy(e::EntModel; weighting=e.weighting, tokens=e.tokens, id2token=e.id2token, m=e.m, n=e.n) =
    EntModel(weighting, tokens, id2token, m, n)

function push!(model::EntModel, p::Pair)
    model.m += 1
    model.tokens[p[1]] = IdWeight(model.m, p[2])
    model.id2token[model.m] = p[1]
end

function append!(model::EntModel, weighted_words)
    for p in weighted_words
        push!(model, p)
    end
end

"""
    EntModel(model::DistModel, weighting::WeightingType; lower=0.0)
    EntModel(weigthing::WeightingType, corpus, y; smooth=3, minocc=3, weights=:balance, lower=0.0, nclasses=0)

Fits an EntModel using the already fitted DistModel. It accepts only symbols with a final weight higher or equal than `lower`.
Parameters:
    - `corpus` is text collection
    - `y` the set of associated labels (one-to-one with corpus)
    - `smooth` is a smoothing factor for the histogram.
    - `weights` accepts a list of weights (one per class) to be applied to the histogram
    - `lower` controls the minimum weight to be accepted
    - `nclasses` specifies the number of classes
	- `minocc`: minimum population to consider a token (without considering the smoothing factor).
"""
function EntModel(model::DistModel, weighting::WeightingType; lower=0.0)
    tokens = WeightedVocabulary()
    nclasses = length(model.sizes)
    maxent = log2(nclasses)

    i = 0
    @inbounds for (token, dist) in model.tokens
        i += 1
        e = 0.0
        pop = sum(dist)

        for j in 1:nclasses
            pj = dist[j] / pop

            if pj > 0.0
                e -= pj * log2(pj)
            end
        end

        e = 1.0 - e / maxent
        if e >= lower
           # tokens[token] = IdWeight(i, e * log2(length(string(token))))
            tokens[token] = IdWeight(i, e)
        end
    end

    id2token = Dict(w.id => t for (t, w) in tokens)
    EntModel(weighting, tokens, id2token, model.m, model.n)
end

function EntModel(weigthing::WeightingType, corpus, y; smooth=3, minocc=3, weights=:balance, lower=0.0, nclasses=0)
    dmodel = DistModel(corpus, y, nclasses=nclasses, weights=weights, fix=false, smooth=smooth, minocc=minocc)
    EntModel(dmodel, weigthing, lower=lower)
end

"""
    prune(model::EntModel, lower)

Prunes the model accepting only those symbols with a weight higher than `lower`

"""
function prune(model::EntModel, lower::Float64)
    tokens = WeightedVocabulary()
    for (t, w) in model.tokens
        if w.weight >= lower
            tokens[t] = IdWeight(w.id, w.weight)
        end
    end
    
    id2token = Dict(w.id => t for (t, w) in tokens)
    EntModel(model.weigthing, tokens, id2token, model.m, model.n)
end

"""
    prune_select_top(model::EntModel, k::Int)
    prune_select_top(model::EntModel, ratio::AbstractFloat)

Creates a new model preserving only the best `k` terms on `model`; the size can be indicated by the ratio of the database to be kept, i.e., ``0 < ratio < 1``.
"""
function prune_select_top(model::EntModel, k::Int)
    X = sort!(collect(model.tokens), by=x->x[2].weight, rev=true)
    
    tokens = WeightedVocabulary()
    for i in 1:k
        t, w = X[i]
        tokens[t] = IdWeight(w.id, w.weight)
    end

    id2token = IdTokenMap(w.id => t for (t, w) in tokens)
    EntModel(model.weighting, tokens, id2token, model.m, model.n)
end

prune_select_top(model::EntModel, ratio::AbstractFloat) = prune_select_top(model, floor(Int, length(model.tokens) * ratio))


"""
    vectorize(model::EntModel, bow::BOW; normalize=true)

Computes a weighted bow for the given `data`; the vector is scaled to the unit if `normalize` is true;
`data` is a bag of words.

"""
function vectorize(model::EntModel{T}, bow::BOW; normalize=true) where T
    len = 0

    if T === EntTpWeighting
        for v in values(bow)
            len += v
        end
    end

    maxfreq = T === EntTfWeighting ? maximum(bow) : 0.0

    vec = SVEC()
    for (token, freq) in bow
        t = get(model.tokens, token, nothing)
        if t === nothing
            continue
        end
    
        w = _weight(model.weighting, t.weight, freq, maxfreq, len)
        if w > 1e-6
            vec[t.id] = w
        end
    end

    if length(vec) == 0
        vec[rand(typemin(Int32):-1)] = 1e-6
    end
 
    normalize && normalize!(vec)
    vec    
end

_weight(::EntTpWeighting, ent, freq, maxfreq, n) = ent * freq / n
_weight(::EntFreqWeighting, ent, freq, maxfreq, n) = ent * freq
_weight(::EntTfWeighting, ent, freq, maxfreq, n) = ent * freq / maxfreq
_weight(::EntWeighting, ent, freq, maxfreq, n) = ent

function broadcastable(model::EntModel)
    (model,)
end
