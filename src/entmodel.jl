export EntModel, EntTfModel, EntTpModel

struct IdWeight
    id::Int32
    weight::Float32
end

const WeightedVocabulary = Dict{Symbol,IdWeight}

mutable struct EntModel <: Model
    config::TextConfig
    tokens::WeightedVocabulary
    id2token::Dict{Int32,Symbol}
    m::Int
    n::Int
end

"""
    fit(::Type{EntModel}, model::DistModel; lower=0.0, normalize_words::Function=identity)
    fit(::Type{EntModel}, config::TextConfig, corpus, y; smooth=3, weights=:balance, lower=0.0, nclasses=0)

Fits an EntModel using the already fitted DistModel. It accepts only symbols with a final weight higher or equal than `lower`.
Parameters:
    - `corpus` is text collection
    - `y` the set of associated labels (one-to-one with corpus)
    - `smooth` is a smoothing factor for the histogram.
    - `weights` accepts a list of weights (one per class) to be applied to the histogram
    - `lower` controls the minimum weight to be accepted
    - `nclasses` specifies the number of classes
"""
function fit(::Type{EntModel}, model::DistModel; lower=0.0)
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
            tokens[token] = IdWeight(i, e)
        end
    end

    id2token = Dict(w.id => t for (t, w) in tokens)
    EntModel(model.config, tokens, id2token, model.m, model.n)
end

function fit(::Type{EntModel}, config::TextConfig, corpus, y; smooth=3, weights=:balance, lower=0.0, nclasses=0)
    dmodel = fit(DistModel, config, corpus, y, nclasses=nclasses, weights=weights, fix=false, smooth=smooth)
    fit(EntModel, dmodel, lower=lower)
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
    EntModel(model.config, tokens, id2token, model.m, model.n)
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

    id2token = Dict(w.id => t for (t, w) in tokens)
    EntModel(model.config, tokens, id2token, model.m, model.n)
end

prune_select_top(model::EntModel, ratio::AbstractFloat) = prune_select_top(model, floor(Int, length(model.tokens) * ratio))

abstract type EntTfModel end
abstract type EntTpModel end

"""
    vectorize(model::EntModel, data)
    vectorize(model::EntModel, scheme::Type, data)

Computes a weighted bow for the given `data`; the vector is scaled to the unit if `normalize` is true;
`data` is an string or an array of strings. The weighting scheme may be any of `EntTfModel`, `EntTpModel`, or `EntModel`;
the default scheme is EntTpModel
"""
function vectorize(model::EntModel, scheme::Type{T}, data::DataType; normalize=true) where
        T <: Union{EntTfModel,EntTpModel,EntModel} where
        DataType <: Union{AbstractString, AbstractVector{S}} where
        S <: AbstractString

    bow, maxfreq = compute_bow(tokenize(model.config, data))
    vectorize(model, scheme, bow, normalize=normalize)
end

"""
    vectorize(model::EntModel, scheme::Type{T}, bow::BOW; normalize=true)

Computes a weighted bow for the given `data`; the vector is scaled to the unit if `normalize` is true;
`data` is an bag of words. The weighting scheme may be any of `EntTfModel`, `EntTpModel`, or `EntModel`

"""
function vectorize(model::EntModel, scheme::Type{T}, bow::BOW; normalize=true) where T <: Union{EntTfModel,EntTpModel,EntModel} where Tv<:Real
    len = 0

    for v in values(bow)
        len += v
    end

    vec = SVEC()
    for (token, freq) in bow
        t = get(model.tokens, token, nothing)
        if t === nothing
            continue
        end
    
        w = _weight(scheme, t.weight, freq,  len)
        if w > 1e-6
            vec[t.id] = w
        end
    end
 
    normalize && normalize!(vec)
    vec    
end

vectorize(model::EntModel, data; normalize=true) = vectorize(model, EntTpModel, data, normalize=normalize)

_weight(::Type{EntTpModel}, ent, freq, n) = ent * freq / n
_weight(::Type{EntTfModel}, ent, freq, n) = ent * freq
_weight(::Type{EntModel}, ent, freq, n) = ent

function broadcastable(model::EntModel)
    (model,)
end
