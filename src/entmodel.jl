export EntModel

mutable struct EntModel <: Model
    tokens::BOW
    config::TextConfig
end

function smooth_factor(dist::AbstractVector)::Float64
    s = sum(dist)
    s < length(dist) ? 1.0 : 0.0
end

"""
    fit(::Type{EntModel}, model::DistModel, smooth::Function=smooth_factor; lower=0.001, normalize_words::Function=identity)


Fits an EntModel using the already fitted DistModel; the `smooth` function is called to compute the smoothing factor
for a given histogram. It accepts only symbols with a final weight higher or equal than `lower`.
Words are normalized with `normalize_words`.

"""
function fit(::Type{EntModel}, model::DistModel, smooth::Function=smooth_factor; lower=0.001, normalize_words::Function=identity)
    tokens = BOW()
    nclasses = length(model.sizes)
    maxent = log2(nclasses)

    @inbounds for (token, dist) in model.tokens
        b = smooth(dist)
        e = 0.0
        pop = b * nclasses + sum(dist)

        for j in 1:nclasses
            pj = (dist[j] + b) / pop

            if pj > 0.0
                e -= pj * log2(pj)
            end
        end
        e = 1.0 - e / maxent
        if e >= lower
            tokens[token] = e
        end
    end

    EntModel(tokens, model.config)
end

function fit(::Type{EntModel}, config::TextConfig, corpus, y; nclasses=0, weights=:balance, smooth=smooth_factor, lower=0.001)
    dmodel = fit(DistModel, config, corpus, y, nclasses=nclasses, weights=weights)
    fit(EntModel, dmodel, smooth, lower=lower)
end

"""
    prune(model::EntModel, lower)

Prunes the model accepting only those symbols with a weight higher than `lower`

"""
function prune(model::EntModel, lower)
    tokens = BOW()
    for (t, ent) in model.tokens
        if ent >= lower
            tokens[t] = ent
        end
    end
    
    EntModel(tokens, model.config)
end

abstract type EntTfModel end
abstract type EntTpModel end

"""
    vectorize(model::EntModel, data)::BOW
    vectorize(model::EntModel, scheme::Type, data)::BOW

Computes a weighted bow for the given `data`; the vector is scaled to the unit if `normalize` is true;
`data` is an string or an array of strings. The weighting scheme may be any of `EntTfModel`, `EntTpModel`, or `EntModel`;
the default scheme is EntTpModel
"""
function vectorize(model::EntModel, scheme::Type{T}, data::DataType; normalize=true)::BOW where
        T <: Union{EntTfModel,EntTpModel,EntModel} where
        DataType <: Union{AbstractString, AbstractVector{S}} where
        S <: AbstractString

    bow, maxfreq = compute_bow(tokenize(model.config, data))
    vectorize(model, scheme, bow, normalize=normalize)
end

"""
    vectorize(model::EntModel, scheme::Type{T}, bow::BOW; normalize=true)::BOW

Computes a weighted bow for the given `data`; the vector is scaled to the unit if `normalize` is true;
`data` is an bag of words. The weighting scheme may be any of `EntTfModel`, `EntTpModel`, or `EntModel`

"""
function vectorize(model::EntModel, scheme::Type{T}, bow::BOW; normalize=true)::BOW where T <: Union{EntTfModel,EntTpModel,EntModel}
    len = 0

    for v in values(bow)
        len += v
    end

    for (token, freq) in bow
        w = get(model.tokens, token, 0.0)
        w = _weight(scheme, w, freq,  len)
        if w > 0.0
            bow[token] = w
        else
            delete!(bow, token)
        end
    end
 
    normalize && normalize!(bow)
    bow    
end

vectorize(model::EntModel, data; normalize=true) = vectorize(model, EntTpModel, data, normalize=normalize)

_weight(::Type{EntTpModel}, ent, freq, n) = ent * freq / n
_weight(::Type{EntTfModel}, ent, freq, n) = ent * freq
_weight(::Type{EntModel}, ent, freq, n) = ent

function broadcastable(model::EntModel)
    (model,)
end
