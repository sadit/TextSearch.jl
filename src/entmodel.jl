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
    fit(::Type{EntModel}, model::DistModel, smooth::Function=smooth_factor; lower=0.001)


Fits an EntModel using the already fitted DistModel; the `smooth` function is called to compute the smoothing factor
for a given histogram. It accepts only symbols with a final weight higher or equal than `lower`.
"""
function fit(::Type{EntModel}, model::DistModel, smooth::Function=smooth_factor; lower=0.001)
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

"""
    weighted_bow(model::EntModel, data, modify_bow!::Function=identity)::BOW
    weighted_bow(model::EntModel, ::Type, data, modify_bow!::Function=identity)::BOW

Computes a weighted bow for a given `data`; the weighting type is ignored when model is an EntModel.
"""
function weighted_bow(model::EntModel, data, modify_bow!::Function=identity)::BOW
    bow, maxfreq = compute_bow(model.config, data)
    len = 0
    for v in values(bow)
        len += v
    end
    bow = modify_bow!(bow)
    for (token, freq) in bow
        w = get(model.tokens, token, 0.0)
        m = freq / len
        if w > 0.0
            bow[token] = w * m
        else
            delete!(bow, token)
        end
    end

    bow    
end

weighted_bow(model::EntModel, ::Type, data, modify_bow!::Function=identity) = weighted_bow(model, data, modify_bow!)