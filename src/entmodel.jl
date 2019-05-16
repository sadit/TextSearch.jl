export EntModel

mutable struct EntModel <: Model
    tokens::BOW
    config::TextConfig
end

function smooth_factor(dist::AbstractVector)::Float64
    s = sum(dist)
    s < length(dist) ? 1.0 : 0.0
end

function fit(::Type{EntModel}, model::DistModel, smooth::Function=smooth_factor)
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

        tokens[token] = 1.0 - e / maxent
    end

    EntModel(tokens, model.config)
end

function weighted_bow(model::EntModel, weighting::Type, data, modify_bow!::Function=identity)::BOW
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
