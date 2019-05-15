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

            if pj > 0
                e -= pj * log2(pj)
            end
        end

        tokens[token] = 1.0 - e / maxent
    end

    EntModel(tokens, model.config)
end

function weighted_bow(model::EntModel, weighting::Type, data, modify_bow!::Function=identity)::BOW
    W = BOW()
    bag, maxfreq = compute_bow(model.config, data)
    len = 0
    for v in values(bag)
        len += v
    end
    bag = modify_bow!(bag)
    for (token, freq) in bag
        w = get(model.tokens, token, 0.0)
        if w > 0
            W[token] = w * freq / len
        end
    end
  
    W
end
