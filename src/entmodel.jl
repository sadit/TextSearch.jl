export EntModel

mutable struct EntModel <: Model
    tokens::Dict{Symbol,Float64}
    config::TextConfig
end

function fit(::Type{EntModel}, model::DistModel, b::Float64=-1.0)
    tokens = Dict{Symbol,Float64}()
    nclasses = length(model.sizes)
    maxent = log2(nclasses)

    @inbounds for (token, dist) in model.tokens
        if b == -1
            sumdist = sum(dist)
            if sumdist < nclasses
                bb = sumdist * 0.5 / nclasses
            else
                bb = 0
            end
        else
            bb = b
        end
        e = 0.0
        pop = bb * nclasses + sum(dist)

        for j in 1:nclasses
            pj = (dist[j] + bb) / pop

            if pj > 0
                e -= pj * log2(pj)
            end
        end

        tokens[token] = 1.0 - e / maxent
    end

    EntModel(tokens, model.config)
end

function weighted_bow(model::EntModel, weighting::Type, data, modify_bow!::Function=identity)::Dict{Symbol, Float64}
    W = Dict{Symbol, Float64}()
    bag, maxfreq = compute_bow(model.config, data)
    bag = modify_bow!(bag)
    for (token, freq) in bag
        # get(model.vocab, token, UNKNOWN_TOKEN)
        # global_tokendata.freq == 0 && continue
        w = get(model.tokens, token, 0.0)
        if w > 0
            W[token] = w
        end
        # ww = _weight(weighting, freq, maxfreq, model.n, global_tokendata.freq)
        # W[token] = w
    end
  
    W
end
