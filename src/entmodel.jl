export EntModel, id2token

mutable struct EntModel <: Model
    tokens::Dict{Symbol,SparseVectorEntry}
    config::TextConfig
end

function fit(::Type{EntModel}, model::DistModel, initial)
    tokens = Dict{Symbol,SparseVectorEntry}()
    nclasses = length(model.sizes)
    tokenID = 0
    maxent = log2(nclasses)

    
    @inbounds for (token, tokendist) in model.tokens
        e = 0.0
        pop = initial * nclasses + sum(tokendist.dist)

        for j in 1:nclasses
            pj = (tokendist.dist[j] + initial) / pop

            if pj > 0
                e -= pj * log2(pj)
            end
        end
        tokenID += 1
        tokens[token] = SparseVectorEntry(tokenID, maxent - e)
    end

    EntModel(tokens, model.config)
end

function id2token(model::EntModel)
    m = Dict{UInt64,Symbol}()
    for (token, term) in model.tokens
        m[term.id] = token
    end

    m
end

function vectorize(model::EntModel, data)
    bow, maxfreq = compute_bow(model.config, data)
    vec = Vector{SparseVectorEntry}(undef, length(bow))

    i = 0
    for (token, freq) in bow
        if haskey(model.tokens, token)
            i += 1
            vec[i] = model.tokens[token]
        end
    end

    resize!(vec, i)
    SparseVector(vec)
end

function weighted_bow(model::EntModel, data)
    W = Dict{Symbol, Float64}()
    bow, maxfreq = compute_bow(model.config, data)
    
    s = 0.0
    for p in bow
        if haskey(model.tokens, p.first)
            i += 1
            w = models.tokens[p.first].weight
            W[p.first] = w
            s += w * w
        end
    end

    W, sqrt(s)
end
