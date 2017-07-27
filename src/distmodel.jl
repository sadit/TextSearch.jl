import Base: normalize!
export DistModel, feed!, fix!, fit!, vectorize, id2token

type TokenDist
    id::UInt64
    dist::Vector{Float64}
end

TokenDist(id, nclasses::Int) = TokenDist(id, zeros(Float64, nclasses))

type DistModel <: Model
    tokens::Dict{String,TokenDist}
    config::TextConfig
    sizes::Vector{Int}
end

function DistModel(config::TextConfig, nclasses)
    DistModel(Dict{String,TokenDist}(), config, zeros(Int, nclasses))
end

function feed!(model::DistModel, corpus; get_text_klass::Function=identity)
    nclasses = length(model.sizes)
    Z = zeros(Int32, nclasses)
    n = 0
    for item in corpus
        text, klass = get_text_klass(item)

        for (token, freq) in compute_bow(text, model.config)
            idtoken = get(model.tokens, token, 0)
            if idtoken == 0
                idtoken = length(model.tokens)
                model.tokens[token] = TokenDist(idtoken, nclasses)
            end

            model.tokens[token].dist[klass] += freq
        end

        model.sizes[klass] += 1
        n += 1
        if n % 10000 == 1
            info("DistModel: ", sum(model.sizes), " processed items")
        end
    end

    model
end

function id2token(model::DistModel)
    nclasses = length(model.sizes)
    H = Dict{UInt64,String}()
    for (token, d) in model.tokens
        b = d.id * nclasses
        for i in 1:nclasses
            H[b + i] = string(token,'<',i,'>')
        end
    end

    H
end

function normalize!(model::DistModel, by=minimum)
    nclasses = length(model.sizes)
    val = by(model.sizes)

    for (token, tokendist) in model.tokens
        for i in 1:nclasses
            tokendist.dist[i] *= val / model.sizes[i]
        end
    end
end

function fix!(model::DistModel)
    nclasses = length(model.sizes)
    nterms = length(model.tokens)

    for (token, tokendist) in model.tokens
        s = sum(tokendist.dist)
        for i in 1:nclasses
            tokendist.dist[i] /= s
        end

    end

    model
end

function fit!(model::DistModel, corpus; get_text_klass::Function=identity, normalize=nothing)
    feed!(model, corpus, get_text_klass=get_text_klass)
    if normalize != nothing
        normalize!(model, normalize)
    end
    fix!(model)
end

function vectorize(text::String, model::DistModel; corrector::Function=identity)
    nclasses = length(model.sizes)
    bow = compute_bow(text, model.config)
    vec = WeightedToken[]
    sizehint!(vec, length(bow) * nclasses)

    for (token, freq) in bow
        tokendist = try
            token = corrector(token)
            model.tokens[token]
        catch err
            continue
        end

        b = tokendist.id * nclasses
        for i in 1:nclasses
            push!(vec, WeightedToken(b+i, tokendist.dist[i]))
        end
    end

    sort!(vec, by=(x) -> x.id)
    VBOW(vec)
end
