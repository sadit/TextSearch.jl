export DistModel, feed!, fix!, id2token

mutable struct TokenDist
    id::UInt64
    dist::Vector{Float64}

    TokenDist(id, nclasses::Int) = new(id, zeros(Float64, nclasses))
end

mutable struct DistModel <: Model
    tokens::Dict{Symbol,TokenDist}
    config::TextConfig
    sizes::Vector{Int}
end

function fit(::Type{DistModel}, config::TextConfig, corpus, y; nclasses=0, norm_by=minimum)
    if nclasses == 0
        nclasses = y |> unique |> length
    end
    
    model = DistModel(Dict{Symbol,TokenDist}(), config, zeros(Int, nclasses))
 
    n = 0
    for (klass, text) in zip(y, corpus)
        @show text
        for token in tokenize(config, text)
            if !haskey(model.tokens, token)
                model.tokens[token] = TokenDist(length(model.tokens), nclasses)
            end

            model.tokens[token].dist[klass] += 1
        end

        model.sizes[klass] += 1
        n += 1
        if n % 10000 == 1
            @info "DistModel: $(sum(model.sizes)) processed items"
        end
    end
    
    normalize!(model, norm_by)
    fix!(model)
    model
end

function feed!(model::DistModel, corpus, y)
    config = model.config
    nclasses = length(model.sizes)
    n = 0
    for (text, klass) in zip(corpus, y)
        for token in tokenize(config, text)
            if !haskey(model.tokens, token)
                model.tokens[token] = TokenDist(length(model.tokens), nclasses)
            end

            model.tokens[token].dist[klass] += 1
        end

        model.sizes[klass] += 1
        n += 1
        if n % 10000 == 1
            @info "DistModel: $(sum(model.sizes)) processed items"
        end
    end
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

function fit(model::DistModel, corpus, y; norm_by=nothing)
    feed!(model, corpus, y)
    if norm_by != nothing
        normalize!(model, norm_by)
    end
    fix!(model)
end

function id2token(model::DistModel)
    nclasses = length(model.sizes)
    H = Dict{UInt64,Symbol}()
    for (token, d) in model.tokens
        b = d.id * nclasses
        for i in 1:nclasses
            H[b + i] = Symbol(token, i)
        end
    end

    H
end

function vectorize(model::DistModel, data)
    nclasses = length(model.sizes)
    bow, maxfreq = compute_bow(model.config, data)
    vec = SparseVectorEntry[]
    sizehint!(vec, length(bow) * nclasses)

    for (token, freq) in bow
        if haskey(model.tokens, token)
            tokendist = model.tokens[token]
            b = tokendist.id * nclasses
            for i in 1:nclasses
                push!(vec, SparseVectorEntry(b+i, tokendist.dist[i]))
            end
        end
    end

    SparseVector(vec)
end
