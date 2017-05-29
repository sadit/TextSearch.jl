export DistModel, feed!, fix!, fit!, vectorize, id2token

type DistModel <: Model
    token2id::Dict{String,UInt64}
    vhist::Vector{Float64}
    config::TextConfig
    sizes::Vector{Int}
end

function DistModel(config::TextConfig, nclasses)
    DistModel(Dict{String,Int}(), Vector{Float64}(), config, zeros(Int, nclasses))
end

function feed!(model::DistModel, corpus, get_text_klass::Function=identity)
    nclasses = length(model.sizes)
    Z = zeros(Int32, nclasses)
    n = 0
    for item in corpus
        text, klass = get_text_klass(item)
        # @show text, klass
        for (token, freq) in compute_bow(text, model.config)
            idtoken = get(model.token2id, token, -1)
            if idtoken == -1
                idtoken = length(model.token2id)
                model.token2id[token] = idtoken
                append!(model.vhist, Z)
            end

            model.vhist[idtoken * nclasses + klass] += freq
        end

        model.sizes[klass] += 1
        n += 1
        if n % 10000 == 1
            info("DistModel", sum(model.sizes), "processed items")
        end
    end

    model
end

function id2token(model::DistModel)
    nclasses = length(model.sizes)
    m = Dict{UInt64,String}()

    for (token, id) in model.token2id
        b = id * nclasses
        for i in 1:nclasses
            m[b + i] = string(token, '<', i,'>')
        end
    end

    m
end

function fix!(model::DistModel)
    nclasses = length(model.sizes)
    nterms = length(model.token2id)

    for i in 1:nterms
        b = (i-1) * nclasses
        s = 0.0
        for j in 1:nclasses
            s += model.vhist[b + j]
        end

        for j in 1:nclasses
            model.vhist[b + j] /= s
        end
    end

    model
end

function fit!(model::DistModel, corpus, get_text_klass::Function=identity)
    feed!(model, corpus, get_text_klass)
    fix!(model)
end

function hist(model::DistModel)
    @show model.token2id
    nclasses = length(model.sizes)

    for (token, id) in model.token2id
        i = id * nclasses
        info(token, ": ", @view model.vhist[i+1:i+nclasses])
    end
end

function vectorize(text::String, model::DistModel; corrector::Function=identity)
    nclasses = length(model.sizes)
    bow = compute_bow(text, model.config)
    vec = WeightedToken[]
    sizehint!(vec, length(bow) * nclasses)

    for (token, freq) in bow
        idtoken = try
            token = corrector(token)
            model.token2id[token]
        catch err
            continue
        end

        b = idtoken * nclasses

        for i in 1:nclasses
            # info((token, i, @view model.vhist[b+1:b+nclasses]))
            push!(vec, WeightedToken(b+i, model.vhist[b+i]))
        end
    end

    sort!(vec, by=(x) -> x.id)
    VBOW(vec)
end
