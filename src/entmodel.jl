export EntModel, vectorize, id2token

type EntModel
    token2term::Dict{String,WeightedToken}
    config::TextConfig
end

function EntModel(model::DistModel)
    token2term = Dict{String,WeightedToken}()
    nclasses = length(model.sizes)
    tokenID = 0
    maxent = log2(nclasses)
    for (token, id) in model.token2id
        b = id * nclasses  # id starts in 0
        e = 0.0

        for j in 1:nclasses
            pj = model.vhist[b+j]
            if pj > 0
                e -= pj * log2(pj)
            end
        end
        tokenID += 1
        token2term[token] = WeightedToken(tokenID, maxent - e)
    end

    EntModel(token2term, model.config)
end

function id2token(model::EntModel)
    m = Dict{UInt64,String}()
    for (token, term) in model.token2term
        m[term.id] = token
    end

    m
end

function vectorize(text::String, model::EntModel; corrector::Function=identity)
    bow = compute_bow(text, model.config)
    vec = Vector{WeightedToken}()
    sizehint!(vec, length(bow))

    for (token, freq) in bow
        term = try
            token = corrector(token)
            model.token2term[token]
        catch err
            continue
        end
        push!(vec, term)
    end

    sort!(vec, by=(x) -> x.id)
    VBOW(vec)
end
