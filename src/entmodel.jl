export EntModel, vectorize, id2token

type EntModel
    token2term::Dict{String,Term}
    config::TextConfig
end

function EntModel(model::DistModel)
    token2term = Dict{String,Term}()
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
        token2term[token] = Term(tokenID, maxent - e)
    end

    EntModel(token2term, model.config)
end

function id2token(model::EntModel)
    m = Dict{Int,String}()
    for (token, term) in model.token2term
        m[term.id] = token
    end

    m
end

function vectorize(text::String, model::EntModel; corrector::Function=identity)
    vec = Term[]
    bow = compute_bow(text, model.config)

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
    DocumentType(vec)
end
