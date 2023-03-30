# This file is a part of TextSearch.jl

function vectorize_knns!(D::Dict, model::SemanticVocabulary, tok)
    klex = model.sel.klex
    ksem = min(model.sel.ksem, size(model.knns, 1))

    res = getknnresult(klex)
    search(model, tok, res)
    sizehint!(D, length(res) * (1 + ksem))

    for p in res
        D[p.id] = get(D, p.id, 0f0) + 1f0
        for i in view(model.knns, 1:ksem, p.id)
            i == 0 && break
            D[i] = get(D, i, 0f0) + 1f0
        end
    end

    D
end

function token2id(model::SemanticVocabulary, tok::AbstractString)::UInt32
    id = token2id(model.voc, tok)::UInt32

    if id == 0
        klex = model.sel.klex
        ksem = min(model.sel.ksem, size(model.knns, 1))

        if ksem == 0
            res = getknnresult(klex)
            search(model, tok, res)
            id = argmin(res)::UInt32
        else
            buff = take!(TEXT_SEARCH_CACHES)
            try
                empty!(buff.vec)
                D = buff.vec
                vectorize_knns!(D, model, tok)
                id = length(D) == 0 ? zero(UInt32) : argmax(D)
            finally
                put!(TEXT_SEARCH_CACHES, buff)
            end
        end
    end

    id
end

function tokenize!(model::SemanticVocabulary, tokens::TokenizedText)
    for i in eachindex(tokens) 
        t = tokens[i]
        id = token2id(model, t)
        tokens[i] = token(model.voc, id)
    end

    tokens
end

function tokenize(model::SemanticVocabulary, text)
    tokenize!(model, tokenize(model.voc.textconfig, text))

end

function tokenize_corpus(model::SemanticVocabulary, corpus)
    n = length(corpus)
    arr = Vector{TokenizedText}(undef, n)
    Threads.@threads for i in 1:n
        arr[i] = tokenize!(model, tokenize(model.voc.textconfig, corpus[i]))
    end

    arr
end

function bagofwords!(bow::BOW, model::SemanticVocabulary{SelectCentralToken}, tokens::TokenizedText)
    for t in tokens 
        id = token2id(model, t)
        bow[id] = get(bow, id, 0) + 1
    end

    bow
end

function bagofwords(model::SemanticVocabulary, text)
    tokens = tokenize(model.voc.textconfig, text)
    bow = BOW()
    sizehint!(bow, length(tokens))
    bagofwords!(bow, model, tokens)
end


function vectorize(model::SemanticVocabulary, text; normalize=true)
    klex = model.sel.klex
    ksem = min(model.sel.ksem, size(model.knns, 1))

    res = getknnresult(klex)
    search(model, text, res)
    D = DVEC{UInt32,Float32}()
    sizehint!(D, length(res) * (1 + ksem))
    if ksem == 0
        for p in res
            D[p.id] = abs(p.weight)
        end 
    else
        for p in res
            D[p.id] = get(D, p.id, 0f0) + abs(p.weight)
            for i in view(model.knns, 1:ksem, p.id)
                i == 0 && break
                D[i] = get(D, i, 0f0) + 1f0
            end
        end 
    end

    normalize && normalize!(D)

    D
end

