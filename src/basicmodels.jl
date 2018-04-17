export TextModel, VectorModel, fit!, inverse_vbow,
    compute_vbow, compute_bow, vectorize, vectorize_tfidf,
    vectorize_tf, vectorize_idf, vectorize_rawfreq, id2token

abstract type Model end

mutable struct VectorModel <: Model
    token2id::Dict{String,Int}
    weights::Dict{Int,Float64}
    size::Int64
    filter_low::Int
    filter_high::Float64
    config::TextConfig
end

function id2token(model::VectorModel)
    m = Dict{Int,String}()
    for (t, id) in model.token2id
        m[id] = t
    end

    m
end

function inverse_vbow(vec, vocmap)
    s = collect(vec.tokens)
    sort!(s, by=x -> -x.weight)
    [(vocmap[token.id], token.weight) for token in s]
end

VectorModel() = VectorModel(Dict{String,Int}(), Dict{Int,Float64}(), 0, 1, 1.0, TextConfig())

function VectorModel(config::TextConfig)
    model = VectorModel()
    model.config = config
    model
end

function fit!(model::VectorModel, corpus; get_text::Function=identity)
    V = Dict{String,Int}()
    for item in corpus
        text = get_text(item)
        voc = compute_bow(text, model.config)
        for (token, freq) in voc
            V[token] = get(V, token, 0) + freq
        end
        
        model.size += 1
        if model.size % 10000 == 1
            info("advance VectorModel: $(model.size) processed items")
        end
    end

    info("finished VectorModel: $(model.size) processed items")

    for (token, freq) in V
        if freq < model.filter_low || freq > model.filter_high * model.size
            continue
        end

        id = length(model.token2id) + 1
        model.token2id[token] = id
        model.weights[id] = freq
    end

   # model.weights[0] = model.filter_low + 1  # for unknown tokens
end

function compute_bow(text::String, config::TextConfig)
    voc = Dict{String,Int}()

    for token in tokenize(text, config)
        freq = get(voc, token, 0) + 1
        voc[token] = freq
    end

    voc
end

function raw_vbow(idtokens, model::VectorModel)
    b = WeightedToken[WeightedToken(idtokens[1], 1.0)]
    sizehint!(b, length(idtokens))
    for i in 2:length(idtokens)
        if idtokens[i-1] == idtokens[i]
            b[end].weight += 1
        else
            push!(b, WeightedToken(idtokens[i], 1.0))
        end
    end
    
    maxfreq_ = 0.0
    for t in b
        if maxfreq_ < t.weight
            maxfreq_ = t.weight
        end
    end
    
    b, maxfreq_
end

function weighted_vector(weighting_function::Function, text::String, model::VectorModel; maxlength=typemax(Int))::VBOW
    # bow = compute_bow(text, model.config)
    tokens = tokenize(text, model.config)
    if length(tokens) == 0
        return VBOW(WeightedToken[])
    end

    idtokens = Int[] # [get(model.token2id, t, 0) for t in tokens]
    sizehint!(idtokens, length(tokens))
    for t in tokens
        id = get(model.token2id, t, 0)
        if id > 0
            push!(idtokens, id)
        end
    end
    if length(idtokens) == 0
        return VBOW(WeightedToken[])
    end
    sort!(idtokens)

    vbow, maxfreq = raw_vbow(idtokens, model)

    for t in vbow
        t.weight = weighting_function(t.weight, maxfreq, model.size, model.weights[t.id])
    end

    if length(vbow) > maxlength
        sort!(vbow, by=(x) -> -x.weight)
        vbow = vbow[1:maxlength]
        sort!(vbow, by=(x) -> x.id)
    end

    VBOW(vbow)
end

function _tfidf(freq, maxfreq, N, freqToken)
    (freq / maxfreq) * log(N / freqToken)
end

function _tf(freq, maxfreq, N, freqToken)
    (freq / maxfreq)
end

function _idf(freq, maxfreq, N, freqToken)
    log(N / freqToken)
end

function _rawfreq(freq, maxfreq, N, freqToken)
    freq
end

function vectorize_tfidf(text::String, model::VectorModel; maxlength=typemax(Int))
    weighted_vector(_tfidf, text, model, maxlength=maxlength)
end

function vectorize_tf(text::String, model::VectorModel; maxlength=typemax(Int))
    weighted_vector(_tf, text, model, maxlength=maxlength)
end

function vectorize_idf(text::String, model::VectorModel; maxlength=typemax(Int))
    weighted_vector(_idf, text, model, maxlength=maxlength)
end

function vectorize_rawfreq(text::String, model::VectorModel; maxlength=typemax(Int))
    weighted_vector(_rawfreq, text, model, maxlength=maxlength)
end

function vectorize(text::String, model::VectorModel; maxlength=typemax(Int))
    vectorize_tfidf(text, model, maxlength=maxlength)
end

function vectorize(textlist::AbstractVector{String}, model::Model)
    [vectorize(text, model) for text in textlist]
end
