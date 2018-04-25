export TextModel, VectorModel, fit!, inverse_vbow,
    compute_bow, vectorize, id2token, Tfidf, Tf, Idf, Freq

abstract type Model end
abstract type Weighting end

abstract type Tfidf <: Weighting end
abstract type Tf <: Weighting end
abstract type Idf <: Weighting end
abstract type Freq <: Weighting end 
	
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
        data = get_text(item)
        voc = compute_bow(data, model.config)
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

function compute_bow(text::String, config::TextConfig, voc=nothing)
	if voc == nothing
		voc = Dict{String,Int}()
	end
	
    for token in tokenize(text, config)
        freq = get(voc, token, 0) + 1
        voc[token] = freq
    end

    voc
end

function compute_bow(arr, config::TextConfig)
	v = Dict{String,Int}()
	
	for text in arr
		compute_bow(text::String, config, v)
	end
	
	v
end

function compute_bow(text::String, model::VectorModel, voc=nothing)
	if voc == nothing
		voc = Dict{Int,Int}()
	end
	
    for token in tokenize(text, model.config)
		i = get(model.token2id, token, 0)
		if i > 0
			voc[i] = get(voc, i, 0) + 1
		end
    end

    voc
end

function compute_bow(arr, model::VectorModel)
	v = Dict{Int,Int}()
	
	for text in arr
		compute_bow(text::String, model, v)
	end
	
	v
end

function maxfreq(vow)::Int
	m = 0
	@inbounds for v in values(vow)
		if v > m
			m = v
		end
		# m = max(m, v)
	end
	
	m
end

function weighted_vector(weighting::Type, data, model::Model; maxlength=typemax(Int))::VBOW
	raw = compute_bow(data, model)
	bow = VBOW(raw)
	m = maxfreq(raw)
	for t in bow.tokens
        t.weight = _weight(weighting, t.weight, m, model.size, model.weights[t.id])
    end
	
	bow.invnorm = -1.0

    if length(bow) > maxlength
        sort!(bow.tokens, by=(x) -> -x.weight)
        bow = bow[1:maxlength]
        sort!(bow.tokens, by=(x) -> x.id)
    end

	bow
end

function _weight(::Type{Tfidf}, freq, maxfreq, N, freqToken)::Float64
    (freq / maxfreq) * log(N / freqToken)
end

function _weight(::Type{Tf}, freq, maxfreq, N, freqToken)::Float64
    (freq / maxfreq)
end

function _weight(::Type{Idf}, freq, maxfreq, N, freqToken)::Float64
    log(N / freqToken)
end

function _weight(::Type{Freq}, freq, maxfreq, N, freqToken)::Float64
    freq
end

#function vectorize(weighting::Weighting, data::String, model::VectorModel; maxlength=typemax(Int))
#	weighted_vector(weighting, data, model, maxlength=maxlength)
#end

function vectorize(weighting::Weighting, data, model::Model; maxlength=typemax(Int))
	weighted_vector(weighting, data, model, maxlength=maxlength)
end

#function vectorize(data::String, model::VectorModel; maxlength=typemax(Int))
#	weighted_vector(Tfidf, data, model, maxlength=maxlength)
#end

function vectorize(data, model::Model; maxlength=typemax(Int))
	weighted_vector(Tfidf, data, model, maxlength=maxlength)
end