export TextModel, VectorModel, fit!, inverse_vbow,
    compute_bow, vectorize, id2token, TfidfModel, TfModel, IdfModel, FreqModel

abstract type Model end
	
mutable struct VectorModel <: Model
    token2id::Dict{String,Int}
    weights::Dict{Int,Float64}
    size::Int64
    filter_low::Int
    filter_high::Float64
    config::TextConfig
end

struct TfidfModel <: Model
    vmodel::VectorModel
end

struct TfModel <: Model
    vmodel::VectorModel
end

struct IdfModel <: Model
    vmodel::VectorModel
end

struct FreqModel <: Model
    vmodel::VectorModel
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

function fit!(model::VectorModel, corpus)
    V = Dict{String,Int}()
    for data in corpus
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

function vectorize(data, model::T)::VBOW where {T <: Union{TfidfModel,TfModel,IdfModel,FreqModel}}
	raw = compute_bow(data, model.vmodel)
	bow = VBOW(raw)
	m = maxfreq(raw)
	for t in bow.tokens
        t.weight = _weight(model, t.weight, m, model.vmodel.size, model.vmodel.weights[t.id])
    end
	
	bow.invnorm = -1.0
	bow
end

function _weight(model::TfidfModel, freq, maxfreq, N, freqToken)::Float64
    (freq / maxfreq) * log(N / freqToken)
end

function _weight(model::TfModel, freq, maxfreq, N, freqToken)::Float64
    (freq / maxfreq)
end

function _weight(model::IdfModel, freq, maxfreq, N, freqToken)::Float64
    log(N / freqToken)
end

function _weight(model::FreqModel, freq, maxfreq, N, freqToken)::Float64
    freq
end
