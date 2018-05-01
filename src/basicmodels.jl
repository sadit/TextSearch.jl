export TextModel, VectorModel, fit!, inverse_vbow,
    compute_bow, vectorize, id2token, TfidfModel, TfModel, IdfModel, FreqModel

abstract type Model end
	
mutable struct VectorModel <: Model
    W::Dict{String,WeightedToken}
    #W::Dict{String,Int}
    #weights::Dict{Int,Float64}
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
    for (t, wtoken) in model.W
        m[wtoken.id] = t
    end

    m
end

function inverse_vbow(vec, vocmap)
    s = collect(vec.tokens)
    sort!(s, by=x -> -x.weight)
    [(vocmap[token.id], token.weight) for token in s]
end

VectorModel() = VectorModel(Dict{String,Float64}(), 0, 1, 1.0, TextConfig())

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

        id = length(model.W) + 1
        model.W[token] = WeightedToken(id, freq)
    end
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
    bag = compute_bow(data, model.vmodel.config)
	m = maxfreq(bag)
    n = model.vmodel.size
    b = WeightedToken[]
    sizehint!(b, length(bag))
	for (token, freq) in bag
        wtoken = try
            model.vmodel.W[token]
        catch KeyError
            continue
        end
        w = _weight(model, freq, m, n, wtoken.weight)
        push!(b, WeightedToken(wtoken.id, w))
    end
    
	VBOW(b)
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
