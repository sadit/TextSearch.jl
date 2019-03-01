import SimilaritySearch: fit
export TextModel, VectorModel, fit, inverse_vbow,
    compute_bow, vectorize, id2token, IdFreq, TfidfModel, TfModel, IdfModel, FreqModel

abstract type Model end

mutable struct IdFreq
    id::Int32
    freq::Int32
    IdFreq() = new(0, 0)
    IdFreq(a, b) = new(a, b)
end

mutable struct VectorModel <: Model
    config::TextConfig
    vocab::Dict{String,IdFreq}
    maxfreq::Int32
    n::Int32
end

abstract type TfidfModel end
abstract type TfModel end
abstract type IdfModel end
abstract type FreqModel end

function id2token(model::VectorModel)
    m = Dict{Int,String}()
    for (t, f) in model.vocab
        m[f.id] = t
    end

    m
end

function inverse_vbow(vec, vocmap)
    s = collect(vec.tokens)
    sort!(s, by=x -> -x.weight)
    [(vocmap[token.id], token.weight) for token in s]
end

function maximum_(vocab::Dict{String,IdFreq})
    m = 0
    for (token, f) in vocab
        m = max(m, f.freq)
    end
    
    m
end

function filter_vocab(vocab, low, high=0)
    X = Dict{String,IdFreq}()
    maxfreq = maximum_(vocab)
    for (t, w) in vocab
        if w.freq < low || w.freq > maxfreq - high
            continue
        end

        X[t] = w
    end

    X, maxfreq
end

function fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector; low=0, high=0) where {T <: Union{TfidfModel,TfModel,IdfModel,FreqModel}}
    voc = Dict{String,IdFreq}()
    n = 1
 
    for data in corpus
        compute_bow(config, data, voc)
        n += 1
        if n % 10000 == 1
            @info "advance VectorModel: $n processed items"
        end
    end

    @info "finished VectorModel: $n processed items"
    if low != 0 || high != 0
        voc, maxfreq = filter_vocab(voc, low, high)
    else
        maxfreq = maximum_(voc)
    end

    VectorModel(config, voc, maxfreq, n)
end

const unknown_token = IdFreq(0, 0)


function compute_bow(config::TextConfig, text::String)
    X = compute_bow(config, text, Dict{String,IdFreq}())
    X = [(token, idfreq.freq) for (token, idfreq) in X]
    sort!(X, by=x->x[1])
    X
end

function compute_bow(config::TextConfig, text::String, voc::Dict{String,IdFreq})
    for token in tokenize(config, text)
        h = get(voc, token, unknown_token)
        if h.freq == 0
            voc[token] = IdFreq(length(voc), 1)
        else
            h.freq += 1
        end
    end

    voc
end

function compute_bow(config::TextConfig, arr::AbstractVector{String}, voc::Dict{String,IdFreq})
	for text in arr
		compute_bow(config, text, voc)
	end
	
	voc
end

function vectorize(model::VectorModel, weighting::Type, data)::VBOW
    bag = compute_bow(model.config, data, Dict{String,IdFreq}())
	maxfreq = maximum_(bag)
    b = Vector{WeightedToken}(undef, length(bag))

    i = 0    
    for (token, idtoken) in bag
        global_idtoken = get(model.vocab, token, unknown_token)
        if global_idtoken.freq == 0
            continue
        end

        w = _weight(weighting, idtoken.freq, maxfreq, model.n, global_idtoken.freq)
        i += 1
        b[i] = WeightedToken(global_idtoken.id, w)
     end

    resize!(b, i)
    VBOW(b)
end

function _weight(::Type{TfidfModel}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64
    (freq / maxfreq) * log(2, 1 + n / global_freq)
end

function _weight(::Type{TfModel}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64
    (freq / maxfreq)
end

function _weight(::Type{IdfModel}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64
    log(2, n / global_freq)
end

function _weight(::Type{FreqModel}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64
    freq
end
