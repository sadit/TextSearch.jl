import SimilaritySearch: fit
export TextModel, VectorModel, fit, inverse_vbow, vectorize, id2token, TfidfModel, TfModel, IdfModel, FreqModel

abstract type Model end

mutable struct VectorModel <: Model
    config::TextConfig
    vocab::Dict{Symbol,TokenData}
    maxfreq::Int32
    n::Int32
end

abstract type TfidfModel end
abstract type TfModel end
abstract type IdfModel end
abstract type FreqModel end

function id2token(model::VectorModel)
    m = Dict{Int,Symbol}()
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

function maximum_(vocab::Dict{Symbol,TokenData})
    m = 0
    for (token, f) in vocab
        m = max(m, f.freq)
    end
    
    m
end

function filter_vocab(vocab, low, high=0)
    X = Dict{String,TokenData}()
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
    voc = Dict{Symbol,TokenData}()
    n = 1
 
    for data in corpus
        compute_dict_bow(config, data, voc)
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

function vectorize(model::VectorModel, weighting::Type, data)::SparseVector
    bag = compute_dict_bow(model.config, data, Dict{Symbol,TokenData}())
	maxfreq = maximum_(bag)
    b = Vector{SparseVectorEntry}(undef, length(bag))

    i = 0    
    for (token, tokendata) in bag
        global_tokendata = get(model.vocab, token, UNKNOWN_TOKEN)
        if global_tokendata.freq == 0
            continue
        end

        w = _weight(weighting, tokendata.freq, maxfreq, model.n, global_tokendata.freq)
        i += 1
        b[i] = SparseVectorEntry(global_tokendata.id, w)
     end

    resize!(b, i)
    SparseVector(b)
end

function weighted_bow(model::VectorModel, weithing::Type, data)
    bag = compute_dict_bow(model.config, data, Dict{Symbol,TokenData}())
	maxfreq = maximum_(bag)
    L = Vector{NamedTuple{(:token, weight), Tuple{Symbol,Float64}}}(undef, length(bag))
    i = 0    
    for (token, idtoken) in bag
        global_idtoken = get(model.vocab, token, UNKNOWN_TOKEN)
        if global_idtoken.freq == 0
            continue
        end

        w = _weight(weighting, idtoken.freq, maxfreq, model.n, global_idtoken.freq)
        i += 1
        b[i] = SparseVectorEntry(global_idtoken.id, w)
     end

    resize!(b, i)
    SparseVector(b)
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
