# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextModel, VectorModel, fit, vectorize, prune, prune_select_top

abstract type TfidfModel <: WeightingType end
abstract type TfModel <: WeightingType end
abstract type IdfModel <: WeightingType end
abstract type FreqModel <: WeightingType end

"""
    abstract type Model

An abstract type that represents a weighting model
"""
abstract type TextModel end

struct IdFreq
    id::Int32
    freq::Int32
end

const Vocabulary = Dict{Symbol, IdFreq}

"""
    mutable struct VectorModel

Models a text through a vector space
"""
mutable struct VectorModel <: TextModel
    config::TextConfig
    tokens::Vocabulary
    id2token::Dict{Int,Symbol}
    maxfreq::Int
    m::Int  # vocsize
    n::Int  # collection size
end

## function corpus_bow(config, corpus; batch_size=128)
##     m = nworkers()
##     n = length(corpus)
## 
##     L = []
##     for _corpus in Iterators.partition(corpus, batch_size)
##         b = @spawn begin
##             bow = BOW()
##             for text in _corpus
##                 compute_bow(tokenize(config, text), bow)
##             end
##             bow
##         end
##         push!(L, b)
##     end
## 
##     sum(fetch.(L))
## end

function corpus_bow(config::TextConfig, corpus::AbstractVector)
    bow = BOW()
    for text in corpus
        compute_bow(tokenize(config, text), bow)
    end

    bow
end

"""
    VectorModel(config::TextConfig, corpus::AbstractVector)

Trains a vector model using the text preprocessing configuration `config` and the input corpus. 
"""
function VectorModel(config::TextConfig, corpus::AbstractVector; minocc::Integer=1)
    bow = corpus_bow(config, corpus)
    tokens = Vocabulary()
    id2token = Dict{Int,Symbol}()
    i = 0
    maxfreq = 0
    for (t, freq) in bow
		freq < minocc && continue
        i += 1
        id2token[i] = t
        tokens[t] = IdFreq(i, freq)
        maxfreq = max(maxfreq, freq)
    end

    VectorModel(config, tokens, id2token, Int(maxfreq), length(tokens), length(corpus))
end

"""
    prune(model::VectorModel, minfreq, rank)

Cuts the vocabulary by frequency using lower and higher filter;
All tokens with frequency below `minfreq` are ignored; top `rank` tokens are also removed.
"""
function prune(model::VectorModel, minfreq::Integer, rank::Integer)
    W = [(token, idfreq) for (token, idfreq) in model.tokens if idfreq.freq >= minfreq]
    sort!(W, by=x->x[2])
    tokens = Vocabulary()
    for i in 1:length(W)-rank+1
        w = W[i]
        tokens[w[1]] = w[2]
    end

    id2token = Dict(idfreq.id => t for (t, idfreq) in tokens)
    VectorModel(model.config, tokens, id2token, model.maxfreq, model.m, model.n)
end

"""
    prune_select_top(model::VectorModel, k::Integer, kind::Type{T}=IdfModel)
    prune_select_top(model::VectorModel, ratio::AbstractFloat, kind::Type{T}=IdfModel)

Creates a new model with the best `k` tokens from `model` based on the `kind` scheme; kind must be either `IdfModel` or `FreqModel`.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer, kind::Type{T}=IdfModel) where T <: Union{IdfModel,FreqModel}
    tokens = Vocabulary()
    maxfreq = 0
    if kind == IdfModel
        X = [(t, idfreq, _weight(kind, 0, 0, model.n, idfreq.freq)) for (t, idfreq) in model.tokens]
        sort!(X, by=x->x[end], rev=true)
        for i in 1:k
            t, idfreq, w = X[i]
            tokens[t] = idfreq
            maxfreq = max(maxfreq, idfreq.freq)
        end

    else kind == FreqModel
        X = [(t, idfreq) for (t, idfreq) in model.tokens]
        sort!(X, by=x->x[end].freq, rev=true)
        for i in 1:k
            t, idfreq = X[i]
            tokens[t] = idfreq
            maxfreq = max(maxfreq, idfreq.freq)
        end
    end

    id2token = Dict(idfreq.id => t for (t, idfreq) in tokens)
    VectorModel(model.config, tokens, id2token, maxfreq, model.m, model.n)
end

prune_select_top(model::VectorModel, ratio::AbstractFloat, kind=IdfModel) = prune_select_top(model, floor(Int, length(model.tokens) * ratio), kind)

"""
    vectorize(model::VectorModel, weighting::Type, data; normalize=true)::Dict{Int, Float64}

Computes `data`'s weighted bow of words using the given model and weighting scheme;
the vector is normalized to the unit normed vector if normalize is true
"""
function vectorize(model::VectorModel, weighting::Type, data::DataType; normalize=true) where DataType <: Union{AbstractString, AbstractVector{S}} where S <: AbstractString
    bow = compute_bow(tokenize(model.config, data))
    vectorize(model, weighting, bow, maximum(bow), normalize=normalize)
end

"""
    vectorize(model::VectorModel, weighting::Type, bow::BOW, maxfreq=0; normalize=true) where Tv<:Real

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize(model::VectorModel, weighting::Type, bow::BOW, maxfreq=0; normalize=true)
    if maxfreq == 0
        for v in values(bow)
            maxfreq = max(maxfreq, v)
        end
    end

    vec = SVEC()
    for (token, freq) in bow
        t = get(model.tokens, token, nothing)

        if t === nothing
            continue
        end

        w = _weight(weighting, freq, maxfreq, model.n, t.freq)

        if w > 1e-6
            vec[t.id] = w
        end
    end
    
    normalize && normalize!(vec)
    vec
end

"""
    vectorize(model::VectorModel, data; normalize=true)

Computes the vector of data using TfidfModel as default
"""
vectorize(model::VectorModel, data; normalize=true) = vectorize(model, TfidfModel, data, normalize=normalize)

function broadcastable(model::VectorModel)
    (model,)
end

"""
    _weight(::Type{T}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64

Computes a weight for the given stats using scheme T
"""
function _weight(::Type{TfidfModel}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    (freq / maxfreq) * log(2, 1 + n / global_freq)
end

function _weight(::Type{TfModel}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    freq / maxfreq
end

function _weight(::Type{IdfModel}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    log(2, n / global_freq)
end

function _weight(::Type{FreqModel}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    freq
end
