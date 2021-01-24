# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextModel, VectorModel, TfidfWeighting, TfWeighting, IdfWeighting, FreqWeighting, fit, vectorize, prune, prune_select_top

abstract type TfidfWeighting <: WeightingType end
abstract type TfWeighting <: WeightingType end
abstract type IdfWeighting <: WeightingType end
abstract type FreqWeighting <: WeightingType end

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
mutable struct VectorModel{W<:WeightingType} <: TextModel
    weighting::Type{W}
    tokens::Vocabulary
    id2token::Dict{Int,Symbol}
    maxfreq::Int
    m::Int  # vocsize
    n::Int  # collection size
end

Base.copy(e::VectorModel; weighting=e.weighting, tokens=e.tokens, id2token=e.id2token, maxfreq=e.maxfreq, m=e.m, n=e.n) =
    VectorModel(weighting, tokens, id2token, maxfreq, m, n)

"""
    VectorModel(weighting::Type{W}, corpus::BOW; minocc::Integer=1) where {W<:WeightingType}

Trains a vector model using the input corpus. 
"""
function VectorModel(weighting::Type{W}, corpus::BOW; minocc::Integer=1) where {W<:WeightingType}
    tokens = Vocabulary()
    id2token = Dict{Int,Symbol}()
    i = 0
    maxfreq = 0
    for (t, freq) in corpus
		freq < minocc && continue
        i += 1
        id2token[i] = t
        tokens[t] = IdFreq(i, freq)
        maxfreq = max(maxfreq, freq)
    end

    VectorModel(weighting, tokens, id2token, Int(maxfreq), length(tokens), length(corpus))
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
    VectorModel(model.weighting, tokens, id2token, model.maxfreq, model.m, model.n)
end

"""
    prune_select_top(model::VectorModel, k::Integer, kind::Type{T}=IdfWeighting)
    prune_select_top(model::VectorModel, ratio::AbstractFloat, kind::Type{T}=IdfWeighting)

Creates a new model with the best `k` tokens from `model` based on the `kind` scheme; kind must be either `IdfWeighting` or `FreqWeighting`.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer, kind::Type{T}=IdfWeighting) where T <: Union{IdfWeighting,FreqWeighting}
    tokens = Vocabulary()
    maxfreq = 0
    if kind == IdfWeighting
        X = [(t, idfreq, _weight(kind, 0, 0, model.n, idfreq.freq)) for (t, idfreq) in model.tokens]
        sort!(X, by=x->x[end], rev=true)
        for i in 1:k
            t, idfreq, w = X[i]
            tokens[t] = idfreq
            maxfreq = max(maxfreq, idfreq.freq)
        end

    else kind == FreqWeighting
        X = [(t, idfreq) for (t, idfreq) in model.tokens]
        sort!(X, by=x->x[end].freq, rev=true)
        for i in 1:k
            t, idfreq = X[i]
            tokens[t] = idfreq
            maxfreq = max(maxfreq, idfreq.freq)
        end
    end

    id2token = Dict(idfreq.id => t for (t, idfreq) in tokens)
    VectorModel(model.weigthing, tokens, id2token, maxfreq, model.m, model.n)
end

prune_select_top(model::VectorModel, ratio::AbstractFloat, kind=IdfWeighting) = prune_select_top(model, floor(Int, length(model.tokens) * ratio), kind)

"""
    vectorize(model::VectorModel, bow::BOW, maxfreq::Integer=maximum(bow); normalize=true) where Tv<:Real

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize(model::VectorModel, bow::BOW, maxfreq::Integer=maximum(bow); normalize=true)
    vec = SVEC()
    for (token, freq) in bow
        t = get(model.tokens, token, nothing)
        t === nothing && continue

        w = _weight(model.weighting, freq, maxfreq, model.n, t.freq)
        if w > 1e-6
            vec[t.id] = w
        end
    end
    
    normalize && normalize!(vec)
    vec
end

function broadcastable(model::VectorModel)
    (model,)
end

"""
    _weight(::Type{T}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64

Computes a weight for the given stats using scheme T
"""
function _weight(::Type{TfidfWeighting}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    (freq / maxfreq) * log(2, 1 + n / global_freq)
end

function _weight(::Type{TfWeighting}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    freq / maxfreq
end

function _weight(::Type{IdfWeighting}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    log(2, n / global_freq)
end

function _weight(::Type{FreqWeighting}, freq::Real, maxfreq::Real, n::Real, global_freq::Real)::Float64
    freq
end
