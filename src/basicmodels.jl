export VectorModel, fit, vectorize, TfidfModel, TfModel, IdfModel, FreqModel, prune, prune_select_top

"""
    abstract type Model

An abstract type that represents a weighting model
"""
abstract type Model end

abstract type TfidfModel end
abstract type TfModel end
abstract type IdfModel end
abstract type FreqModel end

"""
    mutable struct VectorModel

Models a text through a vector space
"""
mutable struct VectorModel <: Model
    config::TextConfig
    tokens::BOW
    maxfreq::Int
    n::Int
end

"""
    fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector)

Trains a vector model using the text preprocessing configuration `config` and the input corpus. 
"""
function fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector)
    voc = BOW()
    n = 0
    maxfreq = 0.0
    println(stderr, "fitting VectorModel with $(length(corpus)) items")

    for data in corpus
        n += 1
        _, _maxfreq = compute_bow(tokenize(config, data), voc)
        maxfreq = max(maxfreq, _maxfreq)
        n % 1000 == 0 && print(stderr, "x")
        n % 100000 == 0 && println(stderr, " $(n/length(corpus))")
    end

    println(stderr, "finished VectorModel: $n processed items")
    VectorModel(config, voc, Int(maxfreq), n)
end

"""
    prune(model::VectorModel, minfreq, rank)

Cuts the vocabulary by frequency using lower and higher filter;
All tokens with frequency below `freq` are ignored; top `rank` tokens are also removed.
"""
function prune(model::VectorModel, freq::Integer, rank::Integer)
    W = [token => f for (token, f) in model.tokens if f >= freq]
    sort!(W, by=x->x[2])
    M = BOW()
    for i in 1:length(W)-rank+1
        w = W[i]
        M[w[1]] = w[2]
    end

    VectorModel(model.config, M, model.maxfreq, model.n)
end

"""
    prune_select_top(model::VectorModel, k::Integer, kind::Type{T}=IdfModel)
    prune_select_top(model::VectorModel, ratio::AbstractFloat, kind::Type{T}=IdfModel)

Creates a new model with the best `k` tokens from `model` based on the `kind` scheme; kind must be either `IdfModel` or `FreqModel`.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer, kind::Type{T}=IdfModel) where T <: Union{IdfModel,FreqModel}
    tokens = BOW()
    maxfreq = 0
    if kind == IdfModel
        X = [(t, freq, _weight(kind, 0, 0, model.n, freq)) for (t, freq) in model.tokens]
        sort!(X, by=x->x[end], rev=true)
        for i in 1:k
            t, freq, w = X[i]
            tokens[t] = freq
            maxfreq = max(maxfreq, freq)
        end

    else kind == FreqModel
        X = [(t, freq) for (t, freq) in model.tokens]
        sort!(X, by=x->x[end], rev=true)
        for i in 1:k
            t, freq = X[i]
            tokens[t] = freq
            maxfreq = max(maxfreq, freq)
        end
    end

    VectorModel(model.config, tokens, maxfreq, model.n)
end

prune_select_top(model::VectorModel, ratio::AbstractFloat, kind=IdfModel) = prune_select_top(model, floor(Int, length(model.tokens) * ratio), kind)

"""
    update!(a::VectorModel, b::VectorModel)

Updates `a` with `b` inplace; returns `a`.
"""
function update!(a::VectorModel, b::VectorModel)
    i = 0
    for (k, freq1) in b.tokens
        i += 1
        freq2 = get(a, k, 0.0)
        if freq1 == 0.0
            a[k] = freq1
        else
            a[k] = freq1 + freq2
        end
    end

    a.maxfreq = max(a.maxfreq, b.maxfreq)
    a.n += b.n
    a
end


"""
    vectorize(model::VectorModel, weighting::Type, data; normalize=true)::Dict{Symbol, Float64}

Computes `data`'s weighted bag of words using the given model and weighting scheme;
the vector is normalized to the unit normed vector if normalize is true
"""
function vectorize(model::VectorModel, weighting::Type, data::DataType; normalize=true)::BOW where DataType <: Union{AbstractString, AbstractVector{S}} where S <: AbstractString
    bag, maxfreq = compute_bow(tokenize(model.config, data))
    vectorize(model, weighting, bag, maxfreq, normalize=normalize)
end

"""
    vectorize(model::VectorModel, weighting::Type, bow::BOW, maxfreq=0; normalize=true)::BOW

Computes a weighted vector using the given bag of words and the specified weighting scheme.
The result is computed on the input bow (replacing or removing entries as needed).
"""
function vectorize(model::VectorModel, weighting::Type, bow::BOW, maxfreq=0; normalize=true)::BOW
    if maxfreq == 0
        for v in values(bow)
            maxfreq = max(maxfreq, v)
        end
    end

    for (token, freq) in bow
        global_freq = get(model.tokens, token, 0.0)
        w = 0.0
        if global_freq > 0.0
            w = _weight(weighting, freq, maxfreq, model.n, global_freq)
        end

        if w <= 1e-6
            delete!(bow, token)
        else
            bow[token] = w
        end
    end
    
    normalize && normalize!(bow)
    bow
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
