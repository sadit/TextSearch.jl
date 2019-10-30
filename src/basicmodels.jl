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

struct IdFreq
    id::Int
    freq::Int
end

const Vocabulary = Dict{Symbol, IdFreq}

"""
    mutable struct VectorModel

Models a text through a vector space
"""
mutable struct VectorModel <: Model
    config::TextConfig
    tokens::Vocabulary
    id2token::Dict{Int,Symbol}
    # id2token::Dict{Int,Symbol}
    maxfreq::Int
    m::Int  # vocsize
    n::Int  # collection size
end

"""
    fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector)

Trains a vector model using the text preprocessing configuration `config` and the input corpus. 
"""
function fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector)
    bow = BOW()
    n = 0
    maxfreq = 0.0
    println(stderr, "fitting VectorModel with $(length(corpus)) items")

    for data in corpus
        n += 1
        _, _maxfreq = compute_bow(tokenize(config, data), bow)
        maxfreq = max(maxfreq, _maxfreq)
        n % 1000 == 0 && print(stderr, "x")
        n % 100000 == 0 && println(stderr, " $(n/length(corpus))")
    end

    tokens = Vocabulary()
    id2token = Dict{Int,Symbol}()
    i = 0
    for (t, freq) in bow
        i += 1
        id2token[i] = t
        tokens[t] = IdFreq(i, freq)
    end

    println(stderr, "finished VectorModel: $n processed items, voc-size: $(length(bow))")
    VectorModel(config, tokens, id2token, Int(maxfreq), length(tokens), n)
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
    update!(a::VectorModel, b::VectorModel)

Updates `a` with `b` inplace; returns `a`.
"""
function update!(a::VectorModel, b::VectorModel)
    m = a.m
    for (token, idfreq) in b.tokens
        x = get(a.tokens, token, nothing)
        if x === nothing
            m += 1
            a.tokens[token] = IdFreq(m, x.freq)
        else
            a.tokens[token] = IdFreq(idfreq.id, idfreq.freq + x.freq)
        end
    end

    a.maxfreq = max(a.maxfreq, b.maxfreq)
    a.n += b.n
    a.m = m
    a.id2token = Dict(idfreq.id => t for (t, idfreq) in a.tokens)
    a
end

"""
    vectorize(model::VectorModel, weighting::Type, data; normalize=true)::Dict{Symbol, Float64}

Computes `data`'s weighted bag of words using the given model and weighting scheme;
the vector is normalized to the unit normed vector if normalize is true
"""
function vectorize(model::VectorModel, weighting::Type, data::DataType; normalize=true) where DataType <: Union{AbstractString, AbstractVector{S}} where S <: AbstractString
    bag, maxfreq = compute_bow(tokenize(model.config, data))
    vectorize(model, weighting, bag, maxfreq, normalize=normalize)
end

"""
    vectorize(model::VectorModel, weighting::Type, bow::BOW, maxfreq=0; normalize=true)

Computes a weighted vector using the given bag of words and the specified weighting scheme.
The result is computed on the input bow (replacing or removing entries as needed).
"""
function vectorize(model::VectorModel, weighting::Type, bow::BOW, maxfreq=0; normalize=true)
    if maxfreq == 0
        for v in values(bow)
            maxfreq = max(maxfreq, v)
        end
    end

    I = Int[]
    F = Float64[]

    for (token, freq) in bow
        t = get(model.tokens, token, nothing)

        if t === nothing
            continue
        end

        w = _weight(weighting, freq, maxfreq, model.n, t.freq)

        if w > 1e-6
            push!(I, t.id)
            push!(F, w)
        end
    end
    
    vec = sparsevec(I, F, model.m)
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
