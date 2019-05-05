import SimilaritySearch:
    fit
export VectorModel, fit, inverse_vbow, vectorize, weighted_bow, id2token, TfidfModel, TfModel, IdfModel, FreqModel

"""
    abstract type Model

An abstract type that represents a weighting model
"""
abstract type Model end

"""
    mutable struct VectorModel

Models a text through a vector space
"""
mutable struct VectorModel <: Model
    config::TextConfig
    vocab::Dict{Symbol,TokenData}
    maxfreq::Int32
    n::Int32
end

"""
    update!(a::VectorModel, b::VectorModel)

Updates `a` with `b` inplace; returns `a`. TokenData's id is solved
consistently, but can destroy any previous info.
"""
function update!(a::VectorModel, b::VectorModel)
    i = 0
    for (k, v) in b.vocab
        i += 1
        t = get(a, k, v)
        if v == t
            a[k] = v
        else
            a[k] = TokenData(i, t.freq + v.freq)
        end
    end

    a.maxfreq = max(a.maxfreq, b.maxfreq)
    a.n += b.n
    a
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

"""
    filter_vocab(vocab, maxfreq, lower::int, higher::Float64=1.0)

Drops terms in the vocabulary with less than `low` and and higher than `high` frequences.
- `lower` is specified as an integer, and must be read as the lower accepted frequency (lower frequencies will be dropped)
- `higher` is specified as a float, 0 < higher <= 1.0; it is readed as the higher frequency that is preserved (it a proportion of the maximum frequency)

"""
function filter_vocab(vocab, maxfreq, lower::Int, higher::Float64=1.0)
    X = Dict{Symbol,TokenData}()
    # maxfreq = maximum_(vocab)
    for (t, w) in vocab
        if w.freq < lower || w.freq > maxfreq * higher
            continue
        end

        X[t] = w
    end

    X, floor(Int, maxfreq * higher)
end

"""
    fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector; lower=0, higher=1.0)

Trains a vector model using the text preprocessing configuration `config` and the input corpus. It also allows for filtering
tokens with low and high number of occurrences. `lower` is specified as an integer and `higher` as a proportion between
the frequency of the current token and the maximum frequency of the model.
"""
function fit(::Type{VectorModel}, config::TextConfig, corpus::AbstractVector; lower=0, higher=1.0)
    voc = Dict{Symbol,TokenData}()
    n = 0
    maxfreq = 0

    for data in corpus
        n += 1
        _, maxfreq = compute_vocabulary(config, data, voc)
        if n % 10000 == 1
            @info "advance VectorModel: $n processed items"
        end
    end

    @info "finished VectorModel: $n processed items"
    if lower != 0 || higher != 1.0
        voc, maxfreq = filter_vocab(voc, maxfreq, lower, higher)
    end

    VectorModel(config, voc, maxfreq, n)
end

function vectorize(model::VectorModel, weighting::Type, data, modify_bow!::Function=identity)::SparseVector
    bag, maxfreq = compute_bow(model.config, data)
    b = Vector{SparseVectorEntry}(undef, length(bag))
    i = 0
    bag = modify_bow!(bag)
    for (token, freq) in bag
        global_tokendata = get(model.vocab, token, UNKNOWN_TOKEN)
        if global_tokendata.freq == 0
            continue
        end

        w = _weight(weighting, freq, maxfreq, model.n, global_tokendata.freq)
        i += 1
        b[i] = SparseVectorEntry(global_tokendata.id, w)
    end

    resize!(b, i)
    SparseVector(b)
end

"""
    weighted_bow(model::VectorModel, weighting::Type, data, modify_bow!::Function=identity)::Dict{Symbol, Float64}

Computes `data`'s weighted bag of words using the given model and weighting scheme.
It takes a function `modify_bow!` (that defaults to `identity`) to modify the bag
before applying the weighting scheme.
"""
function weighted_bow(model::VectorModel, weighting::Type, data, modify_bow!::Function=identity)::Dict{Symbol, Float64}
    W = Dict{Symbol, Float64}()
    bag, maxfreq = compute_bow(model.config, data)
    bag = modify_bow!(bag)
    for (token, freq) in bag
        global_tokendata = get(model.vocab, token, UNKNOWN_TOKEN)
        if global_tokendata.freq == 0
            continue
        end

        w = _weight(weighting, freq, maxfreq, model.n, global_tokendata.freq)
        W[token] = w
    end
  
    W
end

"""
    _weight(::Type{T}, freq::Integer, maxfreq::Integer, n::Integer, global_freq::Integer)::Float64

Computes a weight for the given stats using scheme T
"""
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
