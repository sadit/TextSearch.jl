# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextModel, VectorModel, TfWeighting, IdfWeighting, TpWeighting,
    FreqWeighting, BinaryLocalWeighting, BinaryGlobalWeighting, fit, vectorize, prune, prune_select_top

#####
##
## LocalWeighting
##
#####
"""
    LocalWeighting

Abstract type for local weighting
"""
abstract type LocalWeighting end

"""
    TfWeighting()

Term frequency weighting
"""
struct TfWeighting <: LocalWeighting end

"""
    TpWeighting()

Term probability weighting
"""
struct TpWeighting <: LocalWeighting end

"""
    FreqWeighting()

Frequency weighting
"""
struct FreqWeighting <: LocalWeighting end

"""
    BinaryLocalWeighting()

The weight is 1 for known tokens, 0 for out of vocabulary tokens
"""
struct BinaryLocalWeighting <: LocalWeighting end

#####
##
## GlobalWeighting
##
#####
"""
    GlobalWeighting

Abstract type for global weighting
"""
abstract type GlobalWeighting end


"""
    IdfWeighting()

Inverse document frequency weighting
"""
struct IdfWeighting <: GlobalWeighting end


"""
    BinaryGlobalWeighting()

The weight is 1 for known tokens, 0 for out of vocabulary tokens
"""
struct BinaryGlobalWeighting <: GlobalWeighting end

#####
##
## TextModels
##
#####
"""
    Model

An abstract type that represents a weighting model
"""
abstract type TextModel end

"""
    TokenStats(occs, ndocs)

Stores useful information for a token; i.e., the number of occurrences in the collection, the number of documents having that token
"""
struct TokenStats
    id::Int32
    occs::Int32
    ndocs::Int32
    weight::Float32
end

const UnknownTokenStats = TokenStats(0, 0, 0, 0f0)

const Vocabulary = Dict{Symbol, TokenStats}
const IdTokenMap = Dict{Int32, Symbol}

mutable struct VectorModel{_L<:LocalWeighting, _G<:GlobalWeighting} <: TextModel
    local_weighting::_L
    global_weighting::_G
    tokens::Vocabulary
    id2token::IdTokenMap
    maxfreq::Int
    m::Int  # vocabulary size
    n::Int  # training collection size
end

StructTypes.construct(::Type{Int64}, s::String) = parse(Int64, s)
StructTypes.construct(::Type{Int32}, s::String) = parse(Int32, s)
StructTypes.construct(::Type{Int16}, s::String) = parse(Int16, s)
StructTypes.StructType(::Type{TokenStats}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:LocalWeighting}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:GlobalWeighting}) = StructTypes.Struct()
StructTypes.StructType(::Type{<:VectorModel}) = StructTypes.Struct()

function Base.copy(
        e::VectorModel;
        local_weighting=e.local_weighting,
        global_weighting=e.global_weighting,
        tokens=e.tokens,
        id2token=e.id2token,
        maxfreq=e.maxfreq,
        m=e.m,
        n=e.n
    )
    VectorModel(local_weighting, global_weighting, tokens, id2token, maxfreq, m, n)
end

function create_vocabulary(corpus)
    V = Vocabulary()
    for vec in corpus
        for (t, occ) in vec
            s = get(V, t, UnknownTokenStats)
            V[t] = TokenStats(0, s.occs + occ, s.ndocs + 1, 0f0)
        end
    end

    V
end

"""
    VectorModel(local_weighting::LocalWeighting, global_weighting::GlobalWeighting, corpus::BOW; minocc::Integer=1)

Trains a vector model using the input corpus. 
"""
function VectorModel(local_weighting::LocalWeighting, global_weighting::GlobalWeighting, corpus::AbstractVector{BOW}; minocc::Integer=1)
    tokens = Vocabulary()
    id2token = IdTokenMap()

    V = create_vocabulary(corpus)
    tokens = Vocabulary()
    tokenID = 0
    maxfreq = 0
    for (t, s) in V
        s.occs < minocc && continue
        tokenID += 1
        id2token[tokenID] = t
        tokens[t] = TokenStats(tokenID, s.occs, s.ndocs, 0.0f0)
        maxfreq = max(maxfreq, s.occs)
    end
    
    VectorModel(local_weighting, global_weighting, tokens, id2token, maxfreq, length(tokens), length(corpus))
end

Base.show(io::IO, model::VectorModel) = print(io, "{VectorModel local_weighting=$(model.local_weighting), global_weighting=$(model.global_weighting) vocsize=$(model.m), maxfreq=$(model.maxfreq)}")

"""
    prune(model::VectorModel, minocc, rank)

Cuts the vocabulary by frequency using lower and higher filter;
All tokens with frequency below `minfreq` are ignored; top `rank` tokens are also removed.
"""
function prune(model::VectorModel{_L,_G}, minocc::Integer, rank::Integer) where {_L,_G}
    W = [(token, s) for (token, s) in model.tokens if s.occs >= minocc]
    sort!(W, by=x->x[2])
    tokens = Vocabulary()
    for i in 1:length(W)-rank+1
        w = W[i]
        tokens[w[1]] = w[2]
    end

    id2token = IdTokenMap(s.id => t for (t, s) in tokens)
    VectorModel(_L, _G, tokens, id2token, model.maxfreq, model.m, model.n)
end

"""
    prune_select_top(model::VectorModel, k::Integer)
    prune_select_top(model::VectorModel, ratio::AbstractFloat)

Creates a new model with the best `k` tokens from `model` based on the `kind` scheme; kind must be either `IdfWeighting` or `FreqWeighting`.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer)
    tokens = Vocabulary()
    maxfreq = 0
    
    if model.global_weighting isa IdfWeighting
        idf = IdfWeighting()
        X = [(t, s, global_weighting(idf, s, model)) for (t, s) in model.tokens]
        sort!(X, by=x->x[end], rev=true)
        for i in 1:k
            t, s, w = X[i]
            tokens[t] = s
            maxfreq = max(maxfreq, s.freq)
        end
    elseif model.global_weighting isa EntropyWeighting
        X = [(t, s) for (t, s) in model.tokens]
        sort!(X, by=x->x[end].weight, rev=true)
        for i in 1:k
            t, s, w = X[i]
            tokens[t] = s
            maxfreq = max(maxfreq, s.freq)
        end
    else
        X = [(t, s) for (t, s) in model.tokens]
        sort!(X, by=x->x[end].occs, rev=true)
        for i in 1:k
            t, s = X[i]
            tokens[t] = s
            maxfreq = max(maxfreq, s.freq)
        end
    end

    id2token = Dict(s.id => t for (t, s) in tokens)
    VectorModel(model.local_weighting, model.global_weighting, tokens, id2token, maxfreq, model.m, model.n)
end

prune_select_top(model::VectorModel, ratio::AbstractFloat) = prune_select_top(model, floor(Int, length(model.tokens) * ratio))

"""
    vectorize(model::VectorModel, bow::BOW, maxfreq::Integer=maximum(bow); normalize=true) where Tv<:Real

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize(model::VectorModel{_L, _G}, bow::BOW; normalize=true) where {_L,_G}
    numtokens = 0.0
    if _L === TpWeighting
        for (token, freq) in bow
            numtokens += freq
        end
    end
    
    maxfreq = (_L === TfWeighting) ? maximum(bow) : 0.0
    vec = SVEC()

    for (token, freq) in bow
        s = get(model.tokens, token, nothing)
        s === nothing && continue

        lw = local_weighting(model.local_weighting, freq, maxfreq, numtokens)
        w = lw * global_weighting(model.global_weighting, s, model)
        if w > 1e-6
            vec[s.id] = w
        end
    end

    if length(vec) == 0
        vec[rand(typemin(Int32):-1)] = 1e-6
    end

    normalize && normalize!(vec)
    vec
end

function broadcastable(model::VectorModel)
    (model,)
end

# local weightings: TfWeighting, TpWeighting, FreqWeighting, BinaryLocalWeighting
# global weightings: IdfWeighting, BinaryGlobalWeighting

local_weighting(::TfWeighting, occs, maxfreq, numtokens) = occs / maxfreq
local_weighting(::FreqWeighting, occs, maxfreq, numtokens) = occs
local_weighting(::TpWeighting, occs, maxfreq, numtokens) = occs / numtokens
local_weighting(::BinaryLocalWeighting, occs, maxfreq, numtokens) = 1.0
global_weighting(::IdfWeighting, s::TokenStats, m::TextModel) = log(2, 1 + m.n / s.ndocs)
global_weighting(::BinaryGlobalWeighting, s::TokenStats, m::TextModel) = 1.0
