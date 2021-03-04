# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextModel, VectorModel,
    TfWeighting, IdfWeighting, TpWeighting,
    FreqWeighting, BinaryLocalWeighting, BinaryGlobalWeighting,
    LocalWeighting, GlobalWeighting,
    fit, vectorize, prune, prune_select_top

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

mutable struct VectorModel{_G<:GlobalWeighting, _L<:LocalWeighting} <: TextModel
    global_weighting::_G
    local_weighting::_L
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
    VectorModel(global_weighting, local_weighting, tokens, id2token, maxfreq, m, n)
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
    VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, corpus::BOW; mindocs=1)

Creates a vector model using the input corpus. 
"""
function VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, corpus::AbstractVector{BOW}; mindocs=1)
    tokens = Vocabulary()
    id2token = IdTokenMap()

    V = create_vocabulary(corpus)
    tokens = Vocabulary()
    tokenID = 0
    maxfreq = 0
    for (t, s) in V
        s.ndocs < mindocs && continue
        tokenID += 1
        id2token[tokenID] = t
        tokens[t] = TokenStats(tokenID, s.occs, s.ndocs, 0.0f0)
        maxfreq = max(maxfreq, s.occs)
    end
    
    VectorModel(global_weighting, local_weighting, tokens, id2token, maxfreq, length(tokens), length(corpus))
end

Base.show(io::IO, model::VectorModel) = print(io, "{VectorModel global_weighting=$(model.global_weighting), local_weighting=$(model.local_weighting), train-voc=$(model.m), train-n=$(model.n), maxfreq=$(model.maxfreq)}")


"""
    prune(model::VectorModel, lowerweight::AbstractFloat)

Creates a new vector model without terms with smaller global weights than `lowerweight`.
"""
function prune(model::VectorModel, lowerweight::AbstractFloat)
    tokens = Vocabulary()
    id2token = IdTokenMap()
    i = 0
    for (token, s) in model.tokens
        if prune_global_weighting(model, s) >= lowerweight
            i += 1
            tokens[token] = TokenStats(i, s.occs, s.ndocs, s.weight)
            id2token[i] = token
        end
    end

    VectorModel(model.global_weighting, model.local_weighting, tokens, id2token, model.maxfreq, model.m, model.n)
end

"""
    prune_select_top(model::VectorModel, k::Integer)
    prune_select_top(model::VectorModel, ratio::AbstractFloat)

Creates a new vector model with the best `k` tokens from `model` based on global weighting.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer)
    w = [prune_global_weighting(model, s) for (token, s) in model.tokens]
    sort!(w, rev=true)
    prune(model, w[k])
end

prune_select_top(model::VectorModel, ratio::AbstractFloat) = prune_select_top(model, floor(Int, length(model.tokens) * ratio))

"""
    vectorize(model::VectorModel, bow::BOW, maxfreq::Integer=maximum(bow); normalize=true) where Tv<:Real

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize(model::VectorModel{_G, _L}, bow::BOW; normalize=true) where {_L,_G}
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
        gw = global_weighting(model, s)
        # @info (token, _G => gw, _L => lw)
        w = Float64(lw * gw)
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
global_weighting(m::VectorModel{IdfWeighting}, s::TokenStats) = log(2, 1 + m.n / s.ndocs)
global_weighting(m::VectorModel{BinaryGlobalWeighting}, s::TokenStats) = 1.0
prune_global_weighting(m::VectorModel, s) = global_weighting(m, s)
prune_global_weighting(m::VectorModel{BinaryGlobalWeighting}, s) = log(2, 1 + m.n / s.ndocs) * log2(s.occs)