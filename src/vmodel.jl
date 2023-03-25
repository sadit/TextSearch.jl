# This file is part of TextSearch.jl

export TextModel, VectorModel, trainsize, vocsize,
    TfWeighting, IdfWeighting, TpWeighting,
    FreqWeighting, BinaryLocalWeighting, BinaryGlobalWeighting,
    LocalWeighting, GlobalWeighting, weight, fit, vectorize, vectorize_corpus

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

mutable struct VectorModel{_G<:GlobalWeighting, _L<:LocalWeighting} <: TextModel
    global_weighting::_G
    local_weighting::_L
    voc::Vocabulary
    maxoccs::Int32
    weight::Vector{Float32}
end

function VectorModel(gw::GlobalWeighting, lw::LocalWeighting, voc::Vocabulary; weight=nothing)
    length(voc.occs) > 0 || error("empty vocabulary")
    maxoccs = convert(Int32, maximum(voc.occs))
    W = weight === nothing ? Vector{Float32}(undef, vocsize(voc)) : weight
    model = VectorModel(gw, lw, voc, maxoccs, W)

    if weight === nothing
        for tokenID in eachindex(voc)
            model.weight[tokenID] = w_ = global_weighting(model, tokenID)
            # @assert w_ >= 0 "NEGATIVE WEIGHT $tokenID -- $w_"
        end
    end

    model
end

function VectorModel(
        e::VectorModel;
        local_weighting=e.local_weighting,
        global_weighting=e.global_weighting,
        voc=e.voc,
        maxoccs=e.maxoccs,
        weight=e.weight
    )

    VectorModel(global_weighting, local_weighting, voc, maxoccs, weight)
end

Base.copy(e::VectorModel; kwargs...) = VectorModel(e::VectorModel; kwargs...)

@inline trainsize(model::VectorModel) = trainsize(model.voc)
@inline vocsize(model::VectorModel) = vocsize(model.voc)

@inline Base.length(model::VectorModel) = length(model.voc)
@inline occs(model::VectorModel, tokenID::Integer) = occs(model.voc, tokenID)
@inline ndocs(model::VectorModel, tokenID::Integer) = ndocs(model.voc, tokenID)
@inline token(model::VectorModel, tokenID::Integer) = token(model.voc, tokenID)
@inline Base.eachindex(model::VectorModel) = eachindex(model.voc)
@inline weight(model::VectorModel, tokenID::Integer) = tokenID == 0 ? zero(eltype(model.weight)) : model.weight[tokenID]
@inline weight(model::VectorModel) = model.weight
@inline occs(model::VectorModel) = occs(model.voc)
@inline ndocs(model::VectorModel) = ndocs(model.voc)
@inline token(model::VectorModel) = token(model.voc)

function Base.getindex(model::VectorModel, tokenID::Integer)
    id = convert(UInt32, tokenID)
    voc = model.voc
    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), weight=zero(eltype(model.weight)), token="")
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], weight=model.weight[id], token=voc.token[id])
    end
end

function VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    voc = Vocabulary(textconfig, corpus; minbatch)
    VectorModel(global_weighting, local_weighting, voc)
end

Base.show(io::IO, model::VectorModel) = print(io, "{VectorModel global_weighting=$(model.global_weighting), local_weighting=$(model.local_weighting), train-voc=$(vocsize(model)), train-n=$(trainsize(model)), maxoccs=$(model.maxoccs)}")

function filter_tokens(pred::Function, model::VectorModel)
    voc = model.voc
    V = Vocabulary(voc.corpuslen)
    W = Vector{Float32}(undef, 0)
    
    for i in eachindex(voc)
        t = model[i]
        if pred(t)
            push_token!(V, t.token, t.occs, t.ndocs)
            push!(W, t.weight)
        end
    end

    VectorModel(model.global_weighting, model.local_weighting, V; weight=W)
end

"""
    vectorize!(buff::TextSearchBuffer, model::VectorModel{G_,L_}, bow::BOW; normalize=true, minweight=1e-9) where {G_,L_}

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize!(buff::TextSearchBuffer, model::VectorModel{G_,L_}, bow::BOW; normalize=true, minweight=1e-9) where {G_,L_}
    vec = buff.vec
    numtokens::Int = 0

    if L_ === TpWeighting
        for freq in values(bow)
            numtokens += freq
        end
    end
    
    maxoccs::Int = 0
    if L_ === TfWeighting
        maxoccs = length(bow) == 0 ? 0 : maximum(bow) 
    end
    
    for (tokenID, freq) in bow
        # w = local_weighting(model.local_weighting, freq, maxoccs, numtokens) * global_weighting(model, tokenID)
        w = local_weighting(model.local_weighting, freq, maxoccs, numtokens) * weight(model, tokenID)
        w >= minweight && (vec[tokenID] = w)
    end

    if length(vec) == 0
        vec[0] = 1f0
    else
        normalize && normalize!(vec)
    end

    buff
end

function vectorize!(buff::TextSearchBuffer, model::VectorModel, textconfig::TextConfig, text; normalize=true, minweight=1e-9)
    empty!(buff)
    bagofwords!(buff, model.voc, textconfig, text)
    vectorize!(buff, model, buff.bow; normalize, minweight)
end

function vectorize(model::VectorModel, textconfig::TextConfig, text; normalize=true, minweight=1e-9)
    buff = take!(TEXT_SEARCH_CACHES)
    try
        vectorize!(buff, model, textconfig, text; normalize, minweight)
        buff.vec |> copy
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

function vectorize_corpus(model::VectorModel, textconfig::TextConfig, corpus::AbstractVector; normalize=true, minweight=1e-9, minbatch=0)
    n = length(corpus)
    V = [vectorize(model, textconfig, corpus[1]; normalize, minweight)] # Vector{SVEC}(undef, n)
    resize!(V, n)
    minbatch = getminbatch(minbatch, n)

    # @batch minbatch=minbatch per=thread
    Threads.@threads for i in 2:n
        V[i] = vectorize(model, textconfig, corpus[i]; normalize, minweight)
    end

    V
end

function broadcastable(model::VectorModel)
    (model,)
end

# local weightings: TfWeighting, TpWeighting, FreqWeighting, BinaryLocalWeighting
# global weightings: IdfWeighting, BinaryGlobalWeighting

@inline local_weighting(::TfWeighting, occs, maxoccs, numtokens) = occs / maxoccs
@inline local_weighting(::FreqWeighting, occs, maxoccs, numtokens) = occs
@inline local_weighting(::TpWeighting, occs, maxoccs, numtokens) = occs / numtokens
@inline local_weighting(::BinaryLocalWeighting, occs, maxoccs, numtokens) = 1.0
@inline global_weighting(model::VectorModel{IdfWeighting}, tokenID) = @inbounds log2((0.5 + trainsize(model)) / (0.5 + ndocs(model, tokenID)))
@inline global_weighting(model::VectorModel{BinaryGlobalWeighting}, tokenID) = 1.0
