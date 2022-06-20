# This file is a part of TextSearch.jl

export TextModel, VectorModel, trainsize, vocsize,
    TfWeighting, IdfWeighting, TpWeighting,
    FreqWeighting, BinaryLocalWeighting, BinaryGlobalWeighting,
    LocalWeighting, GlobalWeighting,
    fit, prune, prune_select_top

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
    mindocs::Int32
end

function Base.copy(
        e::VectorModel;
        local_weighting=e.local_weighting,
        global_weighting=e.global_weighting,
        voc=e.voc,
        maxoccs=e.maxoccs,
        mindocs=e.mindocs,
    )
    VectorModel(global_weighting, local_weighting, voc, maxoccs, mindocs)
end

@inline trainsize(model::VectorModel) = trainsize(model.voc)
@inline vocsize(model::VectorModel) = vocsize(model.voc)
@inline Base.length(model::VectorModel) = length(model.voc)
@inline occs(model::VectorModel, tokenID::Integer) = occs(model.voc, tokenID)
@inline ndocs(model::VectorModel, tokenID::Integer) = ndocs(model.voc, tokenID)
@inline weight(model::VectorModel, tokenID::Integer) = weight(model.voc, tokenID)
@inline token(model::VectorModel, tokenID::Integer) = token(model.voc, tokenID)
@inline Base.eachindex(model::VectorModel) = eachindex(model.voc)

"""
    VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, voc::Vocabulary; mindocs=1)

Creates a vector model using the input corpus. 
"""
function VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, voc::Vocabulary; mindocs=1)
    maxocc = convert(Int32, maximum(voc.occs))
    mindocs = convert(Int32, mindocs)

    VectorModel(global_weighting, local_weighting, voc, maxocc, mindocs)
end

function VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, textconfig::TextConfig, corpus::AbstractVector; mindocs=1, minbatch=0)
    voc = Vocabulary(textconfig, corpus; minbatch)
    maxocc = convert(Int32, maximum(voc.occs))
    mindocs = convert(Int32, mindocs)
    VectorModel(global_weighting, local_weighting, voc, maxocc, mindocs)
end

Base.show(io::IO, model::VectorModel) = print(io, "{VectorModel global_weighting=$(model.global_weighting), local_weighting=$(model.local_weighting), train-voc=$(vocsize(model)), train-n=$(trainsize(model)), maxoccs=$(model.maxoccs)}")

"""
    prune(model::VectorModel, lowerweight::AbstractFloat)

Creates a new vector model without terms with smaller global weights than `lowerweight`.
"""
function prune(model::VectorModel, lowerweight::AbstractFloat)
    voc = Vocabulary(trainsize(model))
    old = model.voc
    for tokenID in eachindex(old)
        if prune_global_weighting(model, tokenID) >= lowerweight
            push!(voc, old.token[tokenID], old.occs[tokenID], old.ndocs[tokenID], old.weight[tokenID])
        end
    end

    VectorModel(model.global_weighting, model.local_weighting, voc, maximum(voc.occs), model.mindocs)
end

"""
    prune_select_top(model::VectorModel, k::Integer)
    prune_select_top(model::VectorModel, ratio::AbstractFloat)

Creates a new vector model with the best `k` tokens from `model` based on global weighting.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer)
    voc = model.voc
    w = [prune_global_weighting(model, tokenID) for tokenID in eachindex(voc)]
    sort!(w, rev=true)
    prune(model, float(w[k]))
end

prune_select_top(model::VectorModel, ratio::AbstractFloat) =
    prune_select_top(model, floor(Int, length(model.voc) * ratio))

"""
    vectorize(model::VectorModel, bow::BOW; normalize=true, mindocs=model.mindocs, minweight=1e-9) where Tv<:Real

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize(model::VectorModel{_G,_L}, bow::BOW; normalize=true, mindocs=model.mindocs, minweight=1e-9) where {_G,_L}
    numtokens::Int = 0
    if _L === TpWeighting
        for freq in values(bow)
            numtokens += freq
        end
    end
    
    maxoccs::Int = 0
    if _L === TfWeighting
        maxoccs = length(bow) == 0 ? 0 : maximum(bow) 
    end
    
    vec = SVEC()
    voc = model.voc
    for (tokenID, freq) in bow
        voc.ndocs[tokenID] < mindocs && continue
        w = local_weighting(model.local_weighting, freq, maxoccs, numtokens) * global_weighting(model, tokenID)
        w >= minweight && (vec[tokenID] = w)
    end

    if length(vec) == 0
        vec[0] = 1f0
    else
        normalize && normalize!(vec)
    end

    vec
end

function vectorize(model::VectorModel, textconfig::TextConfig, text, buff::TextSearchBuffer; normalize=true, mindocs=model.mindocs, minweight=1e-9)
    empty!(buff)
    bow = vectorize(model.voc, textconfig, text, buff)
    vectorize(model, bow; normalize, mindocs, minweight)
end

function vectorize(model::VectorModel, textconfig::TextConfig, text; normalize=true, mindocs=model.mindocs, minweight=1e-9)
    buff = take!(CACHES)
    try
        copy(vectorize(model, textconfig, text, buff; normalize, mindocs, minweight))
    finally
        put!(CACHES, buff)
    end
end

function vectorize_corpus(model::VectorModel, textconfig::TextConfig, corpus::AbstractVector; normalize=true, minbatch=0)
    n = length(corpus)
    V = Vector{SVEC}(undef, n)
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        text = corpus[i]
        buff = take!(CACHES)
        try
            V[i] = copy(vectorize(model, textconfig, text, buff; normalize))
        finally
            put!(CACHES, buff)
        end
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
@inline global_weighting(model::VectorModel{IdfWeighting}, tokenID) = @inbounds log2(trainsize(model) / (1 + ndocs(model, tokenID)))
@inline global_weighting(model::VectorModel{BinaryGlobalWeighting}, tokenID) = 1.0
@inline prune_global_weighting(model::VectorModel, tokenID) = global_weighting(model, tokenID)
@inline prune_global_weighting(model::VectorModel{BinaryGlobalWeighting}, tokenID) = @inbounds 1 / (1 + ndocs(model, tokenID))