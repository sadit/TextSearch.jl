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

trainsize(model::VectorModel) = model.voc.corpuslen
vocsize(model::VectorModel) = length(model.voc)

"""
    VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, voc::Vocabulary; mindocs=1)

Creates a vector model using the input corpus. 
"""
function VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, voc::Vocabulary; mindocs=1)
    maxocc = convert(Int32, maximum(voc.occs))
    mindocs = convert(Int32, mindocs)

    VectorModel(global_weighting, local_weighting, voc, maxocc, mindocs)
end

function VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, tok::Tokenizer, corpus::AbstractVector; mindocs=1)
    voc = Vocabulary(tok, corpus)
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
    for tokenID in eachindex(old.occs)
        if prune_global_weighting(model, tokenID) >= lowerweight
            push!(voc, old.token[tokenID], old.occs[tokenID], old.ndocs[tokenID], old.weight[tokenID])
        end
    end

    VectorModel(model.global_weighting, model.local_weighting, voc, model.maxoccs, model.mindocs)
end

"""
    prune_select_top(model::VectorModel, k::Integer)
    prune_select_top(model::VectorModel, ratio::AbstractFloat)

Creates a new vector model with the best `k` tokens from `model` based on global weighting.
`ratio` is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept
"""
function prune_select_top(model::VectorModel, k::Integer)
    voc = model.voc
    w = [prune_global_weighting(model, tokenID) for tokenID in eachindex(voc.token)]
    sort!(w, rev=true)
    prune(model, float(w[k]))
end

prune_select_top(model::VectorModel, ratio::AbstractFloat) =
    prune_select_top(model, floor(Int, length(model.voc) * ratio))

"""
    vectorize(model::VectorModel, bow::BOW; normalize=true, mindocs=model.mindocs, minweight=1e-6) where Tv<:Real

Computes a weighted vector using the given bag of words and the specified weighting scheme.
"""
function vectorize(model::VectorModel{_G,_L}, bow::BOW; normalize=true, mindocs=model.mindocs, minweight=1e-6) where {_G,_L}
    numtokens = 0.0
    if _L === TpWeighting
        for freq in values(bow)
            numtokens += freq
        end
    end
    
    maxoccs = 0
    if _L === TfWeighting
        maxoccs = length(bow) == 0 ? 0 : maximum(bow) 
    end
    
    vec = SVEC()
    voc = model.voc
    for (tokenID, freq) in bow
        voc.ndocs[tokenID] < mindocs && continue
        w = local_weighting(model.local_weighting, freq, maxoccs, numtokens) * global_weighting(model, tokenID)
        # @show _G => global_weighting(model, tokenID), _L => local_weighting(model.local_weighting, freq, maxoccs, numtokens)
        # @show w, tokenID => voc.token[tokenID], vec
        w >= minweight && (vec[tokenID] = w)
    end

    normalize && length(vec) > 0 && normalize!(vec)
    vec
end

function vectorize(model::VectorModel, tok::Tokenizer, text; bow=BOW(), normalize=true, mindocs=model.mindocs, minweight=1e-6)
    vectorize(model, vectorize(model.voc, tok, text, bow); normalize, mindocs, minweight)
end

function vectorize_corpus(model::VectorModel, tok::Tokenizer, corpus; bow=BOW(), normalize=true)
    V = Vector{SVEC}(undef, length(corpus))

    for (i, text) in enumerate(corpus)
        empty!(bow)
        V[i] = vectorize(model, tok, text; bow, normalize)
    end

    V
end

function broadcastable(model::VectorModel)
    (model,)
end

# local weightings: TfWeighting, TpWeighting, FreqWeighting, BinaryLocalWeighting
# global weightings: IdfWeighting, BinaryGlobalWeighting

local_weighting(::TfWeighting, occs, maxoccs, numtokens) = occs / maxoccs
local_weighting(::FreqWeighting, occs, maxoccs, numtokens) = occs
local_weighting(::TpWeighting, occs, maxoccs, numtokens) = occs / numtokens
local_weighting(::BinaryLocalWeighting, occs, maxoccs, numtokens) = 1.0
global_weighting(m::VectorModel{IdfWeighting}, tokenID) = @inbounds log2(1 + trainsize(m) / (0.01 + m.voc.ndocs[tokenID]))
global_weighting(m::VectorModel{BinaryGlobalWeighting}, tokenID) = 1.0
prune_global_weighting(m::VectorModel, tokenID) = global_weighting(m, tokenID)
prune_global_weighting(m::VectorModel{BinaryGlobalWeighting}, tokenID) = @inbounds -m.voc.ndocs[tokenID]