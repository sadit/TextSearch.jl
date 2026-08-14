# This file is part of TextSearch.jl

export TextInvertedFile

"""
    TextInvertedFile{ModelType<:VectorModel, InvFileType<:InvertedFile} <: AbstractInvertedFile

An inverted-file index (built on top of `SimilaritySearch.InvertedFiles`) that pairs a text weighting model
([`VectorModel`](@ref)) with an inverted index for fast kNN / radius search under vector distances
(e.g., `Dist.NormCosine()`, `Dist.Cosine()`) or set distances (e.g., `Dist.Sets.Jaccard()`, `Dist.Sets.Dice()`).

# Fields
- `model`: the [`VectorModel`](@ref) used to vectorize documents and queries.
- `invfile`: the underlying `SimilaritySearch.InvertedFiles.InvertedFile`.

# Example

```julia
julia> using SimilaritySearch, TextSearch

julia> corpus = ["la casa roja", "la casa verde", "la casa azul"];

julia> voc = Vocabulary(TextConfig(), corpus);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> idx = TextInvertedFile(model; dist=Dist.NormCosine());

julia> ctx = InvertedFileContext();

julia> append_items!(idx, ctx, corpus);

julia> res = knnqueue(KnnSorted, 2);

julia> search(idx, ctx, "la casa roja", res);

julia> collect(IdView(res))
UInt32[0x00000001, 0x00000002]
```
"""
struct TextInvertedFile{ModelType<:VectorModel, InvFileType<:InvertedFile} <: AbstractInvertedFile
    model::ModelType
    invfile::InvFileType
end

is_set_distance(dist) = parentmodule(typeof(dist)) === SimilaritySearch.Dist.Sets

# Property forwarding to invfile for Seamless AbstractInvertedFile behavior
function Base.getproperty(idx::TextInvertedFile, s::Symbol)
    s === :model && return getfield(idx, :model)
    s === :invfile && return getfield(idx, :invfile)
    getproperty(getfield(idx, :invfile), s)
end

Base.propertynames(idx::TextInvertedFile) = (:model, :invfile, propertynames(getfield(idx, :invfile))...)

Base.length(idx::TextInvertedFile) = length(idx.invfile)
SimilaritySearch.database(idx::TextInvertedFile) = database(idx.invfile)
SimilaritySearch.distance(idx::TextInvertedFile) = distance(idx.invfile)

# Constructors
"""
    TextInvertedFile(model::VectorModel; dist=Dist.NormCosine(), kwargs...)

Creates an empty [`TextInvertedFile`](@ref) backed by `model` and `dist`.
"""
function TextInvertedFile(model::VectorModel; dist=Dist.NormCosine(), kwargs...)
    invfile = InvertedFile(vocsize(model.voc), dist; kwargs...)
    TextInvertedFile(model, invfile)
end

"""
    TextInvertedFile(voc::Vocabulary, local_weighting=TfWeighting(), global_weighting=IdfWeighting(); dist=Dist.NormCosine(), kwargs...)

Creates a [`TextInvertedFile`](@ref) from a [`Vocabulary`](@ref) and specified local/global weighting schemes.
"""
function TextInvertedFile(voc::Vocabulary, local_weighting=TfWeighting(), global_weighting=IdfWeighting(); dist=Dist.NormCosine(), kwargs...)
    model = VectorModel(global_weighting, local_weighting, voc)
    TextInvertedFile(model; dist, kwargs...)
end

"""
    TextInvertedFile(textconfig::TextConfig, corpus; local_weighting=TfWeighting(), global_weighting=IdfWeighting(), dist=Dist.NormCosine(), kwargs...)

Convenience constructor that builds a [`Vocabulary`](@ref) from `corpus` under `textconfig`, creates a [`VectorModel`](@ref), and returns a [`TextInvertedFile`](@ref).
"""
function TextInvertedFile(textconfig::TextConfig, corpus; local_weighting=TfWeighting(), global_weighting=IdfWeighting(), dist=Dist.NormCosine(), kwargs...)
    voc = Vocabulary(textconfig, corpus)
    TextInvertedFile(voc, local_weighting, global_weighting; dist, kwargs...)
end

# InvertedFile insertion & appending methods
function SimilaritySearch.push_item!(idx::TextInvertedFile, ctx::InvertedFileContext, doc::T) where {T<:Union{AbstractString,TokenizedText}}
    if is_set_distance(distance(idx))
        push_item!(idx.invfile, ctx, bagofwords(idx.model.voc, doc))
    else
        push_item!(idx.invfile, ctx, vectorize(idx.model, doc))
    end
    idx
end

function SimilaritySearch.push_item!(idx::TextInvertedFile, ctx::InvertedFileContext, obj)
    push_item!(idx.invfile, ctx, obj)
    idx
end

function SimilaritySearch.append_items!(idx::TextInvertedFile, ctx::InvertedFileContext, corpus::AbstractVector{T}; kwargs...) where {T<:Union{AbstractString,TokenizedText}}
    if is_set_distance(distance(idx))
        bows = bagofwords_corpus(idx.model.voc, corpus)
        append_items!(idx.invfile, ctx, VectorDatabase(bows); kwargs...)
    else
        vecs = vectorize_corpus(idx.model, corpus)
        append_items!(idx.invfile, ctx, VectorDatabase(vecs); kwargs...)
    end
    idx
end

function SimilaritySearch.append_items!(idx::TextInvertedFile, ctx::InvertedFileContext, db::AbstractDatabase; kwargs...)
    append_items!(idx.invfile, ctx, db; kwargs...)
    idx
end

# Search methods
function SimilaritySearch.search(idx::TextInvertedFile, ctx::InvertedFileContext, qtext::T, res::AbstractKnnQueue; t::Int=1) where {T<:Union{AbstractString,TokenizedText}}
    if is_set_distance(distance(idx))
        q = bagofwords(idx.model.voc, qtext)
    else
        q = vectorize(idx.model, qtext)
    end
    search(idx.invfile, ctx, q, res; t)
end

function SimilaritySearch.search(idx::TextInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnnQueue; t::Int=1)
    search(idx.invfile, ctx, q, res; t)
end
