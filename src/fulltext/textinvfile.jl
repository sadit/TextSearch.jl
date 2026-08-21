# This file is part of TextSearch.jl

export TextInvertedFile

"""
    TextInvertedFile{ModelType<:VectorModel, InvFileType<:InvertedFile, SynType} <: AbstractInvertedFile

An inverted-file index (built on top of `SimilaritySearch.InvertedFiles`) that pairs a text weighting model
([`VectorModel`](@ref)) with an inverted index for fast kNN / radius search under vector distances
(e.g., `Dist.NormCosine()`, `Dist.Cosine()`) or set distances (e.g., `Dist.Sets.Jaccard()`, `Dist.Sets.Dice()`).

# Fields
- `model`: the [`VectorModel`](@ref) used to vectorize documents and queries.
- `invfile`: the underlying `SimilaritySearch.InvertedFiles.InvertedFile`.
- `synonyms`: `nothing`, or a synonym network (e.g. as produced by `LSI.synonyms`) used to expand
  queries -- never documents -- via
  [`expand_synonyms!`](@ref). Never applied to documents, only to queries, and only under vector
  distances (not set distances).

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
struct TextInvertedFile{ModelType<:VectorModel, InvFileType<:InvertedFile, SynType} <: AbstractInvertedFile
    model::ModelType
    invfile::InvFileType
    synonyms::SynType
end

is_set_distance(dist) = parentmodule(typeof(dist)) === SimilaritySearch.Dist.Sets

# Property forwarding to invfile for Seamless AbstractInvertedFile behavior
function Base.getproperty(idx::TextInvertedFile, s::Symbol)
    s === :model && return getfield(idx, :model)
    s === :invfile && return getfield(idx, :invfile)
    s === :synonyms && return getfield(idx, :synonyms)
    getproperty(getfield(idx, :invfile), s)
end

Base.propertynames(idx::TextInvertedFile) = (:model, :invfile, :synonyms, propertynames(getfield(idx, :invfile))...)

Base.length(idx::TextInvertedFile) = length(idx.invfile)
SimilaritySearch.database(idx::TextInvertedFile) = database(idx.invfile)
SimilaritySearch.distance(idx::TextInvertedFile) = distance(idx.invfile)

# Constructors
"""
    TextInvertedFile(model::VectorModel; dist=Dist.NormCosine(), synonyms=nothing, kwargs...)

Creates an empty [`TextInvertedFile`](@ref) backed by `model` and `dist`. Pass `synonyms` (e.g. as
produced by `LSI.synonyms`) to enable query-time synonym expansion (also requires
see [`expand_synonyms!`](@ref)). Handing a network over IS the request to expand with it;
whether a profile wants that is recorded as its `applied.synonyms`.
"""
function TextInvertedFile(model::VectorModel; dist=Dist.NormCosine(), synonyms=nothing, kwargs...)
    invfile = InvertedFile(vocsize(model.voc), dist; kwargs...)
    TextInvertedFile(model, invfile, synonyms)
end

"""
    TextInvertedFile(voc::Vocabulary, local_weighting=TfWeighting(), global_weighting=IdfWeighting(); dist=Dist.NormCosine(), synonyms=nothing, kwargs...)

Creates a [`TextInvertedFile`](@ref) from a [`Vocabulary`](@ref) and specified local/global weighting schemes.
"""
function TextInvertedFile(voc::Vocabulary, local_weighting=TfWeighting(), global_weighting=IdfWeighting(); dist=Dist.NormCosine(), synonyms=nothing, kwargs...)
    model = VectorModel(global_weighting, local_weighting, voc)
    TextInvertedFile(model; dist, synonyms, kwargs...)
end

"""
    TextInvertedFile(textconfig::TextConfig, corpus; local_weighting=TfWeighting(), global_weighting=IdfWeighting(), dist=Dist.NormCosine(), synonyms=nothing, kwargs...)

Convenience constructor that builds a [`Vocabulary`](@ref) from `corpus` under `textconfig`, creates a [`VectorModel`](@ref), and returns a [`TextInvertedFile`](@ref).
"""
function TextInvertedFile(textconfig::TextConfig, corpus; local_weighting=TfWeighting(), global_weighting=IdfWeighting(), dist=Dist.NormCosine(), synonyms=nothing, kwargs...)
    voc = Vocabulary(textconfig, corpus)
    TextInvertedFile(voc, local_weighting, global_weighting; dist, synonyms, kwargs...)
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
        q = vectorize(idx.model, qtext; normalize=false)
        if idx.synonyms !== nothing
            expand_synonyms!(q, idx.model.voc, idx.synonyms)
        else
            normalize!(q)
        end
    end
    search(idx.invfile, ctx, q, res; t)
end

function SimilaritySearch.search(idx::TextInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnnQueue; t::Int=1)
    search(idx.invfile, ctx, q, res; t)
end
