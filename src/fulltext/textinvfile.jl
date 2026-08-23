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
- `query_expansion`: `nothing`, or a query_expansion network (e.g. as produced by `LSI.query_expansion`) used to expand
  queries -- never documents -- via
  [`expand_query!`](@ref). Never applied to documents, only to queries, and only under vector
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
struct TextInvertedFile{ModelType<:VectorModel, InvFileType<:InvertedFile} <: AbstractInvertedFile
    model::ModelType
    invfile::InvFileType
    query::QueryPipeline
end

is_set_distance(dist) = parentmodule(typeof(dist)) === SimilaritySearch.Dist.Sets

# Property forwarding to invfile for Seamless AbstractInvertedFile behavior
function Base.getproperty(idx::TextInvertedFile, s::Symbol)
    s === :model && return getfield(idx, :model)
    s === :invfile && return getfield(idx, :invfile)
    s === :query && return getfield(idx, :query)
    # kept because it is what a caller who handed over a network asks for afterwards
    s === :query_expansion && return getfield(idx, :query).expansion
    getproperty(getfield(idx, :invfile), s)
end

Base.propertynames(idx::TextInvertedFile) = (:model, :invfile, :query, :query_expansion, propertynames(getfield(idx, :invfile))...)

Base.length(idx::TextInvertedFile) = length(idx.invfile)
SimilaritySearch.database(idx::TextInvertedFile) = database(idx.invfile)
SimilaritySearch.distance(idx::TextInvertedFile) = distance(idx.invfile)

# Constructors
"""
    TextInvertedFile(model::VectorModel; dist=Dist.NormCosine(), query_expansion=nothing, kwargs...)

Creates an empty [`TextInvertedFile`](@ref) backed by `model` and `dist`. Pass `query_expansion` (e.g. as
produced by `LSI.query_expansion`) to enable query-time expansion (also requires
see [`expand_query!`](@ref)). Handing a network over IS the request to expand with it;
whether a profile wants that is recorded as its `applied.query_expansion`.
"""
function TextInvertedFile(model::VectorModel; dist=Dist.NormCosine(), query_expansion=nothing,
                          distances=nothing, query=nothing, kwargs...)
    query === nothing || query_expansion === nothing ||
        throw(ArgumentError("pass either `query` (a QueryPipeline) or `query_expansion`, not both"))
    qp = query === nothing ? QueryPipeline(; expansion=query_expansion, distances) : query
    invfile = InvertedFile(vocsize(model.voc), dist; kwargs...)
    TextInvertedFile(model, invfile, qp)
end

"""
    TextInvertedFile(voc::Vocabulary, local_weighting=TfWeighting(), global_weighting=IdfWeighting(); dist=Dist.NormCosine(), query_expansion=nothing, distances=nothing, query=nothing, kwargs...)

Creates a [`TextInvertedFile`](@ref) from a [`Vocabulary`](@ref) and specified local/global weighting schemes.
"""
function TextInvertedFile(voc::Vocabulary, local_weighting=TfWeighting(), global_weighting=IdfWeighting(); dist=Dist.NormCosine(), query_expansion=nothing, distances=nothing, query=nothing, kwargs...)
    model = VectorModel(global_weighting, local_weighting, voc)
    TextInvertedFile(model; dist, query_expansion, distances, query, kwargs...)
end

"""
    TextInvertedFile(textconfig::TextConfig, corpus; local_weighting=TfWeighting(), global_weighting=IdfWeighting(), dist=Dist.NormCosine(), query_expansion=nothing, distances=nothing, query=nothing, kwargs...)

Convenience constructor that builds a [`Vocabulary`](@ref) from `corpus` under `textconfig`, creates a [`VectorModel`](@ref), and returns a [`TextInvertedFile`](@ref).
"""
function TextInvertedFile(textconfig::TextConfig, corpus; local_weighting=TfWeighting(), global_weighting=IdfWeighting(), dist=Dist.NormCosine(), query_expansion=nothing, distances=nothing, query=nothing, kwargs...)
    voc = Vocabulary(textconfig, corpus)
    TextInvertedFile(voc, local_weighting, global_weighting; dist, query_expansion, distances, query, kwargs...)
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
    # one query pipeline, in the library: it corrects and expands on strings, and the
    # representation decides what to do with the weights it produces -- a set distance ignores
    # them (presence only), a vector distance applies them and normalizes.
    rq = query_tokens(idx.model.voc, qtext, idx.query)
    q = is_set_distance(distance(idx)) ? querybow(idx.model.voc, rq) : queryvector(idx.model, rq)
    search(idx.invfile, ctx, q, res; t)
end

function SimilaritySearch.search(idx::TextInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnnQueue; t::Int=1)
    search(idx.invfile, ctx, q, res; t)
end
