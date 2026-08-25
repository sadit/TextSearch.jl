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

"""
    TextInvertedFile(p::TextProfile; dist=Dist.NormCosine(), policy=QueryPolicy(), expansion=p.applied.query_expansion, kwargs...)

Creates an empty [`TextInvertedFile`](@ref) from a fitted [`TextProfile`](@ref), using the
profile's own model -- so the weighting scheme, the idf and the tokenization are the corpus's, not
the indexed subset's.

How queries are answered follows the profile, with one deliberate asymmetry. **Expansion is
gated by the profile**: the network is handed to the index only when `applied.query_expansion`
says the profile endorses it, since it is an artifact the profile may carry without meaning it
to be used -- pass `expansion=true` to take it anyway, which is what a *base* profile needs.
**Correction is gated by the policy**, because it depends on nothing but the vocabulary, which
every profile has; the variant map is derived once here rather than once per query, and comes
out empty at no cost for a profile that folds case and diacritics.

See [`BM25InvertedFile`](@ref)`(p::TextProfile)` for the same thing under BM25 ranking, where the
profile also lends its `avgdoclen`.
"""
function TextInvertedFile(p::TextProfile; dist=Dist.NormCosine(),
                          policy::QueryPolicy=QueryPolicy(),
                          expansion::Bool=p.applied.query_expansion, kwargs...)
    voc = p.model.voc
    TextInvertedFile(p.model; dist, kwargs..., query=QueryPipeline(;
        policy,
        variants = policy.correction === :off ? nothing : derive_variants(voc),
        expansion = expansion ? p.query_expansion : nothing,
        distances = expansion ? p.query_expansion_distances : nothing))
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
function SimilaritySearch.search(idx::TextInvertedFile, ctx::InvertedFileContext, qtext::T, res::AbstractKnnQueue;
                                 t::Int=1, policy::QueryPolicy=idx.query.policy) where {T<:Union{AbstractString,TokenizedText}}
    # one query pipeline, in the library: it corrects and expands on strings, and the
    # representation decides what to do with the weights it produces -- a set distance ignores
    # them (presence only), a vector distance applies them and normalizes.
    #
    # `policy` is per call because it is a property of the query, not of the index: the same
    # index has to be able to answer both the corrected reading and the literal one. The maps
    # it needs stay in `idx.query`, where they were derived once.
    rq = query_tokens(idx.model.voc, qtext, idx.query; policy)
    q = is_set_distance(distance(idx)) ? querybow(idx.model.voc, rq) : queryvector(idx.model, rq)
    search(idx.invfile, ctx, q, res; t)
end

function SimilaritySearch.search(idx::TextInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnnQueue; t::Int=1)
    search(idx.invfile, ctx, q, res; t)
end
