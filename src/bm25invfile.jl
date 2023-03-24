# This file is part of TextSearch.jl

export BM25InvertedFile, search, filter_lists!, append_items!, push_item!

using SimilaritySearch.AdjacencyLists
import SimilaritySearch: search, append_items!, push_item!, database, distance
import SimilaritySearch: serializeindex, restoreindex

using Intersections
using InvertedFiles
using StatsBase

"""
    struct BM25InvertedFile <: AbstractInvertedFile

# Parameters
"""
struct BM25InvertedFile{
                        DbType<:Union{<:AbstractDatabase,Nothing},
                        AdjType<:AbstractAdjacencyList} <: AbstractInvertedFile
    db::DbType
    textconfig::TextConfig
    voc::Vocabulary
    bm25::BM25
    adj::AdjType
    doclens::Vector{Int32}  ## number of tokens per document
end

Base.length(invfile::BM25InvertedFile) = length(invfile.doclens)
database(invfile::BM25InvertedFile) = invfile.db
distance(::BM25InvertedFile) = error("BM25InvertedFile is not a metric index")

BM25InvertedFile(invfile::BM25InvertedFile;
    db=invfile.db,
    textconfig=invfile.textconfig,
    voc=invfile.voc,
    bm25=invfile.bm25,
    adj=invfile.adj,
    doclens=invfile.doclens
) = BM25InvertedFile(db, textconfig, voc, bm25, adj, doclens)

Base.copy(I::BM25InvertedFile; kwargs...) = BM25InvertedFile(I; kwargs...)

"""
    BM25InvertedFile(textconfig, corpus, db=nothing)

Fits the vocabulary and BM25 score, it also creates the associated inverted file structure.
NOTE: The corpus is not indexed since here we expect a relatively small sample of documents here and then an indexing stage on a larger corpus.
"""
function BM25InvertedFile(filter_tokens_::Union{Nothing,Function}, textconfig::TextConfig, corpus, db=nothing)
    tok_corpus = tokenize_corpus(textconfig, corpus)
    voc = Vocabulary(tok_corpus)
    if filter_tokens_ !== nothing
        voc = filter_tokens(filter_tokens_, voc)
    end
    doclen = Int32[length(text) for text in tok_corpus]
    avg_doc_len = mean(doclen)
    bm25 = BM25(avg_doc_len, length(doclen))

    BM25InvertedFile(
        db,
        textconfig,
        voc,
        bm25,
        AdjacencyList(IdIntWeight; n=vocsize(voc)),
        Vector{Int32}(undef, 0),
    )
end

function BM25InvertedFile(textconfig::TextConfig, corpus, db=nothing)
    BM25InvertedFile(nothing, textconfig, corpus, db)
end

function filter_lists!(
        idx::BM25InvertedFile;
        list_min_length_for_checking::Int=96,
        list_max_allowed_length::Int=1024,
        doc_min_freq::Int=1,
        doc_max_freq::Int=128,
        always_sort::Bool=false
    )
    adj = idx.adj
    @assert adj isa AdjacencyList
    buff = IdIntWeight[]
    sizehint!(buff, list_max_allowed_length)

    for i in eachindex(adj)
        L = neighbors(adj, i)
        n = length(L)
        n == 0 && continue
        if n < list_min_length_for_checking
            always_sort && sort!(L, by=p->p.id)
            continue
        end
        empty!(buff)
        for item in L
            if doc_min_freq <= item.weight <= doc_max_freq
                push!(buff, item)
            end
        end

        sort!(buff, by=p->p.weight, rev=true)
        if length(buff) > list_max_allowed_length
            resize!(buff, list_max_allowed_length)
        end

        sort!(buff, by=p->p.id)
        resize!(L, length(buff))
        L .= buff
    end

    idx
end

function append_items!(idx::BM25InvertedFile, corpus::AbstractVector{T}) where {T<:AbstractString}
    append_items!(idx, VectorDatabase(vectorize_corpus(idx.voc, idx.textconfig, corpus)))
end

function push_item!(idx::BM25InvertedFile, doc::AbstractString)
    push_item!(idx, vectorize(idx.voc, idx.textconfig, corpus))
end

function InvertedFiles.internal_push!(idx::BM25InvertedFile, tokenID, objID, freq, sort)
    if sort
        add_edge!(idx.adj, tokenID, IdIntWeight(objID, freq), IdOrder)
    else
        add_edge!(idx.adj, tokenID, IdIntWeight(objID, freq), nothing)
    end
end

function InvertedFiles.internal_push_object!(idx::BM25InvertedFile, docID::Integer, obj, tol::Float64, sort, is_push)
    len = 0
    @inbounds for (tokenID, freq) in InvertedFiles.sparseiterator(obj)  # obj is a BOW-like struct
        freq < tol && continue
        len += freq
        InvertedFiles.internal_push!(idx, tokenID, docID, freq, sort)
    end

    if is_push
        push!(idx.doclens, len)
    else
        idx.doclens[docID] = len
    end
end

function InvertedFiles.internal_parallel_prepare_append!(idx::BM25InvertedFile, new_size::Integer)
    resize!(idx.doclens, new_size)
end
