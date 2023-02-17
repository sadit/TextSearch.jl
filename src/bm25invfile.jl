# This file is part of TextSearch.jl

export BM25InvertedFile
using SimilaritySearch.AdjacencyLists
using Intersections
using InvertedFiles
using StatsBase

"""
    struct BM25InvertedFile <: AbstractInvertedFile

# Parameters
"""
struct BM25InvertedFile{DbType<:Union{<:AbstractDatabase,Nothing}} <: AbstractInvertedFile
    db::DbType
    textconfig::TextConfig
    voc::Vocabulary
    bm25::BM25
    adj::AdjacencyList{IWeightedEndPoint}
    doclens::Vector{Int32}  ## number of tokens per document
end

SimilaritySearch.length(invfile::BM25InvertedFile) = length(invfile.doclens)
SimilaritySearch.database(invfile::BM25InvertedFile) = invfile.db
SimilaritySearch.distance(::BM25InvertedFile) = error("BM25InvertedFile is not a metric index")

function SimilaritySearch.saveindex(filename::AbstractString, index::BM25InvertedFile, meta::Dict)
    index = InvFileType(index; adj=StaticAdjacencyList(index.adj))
    jldsave(filename; index, meta)
end

function restoreindex(index::BM25InvertedFile, meta::Dict, f)
    copy(index; adj=AdjacencyList(index.adj)), meta
end

BM25InvertedFile(invfile::BM25InvertedFile;
    db=invfile.db,
    textconfig=invfile.textconfig,
    voc=invfile.voc,
    bm25=invfile.bm25,
    adj=index.adj,
    doclens=invfile.doclens
) = BM25InvertedFile(db, textconfig, voc, bm25, adj, doclens)

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
        AdjacencyList(IWeightedEndPoint; n=vocsize(voc)),
        Vector{Int32}(undef, 0),
    )
end

function BM25InvertedFile(textconfig::TextConfig, corpus, db=nothing)
    BM25InvertedFile(nothing, textconfig, corpus, db)
end

function SimilaritySearch.append_items!(idx::BM25InvertedFile, corpus::AbstractVector{T}) where {T<:AbstractString}
    append_items!(idx, VectorDatabase(vectorize_corpus(idx.voc, idx.textconfig, corpus)))
end

function SimilaritySearch.push_item!(idx::BM25InvertedFile, doc::AbstractString)
    push_item!(idx, vectorize(idx.voc, idx.textconfig, corpus))
end

function InvertedFiles.internal_push!(idx::BM25InvertedFile, tokenID, objID, freq, sort)
    if sort
        add_edge!(idx.adj, tokenID, IWeightedEndPoint(objID, freq), IdOrder)
    else
        add_edge!(idx.adj, tokenID, IWeightedEndPoint(objID, freq), nothing)
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
