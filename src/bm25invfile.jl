# This file is part of TextSearch.jl

export BM25InvertedFile

"""
    struct BM25InvertedFile <: AbstractInvertedFile

An inverted index is a sparse matrix representation of with floating point weights, it supports only positive non-zero values.
This index is optimized to efficiently solve `k` nearest neighbors (cosine distance, using previously normalized vectors).

# Parameters
- `db`: database
- `lists`: posting lists (non-zero id-elements in rows)
- `freqs`: posting freqs (non-zero id-elements in rows)
- `doclens`: number of tokens per document
- `locks`: per-row locks for multithreaded construction
"""
struct BM25InvertedFile{DbType<:Union{<:AbstractDatabase,Nothing}} <: AbstractInvertedFile
    db::DbType
    textconfig::TextConfig
    voc::Vocabulary
    bm25::BM25
    voc::Vocabulary
    lists::Vector{Vector{UInt32}}  ## posting lists
    freqs::Vector{Vector{Float32}}  ##
    locks::Vector{SpinLock}
    sizes::Vector{Int32}  ## number of non zeros per bow
    doclens::Vector{Int32}  ## number of tokens per document
end

SimilaritySearch.distance(idx::BM25InvertedFile) = error("BM25 is not a metric")

function SimilaritySearch.saveindex(filename::AbstractString, index::InvFileType, meta::Dict) where {InvFileType<:AbstractInvertedFile}
    lists = SimilaritySearch.flat_adjlist(UInt32, index.lists)
    freqs = SimilaritySearch.flat_adjlist(Float32, index.freqs)
    index = InvFileType(index; lists=Vector{UInt32}[], freqs=Vector{Float32}[])
    jldsave(filename; index, meta, lists, freqs)
end

function restoreindex(index::InvFileType, meta::Dict, f) where {InvFileType<:AbstractInvertedFile}
    lists = unflat_adjlist(UInt32, f["lists"])
    freqs = unflat_adjlist(UInt32, f["freqs"])
    copy(index; lists, freqs), meta
end

BM25InvertedFile(invfile::BM25InvertedFile;
    db=invfile.db,
    textconfig=invfile.textconfig,
    voc=invfile.voc,
    bm25=invfile.bm25,
    lists=invfile.lists,
    freqs=invfile.freqs,
    locks=invfile.locks,
    sizes=invfile.sizes,
    doclens=invfile.doclens
) = BM25InvertedFile(db, textconfig, voc, bm25, lists, freqs, locks, sizes, doclens)

function BM25InvertedFile(textconfig, corpus, db=nothing)
    tok_corpus = tokenize_corpus(textconfig, corpus)
    voc = Vocabulary(tok_corpus)
    doclen = Int32[length(text) for text in tok_corpus]    
    avg_doc_len = mean(doclen)
    bm25 = BM25(avg_doc_len, length(doclen))

    BM25InvertedFile(
        db,
        textconfig,
        voc,
        bm25,
        [UInt32[] for i in 1:vocsize],
        [Float32[] for i in 1:vocsize],
        [SpinLock() for i in 1:vocsize],
        Vector{Int32}(undef, 0),
        Vector{Int32}(undef, 0),
    )
end

function InvertedFiles.internal_push!(idx::BM25InvertedFile, tokenID, docID, freq, sort)
    push!(idx.lists[tokenID], docID)
    push!(idx.freqs[tokenID], freq)
    sort && sortlastpush!(idx.lists[tokenID], idx.freqs[tokenID])
end

function InvertedFiles.internal_push_object!(idx::BM25InvertedFile, docID::Integer, obj, tol::Float64)
    nz = 0
    len = 0
    @inbounds for (tokenID, freq) in InvertedFiles.sparseiterator(obj)  # obj is a BOW-like struct
        freq < tol && continue
        nz += 1
        len += freq
        internal_push!(idx, tokenID, docID, freq, false)
    end

    push!(idx.doclens, len)
    nz
end

function InvertedFiles.internal_parallel_prepare_append!(idx::BM25InvertedFile, new_size::Integer)
    resize!(idx.sizes, new_size)
    resize!(idx.doclens, new_size)
end

function InvertedFiles.internal_parallel_finish_append_object!(idx::BM25InvertedFile, objID::Integer, nz::Integer, sumw)
    idx.sizes[objID] = nz
    idx.doclens[objID] = convert(Int32, sumw)
end
