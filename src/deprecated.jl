# This file is part of TextSearch.jl

function append_items!(idx::BM25InvertedFile, corpus::AbstractVector{T}; kwargs...) where {T<:AbstractString}
    append_items!(idx, getcontext(idx), VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end

function append_items!(idx::BM25InvertedFile, corpus::AbstractVector{T}; kwargs...) where {T<:TokenizedText}
    append_items!(idx, getcontext(idx), VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end

