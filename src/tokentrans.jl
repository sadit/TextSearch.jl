# This file is a part of TextSearch.jl

export TextConfig, Skipgram, AbstractTokenTransformation, IdentityTokenTransformation
export IgnoreStopwords, ChainTransformation

abstract type AbstractTokenTransformation end
struct IdentityTokenTransformation <: AbstractTokenTransformation end

"""
    transform_unigram(::AbstractTokenTransformation, tok)

Hook applied in the tokenization stage to change the input token `tok` if needed.
For instance, it can be used to apply stemming or any other kind of normalization.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_unigram(::AbstractTokenTransformation, tok) = tok

"""
    transform_nword(::AbstractTokenTransformation, tok)

Hook applied in the tokenization stage to change the input token `tok` if needed.
For instance, it can be used to apply stemming or any other kind of normalization.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_nword(::AbstractTokenTransformation, tok) = tok

"""
    transform_qgram(::AbstractTokenTransformation, tok)

Hook applied in the tokenization stage to change the input token `tok` if needed.
For instance, it can be used to apply stemming or any other kind of normalization.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_qgram(::AbstractTokenTransformation, tok) = tok

"""
    transform_collocation(::AbstractTokenTransformation, tok)

Hook applied in the tokenization stage to change the input token `tok` if needed.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_collocation(::AbstractTokenTransformation, tok) = tok

"""
    transform_skipgram(::AbstractTokenTransformation, tok)

Hook applied in the tokenization stage to change the input token `tok` if needed.
For instance, it can be used to apply stemming or any other kind of normalization.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_skipgram(::AbstractTokenTransformation, tok) = tok


### some transformations


struct IgnoreStopwords <: AbstractTokenTransformation
    stopwords::Set{String}
end

function TextSearch.transform_unigram(tt::IgnoreStopwords, tok)
    tok in tt.stopwords ? nothing : tok
end

struct ChainTransformation <: AbstractTokenTransformation
    list::AbstractVector{<:AbstractTokenTransformation}    
end
