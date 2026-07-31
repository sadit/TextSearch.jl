# This file is a part of TextSearch.jl

export TextConfig, Skipgram, AbstractTokenTransformation, IdentityTokenTransformation
export IgnoreStopwords, ChainTransformation

"""
    AbstractTokenTransformation

Abstract type for token transformation hooks applied during tokenization (see
[`transform_unigram`](@ref), [`transform_nword`](@ref), [`transform_qgram`](@ref),
[`transform_skipgram`](@ref), and [`transform_collocation`](@ref)). A [`TextConfig`](@ref)
holds one such transformation in its `tt` field; it is applied to every generated token
before it is pushed to the token list, and can be used to implement stemming, casing
rules, or stopword removal (by returning `nothing`).
"""
abstract type AbstractTokenTransformation end

"""
    IdentityTokenTransformation()

The default, no-op [`AbstractTokenTransformation`](@ref): every token is kept unchanged.

# Example

```julia
julia> collect(tokenize(TextConfig(tt=IdentityTokenTransformation()), "the cat sat"))
["the", "cat", "sat"]
```
"""
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

"""
    IgnoreStopwords(stopwords::Set{String})

An [`AbstractTokenTransformation`](@ref) that discards unigrams found in `stopwords`
(returns `nothing` for them, causing the tokenizer to drop the token) and passes every
other token through unchanged.

# Example

```julia
julia> cfg = TextConfig(nlist=[1], tt=IgnoreStopwords(Set(["the", "a"])));

julia> collect(tokenize(cfg, "the cat sat"))
["cat", "sat"]
```
"""
struct IgnoreStopwords <: AbstractTokenTransformation
    stopwords::Set{String}
end

function TextSearch.transform_unigram(tt::IgnoreStopwords, tok)
    tok in tt.stopwords ? nothing : tok
end

"""
    ChainTransformation(list::AbstractVector{<:AbstractTokenTransformation})

Holds an ordered sequence of [`AbstractTokenTransformation`](@ref)s, meant to be applied
one after the other over each token.

!!! note
    `transform_unigram`/`transform_nword`/etc. are not yet specialized for
    `ChainTransformation`; it currently falls back to the identity behavior.

# Example

```julia
julia> ct = ChainTransformation([IdentityTokenTransformation(), IgnoreStopwords(Set(["the"]))]);

julia> ct isa AbstractTokenTransformation
true
```
"""
struct ChainTransformation <: AbstractTokenTransformation
    list::AbstractVector{<:AbstractTokenTransformation}
end
