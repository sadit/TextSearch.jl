# This file is a part of TextSearch.jl

export TextConfig

"""
    TextConfig(;
        normalization::NormalizationConfig=NormalizationConfig(),
        tokenization::TokenizationConfig=TokenizationConfig(),
        transformation::AbstractTokenTransformation=IdentityTokenTransformation()
    )

Defines a preprocessing and tokenization pipeline, composed of 3 independent stages:

- `normalization`: a [`NormalizationConfig`](@ref) (utf8 normalization, character
  removal, whitespace normalization, casing, etc.).
- `tokenization`: a [`TokenizationConfig`](@ref) (unigrams, word n-grams, and any extra
  custom [`AbstractTokenGenerator`](@ref)s).
- `transformation`: an [`AbstractTokenTransformation`](@ref) applied to every generated
  token (e.g. stemming, lemmatization, or stopword removal).
- `expand_query_synonyms`: whether query-time synonym expansion (see [`expand_synonyms!`](@ref))
  should be applied when this config's queries are searched against an index carrying a synonym
  network (e.g. a [`TextInvertedFile`](@ref) built with `synonyms=...`). Has no effect on indexing:
  documents are never expanded, only queries.

# Example

```julia
julia> cfg = TextConfig(tokenization=TokenizationConfig(nlist=[1]));

julia> collect(tokenize(cfg, "cats"))
["cats"]
```
"""
Base.@kwdef struct TextConfig
    normalization::NormalizationConfig = NormalizationConfig()
    tokenization::TokenizationConfig = TokenizationConfig()
    transformation::AbstractTokenTransformation = IdentityTokenTransformation()
    expand_query_synonyms::Bool = false
end

function TextConfig(c::TextConfig;
        normalization::NormalizationConfig=c.normalization,
        tokenization::TokenizationConfig=c.tokenization,
        transformation::AbstractTokenTransformation=c.transformation,
        expand_query_synonyms::Bool=c.expand_query_synonyms
    )
    TextConfig(normalization, tokenization, transformation, expand_query_synonyms)
end

function Base.show(io::IO, c::TextConfig; prefix="", indent="  ")
    println(io, prefix, "TextConfig:")
    prefix = indent * prefix
    show(io, c.normalization; prefix, indent)
    show(io, c.tokenization; prefix, indent)
    print(io, prefix, "transformation: ")
    println(io, c.transformation)
    println(io, prefix, "expand_query_synonyms: ", c.expand_query_synonyms)
end

Base.broadcastable(c::TextConfig) = (c,)
