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
end

function TextConfig(c::TextConfig;
        normalization::NormalizationConfig=c.normalization,
        tokenization::TokenizationConfig=c.tokenization,
        transformation::AbstractTokenTransformation=c.transformation
    )
    TextConfig(normalization, tokenization, transformation)
end

function Base.show(io::IO, c::TextConfig; prefix="", indent="  ")
    println(io, prefix, "TextConfig:")
    prefix = indent * prefix
    show(io, c.normalization; prefix, indent)
    show(io, c.tokenization; prefix, indent)
    print(io, prefix, "transformation: ")
    println(io, c.transformation)
end

Base.broadcastable(c::TextConfig) = (c,)
