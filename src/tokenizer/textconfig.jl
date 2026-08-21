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
  token (lemma normalization or stopword removal).

This is the corpus-independent half of a text model -- it can be written by hand with no data.
The artifacts a corpus produces (stopword sets, lemma maps, synonym networks) live in a
[`TextProfile`](@ref), which materializes the `transformation` from whichever of them it
applies. Query-time synonym expansion is likewise a profile-level decision
(`applied.synonyms`), not a flag here: it is a search-time behaviour whose data does not live
in the tokenizer.

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
