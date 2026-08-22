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
- `language`: which language this configuration is *for*, as an ISO 639-1 symbol (`:es`,
  `:pt`, `:en`, ...) or `:unknown`.

`language` currently changes **nothing** about tokenization -- it is recorded, not acted on.
It lives here rather than in a [`TextProfile`](@ref) because the language is not something a
corpus produces, it is what *selects* the policy: whether to strip diacritics, whether
suffix-anchored morphology fits, whether function words arrive as free tokens at all. Those
are decisions the tokenizer will eventually make from this field, and a field the tokenizer
must read belongs in the tokenizer's config.

It also earns its keep immediately: [`merge_profiles`](@ref) compares policy for equality, so
declaring the language is what stops a Spanish profile from silently merging with a Portuguese
one -- their normalization and tokenization are identical, so nothing else distinguishes them.
A **detected** language distribution ("this corpus turned out 87% Spanish") would be an
observation about data, and would belong in a profile's lineage instead.

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
    language::Symbol = :unknown
end

function TextConfig(c::TextConfig;
        normalization::NormalizationConfig=c.normalization,
        tokenization::TokenizationConfig=c.tokenization,
        transformation::AbstractTokenTransformation=c.transformation,
        language::Symbol=c.language
    )
    TextConfig(normalization, tokenization, transformation, language)
end

function Base.show(io::IO, c::TextConfig; prefix="", indent="  ")
    println(io, prefix, "TextConfig:")
    prefix = indent * prefix
    show(io, c.normalization; prefix, indent)
    show(io, c.tokenization; prefix, indent)
    print(io, prefix, "transformation: ")
    println(io, c.transformation)
    println(io, prefix, "language: ", c.language)
end

Base.broadcastable(c::TextConfig) = (c,)
