# This file is a part of TextSearch.jl

export AbstractTokenGenerator, UnigramGenerator, NWordGenerator, needs_unigrams, tokentag

"""
    AbstractTokenGenerator

Abstract type for a single token-producing strategy inside a [`TokenizationConfig`](@ref)'s
`generators` list. `TokenizationConfig`'s `nlist` keyword argument is convenience sugar
that builds the built-in generators below; passing `generators` directly (or mixing in
your own `AbstractTokenGenerator` subtype) is how new kinds of tokens can be added
without touching `TokenizationConfig` or the tokenizer's dispatch logic (e.g. character
q-grams, skip-grams, or collocations, none of which are built-in anymore).

Implementing a new generator kind requires:
- a struct `<: AbstractTokenGenerator` holding whatever parameters it needs;
- [`needs_unigrams`](@ref) (defaults to `false`) if it needs the shared word-level
  `unigrams` basis computed first;
- [`TextSearch.Tokenizer.generate!`](@ref) performing the actual token production;
- optionally [`tokentag`](@ref) (defaults to `nothing`, i.e. untagged) for the
  single-character tag appended to each token when `mark_token_type=true`.

The [`transform`](@ref) hook already dispatches on `AbstractTokenGenerator` with an
identity default, so a new generator kind is usable with any existing
[`AbstractTokenTransformation`](@ref) without further changes.
"""
abstract type AbstractTokenGenerator end

"""
    needs_unigrams(gen::AbstractTokenGenerator)::Bool

Whether `gen` needs the shared word-level `unigrams` basis (see [`unigrams`](@ref))
computed before it runs. Defaults to `false`; [`NWordGenerator`](@ref) and
[`UnigramGenerator`](@ref) override it to `true`.
"""
needs_unigrams(::AbstractTokenGenerator) = false

"""
    tokentag(gen::AbstractTokenGenerator)::Union{Char,Nothing}

The single-character tag appended (as `\\ttag`) to every token `gen` produces when
`mark_token_type=true`. Defaults to `nothing` (untagged).
"""
tokentag(::AbstractTokenGenerator) = nothing

"""
    UnigramGenerator()

Emits the word-level unigrams themselves as output tokens (untagged). Built from
[`TokenizationConfig`](@ref)'s `nlist` keyword argument when it contains `1`. Every other
generator that needs the word-level basis (see [`needs_unigrams`](@ref)) triggers the
same underlying computation regardless of whether `UnigramGenerator` is present —
this generator only controls whether the plain words also appear in the output.
"""
struct UnigramGenerator <: AbstractTokenGenerator end
needs_unigrams(::UnigramGenerator) = true

"""
    NWordGenerator(q)

Produces word `q`-grams (`q > 1`) from the shared unigram basis (tagged `'n'`). Built
from [`TokenizationConfig`](@ref)'s `nlist` keyword argument for every entry other than `1`.
"""
struct NWordGenerator <: AbstractTokenGenerator
    q::Int8
end
needs_unigrams(::NWordGenerator) = true
tokentag(::NWordGenerator) = 'n'
