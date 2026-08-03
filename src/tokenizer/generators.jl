# This file is a part of TextSearch.jl

export Skipgram, AbstractTokenGenerator, QGramGenerator, UnigramGenerator, NWordGenerator,
       SkipgramGenerator, CollocationGenerator, needs_unigrams, tokentag

"""
    Skipgram(qsize, skip)

A skipgram is a kind of tokenization where `qsize` words having `skip` separation are used as a single token.

# Example

```julia
julia> collect(tokenize(TextConfig(slist=[Skipgram(2, 1)]), "the cat sat down"))
["the sat\ts", "cat down\ts"]
```
"""
struct Skipgram
    qsize::Int8
    skip::Int8
end

Base.isless(a::Skipgram, b::Skipgram) = isless((a.qsize, a.skip), (b.qsize, b.skip))
Base.isequal(a::Skipgram, b::Skipgram) = a.qsize == b.qsize && a.skip == b.skip

"""
    AbstractTokenGenerator

Abstract type for a single token-producing strategy inside a [`TextConfig`](@ref)'s
`generators` list. `TextConfig`'s `qlist`/`nlist`/`slist`/`collocations` keyword
arguments are convenience sugar that build the built-in generators below; passing
`generators` directly (or mixing in your own `AbstractTokenGenerator` subtype) is how
new kinds of tokens can be added without touching `TextConfig` or the tokenizer's
dispatch logic.

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
computed before it runs. Defaults to `false`; [`NWordGenerator`](@ref),
[`SkipgramGenerator`](@ref), [`CollocationGenerator`](@ref), and [`UnigramGenerator`](@ref)
override it to `true`. [`QGramGenerator`](@ref) operates directly on the normalized text
and does not need it.
"""
needs_unigrams(::AbstractTokenGenerator) = false

"""
    tokentag(gen::AbstractTokenGenerator)::Union{Char,Nothing}

The single-character tag appended (as `\\ttag`) to every token `gen` produces when
`mark_token_type=true`. Defaults to `nothing` (untagged).
"""
tokentag(::AbstractTokenGenerator) = nothing

"""
    QGramGenerator(q)

Produces character `q`-grams from the normalized text (tagged `'q'`). Built from
[`TextConfig`](@ref)'s `qlist` keyword argument.
"""
struct QGramGenerator <: AbstractTokenGenerator
    q::Int8
end
tokentag(::QGramGenerator) = 'q'

"""
    UnigramGenerator()

Emits the word-level unigrams themselves as output tokens (untagged). Built from
[`TextConfig`](@ref)'s `nlist` keyword argument when it contains `1`. Every other
generator that needs the word-level basis (see [`needs_unigrams`](@ref)) triggers the
same underlying computation regardless of whether `UnigramGenerator` is present —
this generator only controls whether the plain words also appear in the output.
"""
struct UnigramGenerator <: AbstractTokenGenerator end
needs_unigrams(::UnigramGenerator) = true

"""
    NWordGenerator(q)

Produces word `q`-grams (`q > 1`) from the shared unigram basis (tagged `'n'`). Built
from [`TextConfig`](@ref)'s `nlist` keyword argument for every entry other than `1`.
"""
struct NWordGenerator <: AbstractTokenGenerator
    q::Int8
end
needs_unigrams(::NWordGenerator) = true
tokentag(::NWordGenerator) = 'n'

"""
    SkipgramGenerator(skipgram::Skipgram)

Produces skip-grams from the shared unigram basis (tagged `'s'`). Built from
[`TextConfig`](@ref)'s `slist` keyword argument.
"""
struct SkipgramGenerator <: AbstractTokenGenerator
    skipgram::Skipgram
end
needs_unigrams(::SkipgramGenerator) = true
tokentag(::SkipgramGenerator) = 's'

"""
    CollocationGenerator(window)

Produces word collocations within `window` from the shared unigram basis (tagged
`'c'`). Built from [`TextConfig`](@ref)'s `collocations` keyword argument.
"""
struct CollocationGenerator <: AbstractTokenGenerator
    window::Int8
end
needs_unigrams(::CollocationGenerator) = true
tokentag(::CollocationGenerator) = 'c'
