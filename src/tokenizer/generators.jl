# This file is a part of TextSearch.jl

export AbstractTokenGenerator, UnigramGenerator, NWordGenerator, QgramGenerator, needs_unigrams, tokentag

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

A new generator kind works with any [`TokenPipeline`](@ref) without further changes: the
pipeline's stages act on whatever tokens a generator produces. This is also the right home for
anything that changes how text becomes tokens -- splitting `getUserName` into three words,
keeping `H2O` whole -- since a generator sees the word stream and may emit one token or several,
while the pipeline is strictly per-token and data-driven.
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

"""
    QgramGenerator(q)

Produces character `q`-grams (tagged `'q'`) over the whole normalized text, blanks included:
they are not sub-words, since a window can span a word boundary (`"o de"` is a 4-gram of
`"todo de"`), and the boundary blanks the normalizer adds make word starts and ends visible
(`" to"`, `"do "`). Runs of blanks count as a single blank, so layout does not produce q-grams
of its own. Each field of a multi-field document is its own text, so no q-gram spans two
fields. Texts shorter than `q` produce none.

Aimed at document-vs-document encodings (classification, clustering, dense encoders built on
top) rather than short queries. Several lengths are several generators, and they combine with
word tokens freely:

```julia
julia> tc = TextConfig(tokenization=TokenizationConfig(generators=[QgramGenerator(3)]));

julia> collect(tokenize(tc, "ab c"))
4-element Vector{String}:
 " ab\tq"
 "ab \tq"
 "b c\tq"
 " c \tq"
```

The tag is what keeps the 3-gram `que` (from `porque`) and the word `que` apart when both are
generated, and what keeps word-level stopwords and lemmas from matching a q-gram; with
`mark_token_type=false` they are the same token.
"""
struct QgramGenerator <: AbstractTokenGenerator
    q::Int8

    function QgramGenerator(q::Integer)
        q >= 1 || throw(ArgumentError("QgramGenerator: q must be positive, got $q"))
        new(q)
    end
end
tokentag(::QgramGenerator) = 'q'
