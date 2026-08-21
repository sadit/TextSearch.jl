# This file is a part of TextSearch.jl

export AbstractTokenTransformation, IdentityTokenTransformation, transform
export IgnoreStopwords, LemmaTransformation, ChainTransformation, SnowballTokenTransformation

"""
    AbstractTokenTransformation

Abstract type for token transformation hooks applied during tokenization (see
[`transform`](@ref)). A [`TextConfig`](@ref) holds one such transformation in its
`transformation` field; it is applied to every generated token before it is pushed to
the token list, and can be used to implement stemming, lemmatization, casing rules, or
stopword removal (by returning `nothing`).
"""
abstract type AbstractTokenTransformation end

"""
    IdentityTokenTransformation()

The default, no-op [`AbstractTokenTransformation`](@ref): every token is kept unchanged.

# Example

```julia
julia> collect(tokenize(TextConfig(transformation=IdentityTokenTransformation()), "the cat sat"))
["the", "cat", "sat"]
```
"""
struct IdentityTokenTransformation <: AbstractTokenTransformation end

"""
    transform(tt::AbstractTokenTransformation, gen::AbstractTokenGenerator, tok)

Hook applied in the tokenization stage to change the input token `tok`, produced by
generator `gen` (e.g. a [`UnigramGenerator`](@ref) or [`NWordGenerator`](@ref)), if
needed. For instance, it can be used to apply stemming or any other kind of
normalization. Return `nothing` to ignore the `tok` occurrence (e.g., stop words).

The default falls through to identity for any `gen` a custom
[`AbstractTokenTransformation`](@ref) doesn't specialize, so adding a new
[`AbstractTokenGenerator`](@ref) kind never requires touching existing
transformations. The built-in generators dispatch to the legacy
`transform_unigram`/`transform_nword` names for backward compatibility with
transformations written against those.

# Example

```julia
julia> transform(IdentityTokenTransformation(), UnigramGenerator(), "cat")
"cat"
```
"""
transform(tt::AbstractTokenTransformation, gen::AbstractTokenGenerator, tok) = legacy_transform(gen, tt, tok)

legacy_transform(::UnigramGenerator, tt, tok) = transform_unigram(tt, tok)
legacy_transform(::NWordGenerator, tt, tok) = transform_nword(tt, tok)
legacy_transform(::AbstractTokenGenerator, tt, tok) = tok

"""
    transform_unigram(::AbstractTokenTransformation, tok)

Legacy per-kind hook kept for backward compatibility; prefer specializing
[`transform`](@ref) on [`AbstractTokenGenerator`](@ref) subtypes in new code. Called by
the default [`transform`](@ref) method for [`UnigramGenerator`](@ref) tokens.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_unigram(::AbstractTokenTransformation, tok) = tok

"""
    transform_nword(::AbstractTokenTransformation, tok)

Legacy per-kind hook kept for backward compatibility; prefer specializing
[`transform`](@ref) on [`AbstractTokenGenerator`](@ref) subtypes in new code. Called by
the default [`transform`](@ref) method for [`NWordGenerator`](@ref) tokens.
Return `nothing` to ignore the `tok` occurence (e.g., stop words).
"""
transform_nword(::AbstractTokenTransformation, tok) = tok


### some transformations

"""
    IgnoreStopwords(stopwords::Set{String})

An [`AbstractTokenTransformation`](@ref) that discards unigrams found in `stopwords`
(returns `nothing` for them, causing the tokenizer to drop the token) and passes every
other token through unchanged.

# Example

```julia
julia> cfg = TextConfig(transformation=IgnoreStopwords(Set(["the", "a"])));

julia> collect(tokenize(cfg, "the cat sat"))
["cat", "sat"]
```
"""
struct IgnoreStopwords <: AbstractTokenTransformation
    stopwords::Set{String}
end

function transform_unigram(tt::IgnoreStopwords, tok)
    tok in tt.stopwords ? nothing : tok
end

"""
    LemmaTransformation(lemmas::AbstractDict{<:AbstractString,<:AbstractString})

An [`AbstractTokenTransformation`](@ref) that rewrites each unigram to its lemma, e.g. a
mapping produced by [`lemma_clusters`](@ref). Only non-identity entries need to be
present: a token absent from `lemmas` is its own lemma and passes through unchanged.

A lemma is a *normalization*, not an expansion, so unlike query-time synonym expansion
(see [`expand_synonyms!`](@ref), which must only ever touch queries) it belongs on both
sides. Putting it here rather than in each consumer is what makes that automatic: a
`Vocabulary` built under this config is already lemmatized, so its `occs`/`ndocs` (and
therefore the idf a [`VectorModel`](@ref) derives) count a whole inflection family
together, and documents and queries meet on the same footing without either side having
to remember to apply the mapping.

The mapping is keyed by single words, and only `transform_unigram` is defined -- but that
is enough to cover n-grams too, because the n-gram generators consume the already
transformed word stream rather than the raw text. So an `nlist=[2]` config over
`"las casas rojas"` yields the bigrams `"las casa"` and `"casa rojas"`: the lemma
propagates into each n-gram word by word, which is what a normalization should do. (This
mirrors [`IgnoreStopwords`](@ref), whose dropped words are likewise absent from the
n-grams built around them.)

Chain it *before* [`IgnoreStopwords`](@ref), not after (see [`ChainTransformation`](@ref),
which applies its steps in order). The steps are not commutative and the wrong order
silently reintroduces stopwords: with `IgnoreStopwords(Set(["la"]))` first, `"las"` is not
in the set, survives the filter, and is only then rewritten to `"la"` -- so the stopword
lands in the vocabulary through the back door. Lemmatizing first means the filter sees the
form that will actually be indexed:

```julia
julia> lt = LemmaTransformation(Dict("las" => "la", "casas" => "casa", "rojas" => "roja"));

julia> cfg(ch) = TextConfig(tokenization=TokenizationConfig(nlist=[1]), transformation=ch);

julia> collect(tokenize(cfg(ChainTransformation([IgnoreStopwords(Set(["la"])), lt])), "las casas rojas"))
["la", "casa", "roja"]      # wrong: "la" is a stopword

julia> collect(tokenize(cfg(ChainTransformation([lt, IgnoreStopwords(Set(["la"]))])), "las casas rojas"))
["casa", "roja"]
```

This works out because a lemma is always one of its family's own surface forms, so a
stopword set collected from unlemmatized text already contains the lemma of any family
that is a stopword family.

# Example

```julia
julia> cfg = TextConfig(tokenization=TokenizationConfig(nlist=[1]),
                        transformation=LemmaTransformation(Dict("casas" => "casa")));

julia> collect(tokenize(cfg, "las casas rojas"))
["las", "casa", "rojas"]
```
"""
struct LemmaTransformation <: AbstractTokenTransformation
    lemmas::Dict{String,String}
end

LemmaTransformation(lemmas::AbstractDict{<:AbstractString,<:AbstractString}) =
    LemmaTransformation(Dict{String,String}(String(k) => String(v) for (k, v) in lemmas))

transform_unigram(tt::LemmaTransformation, tok) = get(tt.lemmas, tok, tok)

"""
    ChainTransformation(list::AbstractVector{<:AbstractTokenTransformation})

Holds an ordered sequence of [`AbstractTokenTransformation`](@ref)s, applied one after
the other over each token via [`transform`](@ref); if any step returns `nothing` the
token is dropped and the remaining steps are skipped.

# Example

```julia
julia> ct = ChainTransformation([IdentityTokenTransformation(), IgnoreStopwords(Set(["the"]))]);

julia> collect(tokenize(TextConfig(transformation=ct), "the cat sat"))
["cat", "sat"]
```
"""
struct ChainTransformation <: AbstractTokenTransformation
    list::AbstractVector{<:AbstractTokenTransformation}
end

function transform(ct::ChainTransformation, gen::AbstractTokenGenerator, tok)
    for tt in ct.list
        tok === nothing && return nothing
        tok = transform(tt, gen, tok)
    end

    tok
end

"""
    SnowballTokenTransformation(stemmer)

An [`AbstractTokenTransformation`](@ref) that stems unigrams using `stemmer` (typically
a `Snowball.Stemmer`). The struct itself has no dependency on the `Snowball`/`Languages`
packages; loading them (`using Snowball, Languages`) activates the
`TextSearchSnowballExt` package extension, which implements the actual stemming
(`transform_unigram`) and adds the `SnowballTokenTransformation(lang::Languages.Language)`
convenience constructor.
"""
struct SnowballTokenTransformation{S} <: AbstractTokenTransformation
    stemmer::S
end
