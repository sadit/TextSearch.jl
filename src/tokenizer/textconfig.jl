# This file is a part of TextSearch.jl

export TextConfig

"""
    TextConfig(;
        normalization::NormalizationConfig=NormalizationConfig(),
        tokenization::TokenizationConfig=TokenizationConfig(),
        pipeline::TokenPipeline=TokenPipeline()
    )

Defines a preprocessing and tokenization pipeline, composed of 3 independent stages:

- `normalization`: a [`NormalizationConfig`](@ref) (utf8 normalization, character
  removal, whitespace normalization, casing, etc.).
- `tokenization`: a [`TokenizationConfig`](@ref) (unigrams, word n-grams, and any extra
  custom [`AbstractTokenGenerator`](@ref)s).
- `pipeline`: a [`TokenPipeline`](@ref) of per-token stages applied to every generated
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
The artifacts a corpus produces (stopword sets, lemma maps, query_expansion networks) live in a
[`TextProfile`](@ref), which materializes the `pipeline` from whichever of them it
applies. Query-time expansion is likewise a profile-level decision
(`applied.query_expansion`), not a flag here: it is a search-time behaviour whose data does not live
in the tokenizer.

# Two ways to say the same thing

Any setting of the two sub-configs can be given directly, so the nesting is there when a whole
sub-config is being passed around and absent when only one flag is being changed:

```julia
TextConfig(lc=false, del_diac=false)                                     # flat
TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false))  # nested, identical
```

The flat form exists because the nested one is what actually gets written, over and over, for a
single flag -- and because `nlist=[1]` was spelled out in fifty places across this repository
while already being the default. Naming a setting both ways is an error rather than one silently
winning.

# Example

```julia
julia> collect(tokenize(TextConfig(), "cats"))
["cats"]

julia> collect(tokenize(TextConfig(lc=false), "Cats"))
["Cats"]
```
"""
struct TextConfig
    normalization::NormalizationConfig
    tokenization::TokenizationConfig
    pipeline::TokenPipeline
    language::Symbol
end

const _NORMALIZATION_KEYS = fieldnames(NormalizationConfig)
const _TOKENIZATION_KEYS = fieldnames(TokenizationConfig)

"Splits flat keywords into the sub-config each one belongs to, refusing anything unknown."
function _split_config_kwargs(kwargs)
    norm = NamedTuple(k => v for (k, v) in kwargs if k in _NORMALIZATION_KEYS)
    tok = NamedTuple(k => v for (k, v) in kwargs if k in _TOKENIZATION_KEYS)
    unknown = [k for k in keys(kwargs) if !(k in _NORMALIZATION_KEYS || k in _TOKENIZATION_KEYS)]
    isempty(unknown) ||
        throw(ArgumentError("unknown TextConfig setting(s): " * join(unknown, ", ") *
                            "; normalization takes " * join(_NORMALIZATION_KEYS, ", ") *
                            " and tokenization takes " * join(_TOKENIZATION_KEYS, ", ")))
    norm, tok
end

function TextConfig(; normalization=nothing, tokenization=nothing,
                      pipeline::TokenPipeline=TokenPipeline(), language::Symbol=:unknown,
                      kwargs...)
    normkw, tokkw = _split_config_kwargs(kwargs)
    normalization === nothing || isempty(normkw) ||
        throw(ArgumentError("normalization was given both as a NormalizationConfig and as " *
                            "the keyword(s) " * join(keys(normkw), ", ")))
    tokenization === nothing || isempty(tokkw) ||
        throw(ArgumentError("tokenization was given both as a TokenizationConfig and as " *
                            "the keyword(s) " * join(keys(tokkw), ", ")))
    TextConfig(normalization === nothing ? NormalizationConfig(; normkw...) : normalization,
               tokenization === nothing ? TokenizationConfig(; tokkw...) : tokenization,
               pipeline, language)
end

# Julia's default `==` for these is field-wise `===`, and several fields are heap objects --
# `nlist`, `emojis`, the compiled regexes -- so two configs built from the same settings compared
# as unequal. `merge_profiles` needed a real comparison and wrote its own, privately; anyone else
# asking `a == b` got a wrong answer quietly. These are it, once.
Base.:(==)(a::NormalizationConfig, b::NormalizationConfig) =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(NormalizationConfig))
Base.:(==)(a::TokenizationConfig, b::TokenizationConfig) =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(TokenizationConfig))
Base.:(==)(a::TextConfig, b::TextConfig) =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(TextConfig))

function TextConfig(c::TextConfig;
        normalization=nothing,
        tokenization=nothing,
        pipeline::TokenPipeline=c.pipeline,
        language::Symbol=c.language,
        kwargs...
    )
    normkw, tokkw = _split_config_kwargs(kwargs)
    normalization === nothing || isempty(normkw) ||
        throw(ArgumentError("normalization was given both as a NormalizationConfig and as " *
                            "the keyword(s) " * join(keys(normkw), ", ")))
    tokenization === nothing || isempty(tokkw) ||
        throw(ArgumentError("tokenization was given both as a TokenizationConfig and as " *
                            "the keyword(s) " * join(keys(tokkw), ", ")))
    # rebuilt only when something is actually being overridden, so `TextConfig(c)` keeps c's own
    # sub-config objects rather than equal-but-distinct copies
    normalization = normalization !== nothing ? normalization :
                    isempty(normkw) ? c.normalization : NormalizationConfig(c.normalization; normkw...)
    tokenization = tokenization !== nothing ? tokenization :
                   isempty(tokkw) ? c.tokenization : TokenizationConfig(c.tokenization; tokkw...)
    TextConfig(normalization, tokenization, pipeline, language)
end

function Base.show(io::IO, c::TextConfig; prefix="", indent="  ")
    println(io, prefix, "TextConfig:")
    prefix = indent * prefix
    show(io, c.normalization; prefix, indent)
    show(io, c.tokenization; prefix, indent)
    print(io, prefix, "pipeline: ")
    println(io, c.pipeline)
    println(io, prefix, "language: ", c.language)
end

Base.broadcastable(c::TextConfig) = (c,)
