# This file is a part of TextSearch.jl

export TokenPipeline, apply_pipeline, isidentity

"""
    TokenPipeline(; lemmas=nothing, stopwords=nothing)

The per-token stages of a [`TextConfig`](@ref), as plain data in a fixed order rather than an
open set of composable hooks.

    lemmas      rewrite a token to its lemma      `Dict{String,String}`
    stopwords   drop a token entirely             `Set{String}`

`nothing` means the stage does not run.

# Why a fixed pipeline and not composable transformations

This replaced an `AbstractTokenTransformation` hierarchy (`IgnoreStopwords`,
`LemmaTransformation`, `ChainTransformation`, plus a `transform` hook dispatching on both the
transformation and the token generator). That design was right when TextSearch was more open, and
by the time it was removed it held exactly two real stages, both of which are *data* rather than
behaviour: a set to filter by and a map to rewrite through. A generic mechanism for two known
things bought nothing and cost three specific problems.

**Order.** The stages are not commutative and the wrong order fails silently. With the stopword
filter first, `"las"` is not in a set containing `"la"`, survives the filter, and is only then
rewritten to `"la"` -- so the stopword lands in the vocabulary through the back door. This was
documented backwards once and only measurement caught it. Here the order is in the code, once, and
there is nowhere else to express it.

**Per-type dispatch.** `merge_profiles` compared transformations through a method per type, and the
missing method for `LemmaTransformation` made it reject two profiles carrying *identical* lemma maps
as incompatible. Comparing two `TokenPipeline`s is comparing two fields and cannot have a missing
method.

**Cost.** `ChainTransformation`'s field was typed `AbstractVector{<:AbstractTokenTransformation}`,
which is not concrete, so every step of every token went through a dynamic dispatch. Measured on
120,000 Spanish Wikipedia paragraphs (71.0M characters): lemmas+stopwords chained took 14.65s and
3.62 GB against 10.12s and 2.61 GB for the stopword filter alone -- 45% more time and a gigabyte
more garbage for one extra dictionary lookup per token.

# Where extensibility lives now

Not here. A new *kind of token* -- character q-grams, skip-grams, collocations, chemical formulas,
splitting `getUserName` into three words -- is a [`AbstractTokenGenerator`](@ref) in
[`TokenizationConfig`](@ref)'s `generators` list, which is the documented extension point and the
right place for it: generators see the word stream, so they can emit one token or several.

What has no home here is an *algorithmic* per-token rewrite or filter that cannot be expressed as
data -- a stemmer, say, which is exactly what the removed Snowball extension was. Adding one means
adding a named field with a documented position in the order, which is cheaper than the mechanism
this replaced: that needed a new type, a `transform_unigram` method, a comparison method (the one
that was forgotten), and a decision about where it chained.

# Example

```julia
julia> p = TokenPipeline(lemmas=Dict("casas" => "casa", "rojas" => "roja"),
                         stopwords=Set(["la"]));

julia> cfg = TextConfig(tokenization=TokenizationConfig(nlist=[1]), pipeline=p);

julia> collect(tokenize(cfg, "las casas rojas"))
["casa", "roja"]
```

`"las"` becomes `"la"` and is then dropped, which is the order this type exists to guarantee.
"""
struct TokenPipeline
    lemmas::Union{Nothing,Dict{String,String}}
    stopwords::Union{Nothing,Set{String}}
end

TokenPipeline(; lemmas=nothing, stopwords=nothing) =
    TokenPipeline(_as_lemmas(lemmas), _as_stopwords(stopwords))

_as_lemmas(::Nothing) = nothing
_as_lemmas(d::Dict{String,String}) = isempty(d) ? nothing : d
_as_lemmas(d::AbstractDict) =
    isempty(d) ? nothing : Dict{String,String}(String(k) => String(v) for (k, v) in d)

_as_stopwords(::Nothing) = nothing
_as_stopwords(s::Set{String}) = isempty(s) ? nothing : s
_as_stopwords(s) = (t = Set{String}(String(x) for x in s); isempty(t) ? nothing : t)

"""
    TokenPipeline(p::TokenPipeline; lemmas, stopwords)

Copy constructor: rebuilds `p` overriding only the named stages. Pass `nothing` to turn a stage
off.
"""
TokenPipeline(p::TokenPipeline; lemmas=p.lemmas, stopwords=p.stopwords) =
    TokenPipeline(_as_lemmas(lemmas), _as_stopwords(stopwords))

Base.:(==)(a::TokenPipeline, b::TokenPipeline) =
    a.lemmas == b.lemmas && a.stopwords == b.stopwords

"whether `p` runs no stage at all, i.e. every token passes through unchanged"
isidentity(p::TokenPipeline) = p.lemmas === nothing && p.stopwords === nothing

function Base.show(io::IO, p::TokenPipeline)
    print(io, "TokenPipeline(")
    p.lemmas === nothing ? print(io, "lemmas=nothing") : print(io, "lemmas=", length(p.lemmas), " entries")
    print(io, ", ")
    p.stopwords === nothing ? print(io, "stopwords=nothing") : print(io, "stopwords=", length(p.stopwords), " tokens")
    print(io, ")")
end

"""
    apply_pipeline(p::TokenPipeline, tok) -> Union{Nothing,String}

Runs `p`'s stages over one token, in the fixed order (lemma rewrite, then the stopword filter),
returning `nothing` when the token is dropped. On the hot path: an inactive stage is a `=== nothing`
check the compiler can hoist, so an identity pipeline costs two comparisons per token.
"""
@inline function apply_pipeline(p::TokenPipeline, tok)
    lem = p.lemmas
    lem === nothing || (tok = get(lem, tok, tok))
    sw = p.stopwords
    sw === nothing || (tok in sw && return nothing)
    tok
end
