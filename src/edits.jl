# This file is a part of TextSearch.jl

export EditIndex, derive_edits, edit_candidates

# ── edit-distance correction ─────────────────────────────────────────────────
#
# `derive_variants` bridges the spellings a fold can *compute* or a map can *store*: case and
# diacritics. What it cannot reach is a token that is simply mistyped -- `guerar` for `guerra`,
# a transposition no fold produces -- and such a token contributes nothing at all, since
# `bagofwords!`/`vectorize!` skip an id of 0 in silence.
#
# This file is the third source of candidates, and the only one that guesses. It is kept apart
# from `variants.jl` for that reason: a fold is a deterministic claim about orthography and a
# distance-1 neighbour is a probabilistic claim about intent, and `ResolvedToken`'s `:derived` /
# `:variant` / `:edit` reasons exist so a consumer can tell a person which of the two happened.

"""
    EditIndex(index, context, ids, minlength)

A vocabulary indexed for Damerau-Levenshtein lookup, as [`derive_edits`](@ref) builds it.

- `index`: a [`SimilaritySearch.BKT`](@ref) over the indexed tokens as `Vector{Char}`.
- `context`: the search context the BK-tree is queried with.
- `ids`: `database position -> vocabulary id`, since the indexed set may be a subset.
- `minlength`: typed tokens shorter than this are not looked up at all.

Like the variant map, this is **derived, never stored**: it is a pure function of the
vocabulary, so a profile carrying one would be carrying a second copy of `vocabulary.json`.
Building it costs 2.13s over 60,636 Spanish tokens on 8 threads, which is a per-index cost, not
a per-query one.

Tokens are indexed as `Vector{Char}` rather than `String` on purpose: `Dist.Seqs.*` index their
arguments positionally (`a[i]`), which on a `String` means *byte* offsets and throws
`StringIndexError` on any multi-byte character -- and a real Spanish vocabulary is full of them.
This is the same reason `_morphology_metric` collects its tokens in `lemmas.jl`.

# Thread safety

The distance is the plain `Dist.Seqs.DamerauLevenshtein()`, whose scratch is empty and therefore
allocated per call rather than shared, so concurrent lookups compute correct distances. The one
piece of shared mutable state is `context`'s distance-evaluation counter, which
`SimilaritySearch.add_distance_evaluations!` increments non-atomically; concurrent queries can
therefore lose counts from that *statistic*. No result depends on it. Give each concurrent
caller its own `EditIndex` if the count matters.
"""
struct EditIndex{IndexType,ContextType}
    index::IndexType
    context::ContextType
    ids::Vector{UInt32}
    minlength::Int
end

Base.length(ei::EditIndex) = length(ei.ids)

Base.show(io::IO, ei::EditIndex) =
    print(io, "EditIndex(", length(ei.ids), " tokens, minlength=", ei.minlength, ")")

"""
    derive_edits(voc::Vocabulary; minlength=4, min_ndocs=1, verbose=false) -> EditIndex

Indexes `voc`'s tokens for distance-1 Damerau-Levenshtein lookup, so
[`resolve_query_tokens`](@ref) can correct a mistyped query token.

The BK-tree is keyed by `DamerauLevenshtein` itself, with `checkmetric=false`. That makes the
index **approximate**, and knowingly so: the restricted (OSA) variant violates the triangle
inequality that `BKT`'s pruning relies on, so a true neighbour can be missed. Measured over
3,000 synthetic typos against a 13,871-token Spanish vocabulary, the true source was among the
returned candidates **99.2%** of the time. The exact alternative -- key by `Levenshtein`, search
at radius 2, filter by Damerau-Levenshtein -- costs 33% of an exhaustive scan against 7.9% for
this route, per `BKT`'s own measurements, which is not a good trade for 0.8%.

# `min_ndocs`

A floor on what may be *corrected to*. `1` indexes every token, which is usually right because a
fitted vocabulary is already pruned (`fit_profile`'s own `min_ndocs`). Raise it to keep a
correction from landing on a token the corpus barely holds.

# `minlength`

A floor on the *typed* token, and it is a cost gate rather than a quality one -- which is worth
saying plainly, because it was built expecting the opposite. The uniqueness rule in
[`edit_candidates`](@ref) already subsumes it: measured by typed length over 3,860 synthetic
typos, precision given a unique candidate is flat at ~0.999 across *every* band, including the
short ones, because a short token essentially never has a unique distance-1 neighbourhood in the
first place:

| typed length | 3 | 4 | 5 | 6 | 7 | 8-9 | 10+ |
|---|---|---|---|---|---|---|---|
| fraction with a unique candidate | 0.01 | 0.24 | 0.51 | 0.72 | 0.85 | 0.91 | 0.96 |
| precision when it fires | 1.00 | 1.00 | 1.00 | 0.998 | 1.00 | 0.999 | 1.00 |

So raising the floor only buys cost: `minlength=7` drops coverage from 0.684 to 0.442 and leaves
precision at 0.999. The default of `4` skips the band where a lookup is nearly always wasted --
at length 1 every single-character token (emoji, punctuation, single letters) is one substitution
from every other, ~909 of them in a 60,636-token vocabulary, so uniqueness can never hold.

See also [`edit_candidates`](@ref), [`derive_variants`](@ref).
"""
function derive_edits(voc::Vocabulary; minlength::Integer=4, min_ndocs::Integer=1,
                      verbose::Bool=false)
    minlength >= 1 || throw(ArgumentError("minlength must be positive; got $minlength"))

    ids = UInt32[]
    words = Vector{Char}[]
    for i in eachindex(voc)
        getndocs(voc, i) >= min_ndocs || continue
        push!(ids, convert(UInt32, i))
        push!(words, collect(gettoken(voc, i)))
    end

    # `reporters=nothing` silences SimilaritySearch's own build log: this is a library call made
    # once per index, and `verbose` below is the knob a caller of *this* function reaches for
    ctx = GenericContext(; reporters=(verbose ? SimilaritySearch.InformativeLog() : nothing))
    bkt = BKT(Dist.Seqs.DamerauLevenshtein(), VectorDatabase(words); checkmetric=false)
    index!(bkt, ctx)
    verbose && println(stderr,
        "derive_edits: indexed $(length(ids)) of $(vocsize(voc)) token(s) for distance-1 lookup")

    EditIndex(bkt, ctx, ids, Int(minlength))
end

"""
    edit_candidates(ei::EditIndex, tok) -> Vector{UInt32}

The vocabulary ids within Damerau-Levenshtein distance 1 of `tok`, ascending. Empty when `tok`
is shorter than `ei.minlength`, so a caller pays nothing for the band where a lookup cannot
produce a usable answer.

This returns the whole neighbourhood; deciding whether it is safe to *act* on belongs to
[`resolve_query_tokens`](@ref), which acts only when there is exactly one candidate. That rule
is measured rather than chosen for tidiness -- over 3,000 synthetic typos against a
13,871-token Spanish vocabulary:

| rule | coverage | precision when it fires |
|---|---|---|
| act on the most frequent candidate, always | 1.000 | 0.868 |
| `length >= 8` | 0.328 | 0.968 |
| dominant beats runner-up by `negligible_ratio` (50x) | 0.710 | 0.990 |
| **exactly one candidate** | **0.701** | **0.999** |

A unique candidate has a dominance ratio of `Inf`, so the unique set is a strict subset of the
dominant one: those extra 0.9 points of coverage cost an order of magnitude of precision. There
is therefore no ratio knob here, and `QueryPolicy` gains no field.

Why it works at all: an out-of-vocabulary string sits in a far sparser region than a real token.
Measured on a 60,636-token vocabulary, a typo string has 1.91 distance-1 neighbours on average
(median 1, p90 4) against 18.9 for a token the vocabulary holds.
"""
function edit_candidates(ei::EditIndex, tok::AbstractString)
    out = UInt32[]
    (isempty(ei.ids) || length(tok) < ei.minlength) && return out

    res = search(ei.index, ei.context, collect(tok), RadiusSorted(1f0))
    for i in res.ids
        push!(out, ei.ids[i])
    end
    # ascending by id rather than by the tree's traversal order: the neighbourhood is a set, and
    # a query answered twice should name it the same way both times
    sort!(out)
    out
end
