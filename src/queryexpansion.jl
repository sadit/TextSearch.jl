# This file is a part of TextSearch.jl

export expand_synonyms!

@inline _synlt_id(X, i, j) = @inbounds X[1][i] < X[1][j]
@inline function _synswap_id_val(X, i, j)
    @inbounds X[1][i], X[1][j] = X[1][j], X[1][i]
    @inbounds X[2][i], X[2][j] = X[2][j], X[2][i]
end

# Default weightings for the two modes of the SparseVector expansion below. Rank is the
# default because a synonym network stores its neighbors in rank order and that is the part
# that transfers between models; distances are optional side data (see `synonyms`).
@inline _rank_weight(rank::Integer) = 1f0 / rank
@inline _dist_weight(dist::Real) = Float32(exp(-dist))

"""
    expand_synonyms!(vec::SparseVector, voc::Vocabulary, synonyms;
                      distances=nothing, weight_fn=nothing, normalize::Bool=true) -> vec

Expands a **query**'s sparse tf-idf vector IN PLACE with weighted contributions from each present
token's synonyms (`synonyms`, e.g. as produced by [`LSI.synonyms`](@ref)). This mutates `vec` --
pass an unnormalized, disposable query vector (`vectorize(model, query; normalize=false)`); never a
vector you still need afterwards, and never a document vector (documents are never expanded, only
queries). Normalizing before calling this would also make the original-vs-synonym weight ratio
depend on how many tokens the query had, not on the intended per-synonym weighting -- that's why
`normalize` (default `true`) happens here, as the final step.

`synonyms` maps a token to its neighbor tokens **in rank order** (nearest first). For each of
`vec`'s original nonzero `(tokenID, weight)` pairs (captured once, before any appending), looks up
its string via `token(voc, tokenID)`; if it's a key of `synonyms`, appends `weight * weight_fn(...)`
at `token2id(voc, synonym)` for every neighbor (an OOV synonym -- `token2id` returning `0` -- is
silently skipped, matching `bagofwords!`/`vectorize!`'s existing convention). The appended entries
are then merged into `vec`'s existing nonzeros: the combined `(nzind, nzval)` arrays are heap-sorted
by id (reusing `SimilaritySearch.heapify!`/`heapsort!`, the same coupled-array sort used to build a
`SparseVector` out of a `KnnQueue`), duplicate ids (a synonym that was also already present, or
reached via two different original tokens) are combined in a single two-pointer reduction pass, and
the backing arrays are `resize!`d down to the final count -- an in-place O(n log n) merge, no new
allocation for the index/value storage itself.

# Weighting modes

There are two, chosen by whether `distances` is given:

- **rank** (`distances === nothing`, the default): `weight_fn` receives the neighbor's 1-based
  rank, and defaults to `1/rank`. This is the normal mode. A network's ranking is what
  transfers between models -- distances live in whichever embedding space produced them, and a
  merged or refitted network's distances are no longer distances in any single space at all.
- **distance**: pass `distances`, a parallel mapping `token => Vector{Float32}` aligned with
  `synonyms[token]`; `weight_fn` then receives the distance and defaults to `exp(-d)` (`1.0` at
  distance `0`, decaying smoothly). Pass e.g. `d -> d < 0.3 ? 0.5 : 0.0` for a hard cutoff. A
  token missing from `distances`, or a short distance list, falls back to rank weighting for the
  neighbors it does not cover, so a partially-populated `distances` is safe rather than an error.
"""
function expand_synonyms!(vec::SparseVector, voc::Vocabulary, synonyms;
                           distances=nothing, weight_fn=nothing, normalize::Bool=true)
    nzind = vec.nzind
    nzval = vec.nzval
    m0 = length(nzind)
    bydist = distances !== nothing
    wf = weight_fn === nothing ? (bydist ? _dist_weight : _rank_weight) : weight_fn

    for i in 1:m0
        tok = token(voc, nzind[i])
        haskey(synonyms, tok) || continue
        v = nzval[i]
        neighbors = synonyms[tok]
        dl = bydist ? get(distances, tok, nothing) : nothing
        for (rank, syn) in enumerate(neighbors)
            sid = token2id(voc, syn)
            sid == 0 && continue
            # rank weighting also covers a neighbor the distance list does not reach, so a
            # partially-populated `distances` degrades instead of erroring
            w = if dl !== nothing && rank <= length(dl)
                Float32(wf(dl[rank]))
            elseif bydist
                _rank_weight(rank)
            else
                Float32(wf(rank))
            end
            push!(nzind, sid)
            push!(nzval, v * w)
        end
    end

    n = length(nzind)
    if n > m0
        X = (nzind, nzval)
        SimilaritySearch.heapify!(_synlt_id, _synswap_id_val, X, n)
        SimilaritySearch.heapsort!(_synlt_id, _synswap_id_val, X, n)

        w = 1
        @inbounds for r in 2:n
            if nzind[r] == nzind[w]
                nzval[w] += nzval[r]
            else
                w += 1
                nzind[w] = nzind[r]
                nzval[w] = nzval[r]
            end
        end

        resize!(nzind, w)
        resize!(nzval, w)
    end

    normalize && normalize!(vec)
    vec
end

"""
    expand_synonyms!(bow::AbstractDict{<:Integer,<:Real}, voc::Vocabulary, synonyms) -> bow

Expands a **query**'s bag-of-words IN PLACE by adding every present token's synonyms as
extra keys -- the [`BM25InvertedFile`](@ref) counterpart of the `SparseVector` method above.
There is no `weight_fn`/`normalize`/`distances` here: BM25 scoring (`bm25score`) never reads
the query side's frequencies, only which token ids are present ("query's own frequencies are
not used"), so an injected synonym only needs to make its id present in `bow` -- any positive
count works, and an id already present (e.g. the synonym also appears literally in the
query) is left untouched rather than overwritten. This is why a network's distances are not
needed on the normal path at all.

As with the `SparseVector` method, `bow`'s original keys are snapshotted once (via
`collect`) before any insertion, so newly-added synonym ids are never themselves expanded.
An OOV synonym (`token2id` returning `0`) is silently skipped, matching `bagofwords!`'s
existing convention.
"""
function expand_synonyms!(bow::AbstractDict{K,V}, voc::Vocabulary, synonyms) where {K<:Integer,V<:Real}
    for (tokenID, _) in collect(bow)
        tok = token(voc, tokenID)
        haskey(synonyms, tok) || continue
        for syn in synonyms[tok]
            sid = token2id(voc, syn)
            sid == 0 && continue
            k = K(sid)
            haskey(bow, k) || (bow[k] = one(V))
        end
    end
    bow
end
