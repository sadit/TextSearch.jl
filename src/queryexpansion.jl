# This file is a part of TextSearch.jl

export expand_synonyms!

@inline _synlt_id(X, i, j) = @inbounds X[1][i] < X[1][j]
@inline function _synswap_id_val(X, i, j)
    @inbounds X[1][i], X[1][j] = X[1][j], X[1][i]
    @inbounds X[2][i], X[2][j] = X[2][j], X[2][i]
end

"""
    expand_synonyms!(vec::SparseVector, voc::Vocabulary, synonyms;
                      weight_fn=d -> exp(-d), normalize::Bool=true) -> vec

Expands a **query**'s sparse tf-idf vector IN PLACE with weighted contributions from each present
token's synonyms (`synonyms`, e.g. as produced by [`LSI.synonyms`](@ref)). This mutates `vec` --
pass an unnormalized, disposable query vector (`vectorize(model, query; normalize=false)`); never a
vector you still need afterwards, and never a document vector (documents are never expanded, only
queries). Normalizing before calling this would also make the original-vs-synonym weight ratio
depend on how many tokens the query had, not on the intended per-synonym weighting -- that's why
`normalize` (default `true`) happens here, as the final step.

For each of `vec`'s original nonzero `(tokenID, weight)` pairs (captured once, before any
appending), looks up its string via `token(voc, tokenID)`; if it's a key of `synonyms`, appends
`weight * weight_fn(distance)` at `token2id(voc, synonym)` for every `(synonym, distance)` pair (an
OOV synonym -- `token2id` returning `0` -- is silently skipped, matching `bagofwords!`/`vectorize!`'s
existing convention). The appended entries are then merged into `vec`'s existing nonzeros: the
combined `(nzind, nzval)` arrays are heap-sorted by id (reusing `SimilaritySearch.heapify!`/`heapsort!`,
the same coupled-array sort used to build a `SparseVector` out of a `KnnQueue`), duplicate ids (a
synonym that was also already present, or reached via two different original tokens) are combined
in a single two-pointer reduction pass, and the backing arrays are `resize!`d down to the final
count -- an in-place O(n log n) merge, no new allocation for the index/value storage itself.

`weight_fn` defaults to `exp(-d)` (`1.0` at distance `0`, decaying smoothly with distance); pass
e.g. `d -> d < 0.3 ? 0.5 : 0.0` for a hard cutoff instead.
"""
function expand_synonyms!(vec::SparseVector, voc::Vocabulary, synonyms; weight_fn=d -> exp(-d), normalize::Bool=true)
    nzind = vec.nzind
    nzval = vec.nzval
    m0 = length(nzind)

    for i in 1:m0
        tok = token(voc, nzind[i])
        haskey(synonyms, tok) || continue
        v = nzval[i]
        for (syn, dist) in synonyms[tok]
            sid = token2id(voc, syn)
            sid == 0 && continue
            push!(nzind, sid)
            push!(nzval, v * Float32(weight_fn(dist)))
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
