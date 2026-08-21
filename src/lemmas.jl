# This file is a part of TextSearch.jl

export lemma_clusters

function _lemma_pick(selector::Symbol)
    if selector === :shortest
        (voc, group) -> begin
            best, best_len = group[1], length(token(voc, group[1]))
            for tid in group
                len = length(token(voc, tid))
                len < best_len && ((best, best_len) = (tid, len))
            end
            best
        end
    elseif selector === :most_frequent
        (voc, group) -> begin
            best, best_occs = group[1], occs(voc, group[1])
            for tid in group
                o = occs(voc, tid)
                o > best_occs && ((best, best_occs) = (tid, o))
            end
            best
        end
    elseif selector === :shortest_then_most_frequent
        (voc, group) -> begin
            best = group[1]
            best_key = (length(token(voc, best)), -occs(voc, best))
            for tid in group
                key = (length(token(voc, tid)), -occs(voc, tid))
                key < best_key && ((best, best_key) = (tid, key))
            end
            best
        end
    else
        error("unknown lemma selector: $selector; supported: shortest, most_frequent, shortest_then_most_frequent")
    end
end

# ── morphological similarity ─────────────────────────────────────────────────
#
# Semantic clustering alone does not produce lemmas. LSI embeddings capture distributional
# similarity, so a token's nearest neighbors are its *topical* relatives ("guerra" ->
# "belico", "aliados"), not its inflected forms -- which is exactly what the synonym network
# is for. Measured on Spanish Wikipedia, purely semantic clusters put two morphological
# variants together only ~2% of the time even when shrunk to an average of 1.5 tokens each.
#
# So morphology has to enter explicitly: the semantic clusters are split into subclusters of
# tokens that also *look* alike, and a lemma is elected per subcluster. That keeps the
# semantic step (which prevents merging unrelated look-alikes such as "casa"/"caso") while
# letting surface form decide what counts as the same word.

"""
    _qgram_ids(s, q, vocab) -> Vector{Int32}

Sorted, deduplicated ids of `s`'s character `q`-grams, interning grams through `vocab` so
distinct grams never collide. This is the representation
`SimilaritySearch.Dist.Sets.Jaccard` expects (a sorted set as a vector). Tokens shorter
than `q` are represented by themselves, so they can only match identical tokens.
"""
function _qgram_ids(s::AbstractString, q::Int, vocab::Dict{String,Int32})
    cs = collect(s)
    ids = Int32[]
    if length(cs) < q
        push!(ids, get!(vocab, s, Int32(length(vocab) + 1)))
    else
        for i in 1:(length(cs) - q + 1)
            g = String(@view cs[i:(i + q - 1)])
            push!(ids, get!(vocab, g, Int32(length(vocab) + 1)))
        end
    end
    sort!(ids)
    unique!(ids)
    ids
end

"""
    _morphology_metric(morphology, qgram) -> (prepare, distance)

Builds the pair of functions the subclustering needs: `prepare(token_string)` computes
whatever representation the metric compares, and `distance(a, b)` scores two prepared
representations on `[0, 1]` (0 = identical surface form).

- `:jaccard`: Jaccard distance over character `qgram`-gram sets, via
  `Dist.Sets.Jaccard`. Insensitive to where the difference falls, so it handles prefixal,
  suffixal and infixal variation alike.
- `:levenshtein`: edit distance via `Dist.Seqs.Levenshtein`, normalized by the longer token so a
  single threshold means the same thing for short and long words.

Both are normalized deliberately: an absolute edit distance of 2 is negligible between long
words and total between short ones, so a raw threshold would behave inconsistently across a
real vocabulary.
"""
function _morphology_metric(morphology::Symbol, qgram::Int)
    if morphology === :jaccard
        vocab = Dict{String,Int32}()
        jac = Dist.Sets.Jaccard()
        (s -> _qgram_ids(s, qgram, vocab)), ((a, b) -> Dist.evaluate(jac, a, b))
    elseif morphology === :levenshtein
        lev = Dist.Seqs.Levenshtein()
        # tokens are compared as Char vectors, not Strings: `Dist.Seqs.Levenshtein` indexes
        # its arguments positionally (`a[i]`), which on a String means *byte* offsets and
        # throws StringIndexError on any multi-byte character -- and a real Spanish
        # vocabulary is full of them ("»", "—", ...). Char vectors also make the length
        # normalization below count characters rather than bytes.
        collect, ((a, b) -> begin
            n = max(length(a), length(b))
            n == 0 ? 0f0 : Dist.evaluate(lev, a, b) / n
        end)
    else
        error("unknown morphology: $morphology; supported: none, jaccard, levenshtein")
    end
end

"""
    _common_prefix_len(a, b) -> Int

Number of leading characters `a` and `b` share (both given as `Char` vectors).
"""
function _common_prefix_len(a::Vector{Char}, b::Vector{Char})
    n = 0
    @inbounds for i in 1:min(length(a), length(b))
        a[i] == b[i] || break
        n += 1
    end
    n
end

"""
    _morph_subclusters(voc, group, prepare, distance, threshold, min_common_prefix) -> Vector{Vector{UInt32}}

Splits one semantic cluster into morphological subclusters by single-linkage: two tokens
land together when their surface forms are within `threshold`, transitively. Single linkage
is the right shape here because an inflection family is a chain (`clara`-`claras`-`claro`),
not a ball -- requiring every pair to be close would fragment it.

`min_common_prefix > 0` additionally requires two tokens to agree on that many leading
characters before they can link. Character n-gram similarity is position-blind, which is
what lets it match `abioticos` with `bioticos` or `abandonadas` with `donadas` -- pairs that
share almost every gram yet are different words. Requiring a shared prefix encodes that the
target language inflects by suffix; set it to `0` for languages where that does not hold.

Cost is `O(|group|^2)` distance evaluations, so it is the semantic step's `num_clusters`
that keeps this affordable: fewer, larger clusters make this quadratic term dominate.
"""
function _morph_subclusters(voc::Vocabulary, group::Vector{UInt32}, prepare, distance, threshold::Real,
                            min_common_prefix::Int)
    n = length(group)
    n == 1 && return [group]

    toks = [token(voc, tid) for tid in group]
    chars = min_common_prefix > 0 ? [collect(t) for t in toks] : nothing
    reps = [prepare(t) for t in toks]
    parent = collect(1:n)
    find(x) = begin
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end

    @inbounds for i in 1:(n - 1), j in (i + 1):n
        if min_common_prefix > 0 && _common_prefix_len(chars[i], chars[j]) < min_common_prefix
            continue
        end
        if distance(reps[i], reps[j]) <= threshold
            ri, rj = find(i), find(j)
            ri != rj && (parent[ri] = rj)
        end
    end

    subs = Dict{Int,Vector{UInt32}}()
    @inbounds for i in 1:n
        push!(get!(() -> UInt32[], subs, find(i)), group[i])
    end

    collect(values(subs))
end

"""
    _link_subclusters(items, close) -> Vector{Vector{T}}

Single-linkage grouping of `items` under the predicate `close(i, j)` (indices into `items`),
by union-find. `O(length(items)^2)` predicate calls, so callers must keep the input small
(by blocking, or by having partitioned already).
"""
function _link_subclusters(items::Vector{T}, close) where {T}
    n = length(items)
    n <= 1 && return [items]

    parent = collect(1:n)
    find(x) = begin
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end

    @inbounds for i in 1:(n - 1), j in (i + 1):n
        if close(i, j)
            ri, rj = find(i), find(j)
            ri != rj && (parent[ri] = rj)
        end
    end

    subs = Dict{Int,Vector{T}}()
    @inbounds for i in 1:n
        push!(get!(() -> T[], subs, find(i)), items[i])
    end
    collect(values(subs))
end

"""
    _prefix_blocks(voc, ids, prefix_len) -> Vector{Vector{UInt32}}

Buckets `ids` by their tokens' first `prefix_len` characters. When linking *requires* a
shared prefix of that length, this blocking is exact -- two tokens in different buckets can
never link -- and it is what makes morphology-first clustering affordable: comparing the
whole vocabulary pairwise is `O(vocsize^2)` (10^10 pairs at 143k tokens), while the sum over
buckets is smaller by orders of magnitude.

`prefix_len <= 0` cannot block, so everything lands in a single bucket.
"""
function _prefix_blocks(voc::Vocabulary, ids, prefix_len::Int)
    prefix_len <= 0 && return [collect(UInt32, ids)]
    blocks = Dict{String,Vector{UInt32}}()
    for tid in ids
        cs = collect(token(voc, tid))
        key = length(cs) >= prefix_len ? String(@view cs[1:prefix_len]) : String(cs)
        push!(get!(() -> UInt32[], blocks, key), UInt32(tid))
    end
    collect(values(blocks))
end

"""
    lemma_clusters(voc::Vocabulary, wordvecs::AbstractDatabase;
                   algorithm::Symbol=:fft, num_clusters::Integer=0,
                   selector::Symbol=:shortest, dist=Dist.Cosine(),
                   morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                   qgram::Integer=2, min_common_prefix::Integer=3,
                   order::Symbol=:semantic_first, semantic_threshold::Real=0.5) -> Dict{String,String}

Derives a `token => lemma` map by combining two signals:

1. **Semantic clustering** of `voc`'s tokens by their embeddings in `wordvecs` (column `t` =
   embedding of `token(voc, t)`, e.g. from [`LSI.wordvectors`](@ref)), via one of
   `SimilaritySearch`'s `fft`/`dnet`/`randsel`/`multirandsel`. `num_clusters = 0` defaults
   to `ceil(sqrt(vocsize(voc)))`.
2. **Morphological subclustering** inside each semantic cluster (`morphology`,
   `morphology_threshold`, `qgram` -- see [`_morphology_metric`](@ref)), so only tokens that
   also *look* alike end up sharing a lemma.

Then one canonical token is elected per subcluster (`selector`: `:shortest`,
`:most_frequent`, or `:shortest_then_most_frequent`) and every other member maps to it.
Subclusters of one are left alone. Returns only non-identity entries -- a lookup miss means
the token is its own lemma.

Step 2 is what makes the result lemma-shaped rather than topic-shaped: embeddings alone put
"guerra" next to "belico" rather than next to "guerras" (measured ~2% morphological pairs
on Spanish Wikipedia), while surface similarity alone would happily merge "casa" with
"caso". Pass `morphology=:none` to recover the purely semantic behaviour, which elects one
representative per *semantic* cluster and is better described as topic representatives than
as lemmas.

# Example
```julia
lemmas = lemma_clusters(voc, wordvectors(lsi))
lemmas["casas"]   # "casa"
```
"""
function lemma_clusters(voc::Vocabulary, wordvecs::AbstractDatabase;
                         algorithm::Symbol=:fft, num_clusters::Integer=0,
                         selector::Symbol=:shortest, dist=Dist.Cosine(),
                         morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                         qgram::Integer=2, min_common_prefix::Integer=3,
                         order::Symbol=:semantic_first, semantic_threshold::Real=0.5)
    m = vocsize(voc)
    pick = _lemma_pick(selector)
    mp = Int(min_common_prefix)

    prepare, morphdist = morphology === :none ? (identity, nothing) :
                         _morphology_metric(morphology, Int(qgram))

    # a token whose embedding collapsed to zero has no direction to compare (cosine gives
    # NaN), so it is treated as far from everything rather than silently linking
    semclose(a, b) = begin
        d = Dist.evaluate(dist, a, b)
        !isnan(d) && d <= semantic_threshold
    end

    finalgroups = Vector{Vector{UInt32}}()

    if order === :semantic_first
        R = _semantic_clustering(algorithm, dist, wordvecs, num_clusters, m)
        groups = Dict{UInt32,Vector{UInt32}}()
        for tid in 1:m
            push!(get!(() -> UInt32[], groups, R.nn[tid]), UInt32(tid))
        end
        for group in values(groups)
            length(group) <= 1 && continue
            if morphdist === nothing
                push!(finalgroups, group)
            else
                for blk in _prefix_blocks(voc, group, mp)
                    length(blk) <= 1 && continue
                    reps = [prepare(token(voc, tid)) for tid in blk]
                    append!(finalgroups, _link_subclusters(blk, (i, j) -> morphdist(reps[i], reps[j]) <= morphology_threshold))
                end
            end
        end
    elseif order === :morphology_first
        morphdist === nothing &&
            error("order=:morphology_first needs a morphology (:jaccard or :levenshtein), got :none")
        # 1. morphological families over the WHOLE vocabulary, made affordable by blocking
        for blk in _prefix_blocks(voc, 1:m, mp)
            length(blk) <= 1 && continue
            reps = [prepare(token(voc, tid)) for tid in blk]
            for fam in _link_subclusters(blk, (i, j) -> morphdist(reps[i], reps[j]) <= morphology_threshold)
                length(fam) <= 1 && continue
                # 2. embeddings then only *split* a family, keeping homographs apart
                vecs = [wordvecs[tid] for tid in fam]
                append!(finalgroups, _link_subclusters(fam, (i, j) -> semclose(vecs[i], vecs[j])))
            end
        end
    else
        error("unknown order: $order; supported: semantic_first, morphology_first")
    end

    lemmas = Dict{String,String}()
    for group in finalgroups
        length(group) <= 1 && continue
        chosen_tok = token(voc, pick(voc, group))
        for tid in group
            tok = token(voc, tid)
            tok == chosen_tok || (lemmas[tok] = chosen_tok)
        end
    end

    lemmas
end

"""
    _semantic_clustering(algorithm, dist, wordvecs, num_clusters, m)

Runs the requested `SimilaritySearch` clustering over the token embeddings, defaulting
`num_clusters` to `ceil(sqrt(m))`.
"""
function _semantic_clustering(algorithm::Symbol, dist, wordvecs, num_clusters::Integer, m::Integer)
    k = num_clusters > 0 ? num_clusters : max(1, ceil(Int, sqrt(m)))
    if algorithm === :fft
        fft(dist, wordvecs, k; verbose=false)
    elseif algorithm === :dnet
        dnet(dist, wordvecs, k; verbose=false)
    elseif algorithm === :randsel
        randsel(dist, wordvecs, k; verbose=false)
    elseif algorithm === :multirandsel
        multirandsel(dist, wordvecs, k; verbose=false)
    else
        error("unknown lemma clustering algorithm: $algorithm; supported: fft, dnet, randsel, multirandsel")
    end
end
