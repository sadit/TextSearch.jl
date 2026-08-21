# This file is a part of TextSearch.jl

export lemma_clusters, extend_lemmas_morphological

"""
    _selector_key(selector) -> (voc, tid) -> key

Sort key matching `_lemma_pick`'s preference, so a leader pass can visit candidates in the
order the selector would elect them and its seed is already the lemma.
"""
function _selector_key(selector::Symbol)
    if selector === :shortest
        (voc, tid) -> (length(gettoken(voc, tid)), gettoken(voc, tid))
    elseif selector === :most_frequent
        (voc, tid) -> (-getoccs(voc, tid), gettoken(voc, tid))
    elseif selector === :shortest_then_most_frequent
        (voc, tid) -> (length(gettoken(voc, tid)), -getoccs(voc, tid), gettoken(voc, tid))
    else
        error("unknown lemma selector: $selector; supported: shortest, most_frequent, shortest_then_most_frequent")
    end
end

function _lemma_pick(selector::Symbol)
    if selector === :shortest
        (voc, group) -> begin
            best, best_len = group[1], length(gettoken(voc, group[1]))
            for tid in group
                len = length(gettoken(voc, tid))
                len < best_len && ((best, best_len) = (tid, len))
            end
            best
        end
    elseif selector === :most_frequent
        (voc, group) -> begin
            best, best_occs = group[1], getoccs(voc, group[1])
            for tid in group
                o = getoccs(voc, tid)
                o > best_occs && ((best, best_occs) = (tid, o))
            end
            best
        end
    elseif selector === :shortest_then_most_frequent
        (voc, group) -> begin
            best = group[1]
            best_key = (length(gettoken(voc, best)), -getoccs(voc, best))
            for tid in group
                key = (length(gettoken(voc, tid)), -getoccs(voc, tid))
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
# Embeddings do not find lemmas. LSI captures distributional similarity, so a token's
# nearest neighbours are its *topical* relatives ("guerra" -> "belico", "aliados"), which is
# what the synonym network is for. Measured on Spanish Wikipedia, purely semantic clusters
# put two morphological variants together only ~2% of the time even when shrunk to an
# average of 1.5 tokens each, so electing one representative per semantic cluster produced
# mappings like "casas" -> "dia".
#
# Morphology therefore leads (`order=:morphology_first`, the default): tokens are grouped by
# surface similarity over the whole vocabulary, and embeddings are demoted to *splitting* a
# family when its members turn out to mean different things. Leading with the embeddings
# instead (`:semantic_first`) imposes a hard partition that morphology can never cross, so
# variants landing in different clusters are unreachable however alike they are spelled --
# measured, that costs roughly two thirds of the coverage and runs ~10x slower.

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
    _leader_groups(items, order, close) -> Vector{Vector{T}}

Groups `items` around seeds: walking them in `order`, the first unassigned item becomes a
seed and every still-unassigned item that is `close` **to that seed** joins it.

This deliberately replaces single-linkage for morphology. Single linkage chains -- `A~B` and
`B~C` merge even when `A` and `C` are unrelated -- and on a real vocabulary the chains
swallow everything sharing a prefix: measured on 143k Spanish tokens it produced a
292-member "family" spanning `concentra`...`cons`, and merged `cara` with `caracas` and
`caracalla`. Requiring closeness to the seed instead bounds every group by one radius around
its lemma, which is also exactly the shape "a lemma plus its variants" should have.

Visiting in the selector's own order (see [`_selector_key`](@ref)) makes the seed the token
the selector would have elected anyway.
"""
function _leader_groups(items::Vector{T}, order::Vector{Int}, close) where {T}
    n = length(items)
    n <= 1 && return [items]

    assigned = falses(n)
    groups = Vector{Vector{T}}()
    @inbounds for si in order
        assigned[si] && continue
        assigned[si] = true
        grp = T[items[si]]
        for qi in order
            assigned[qi] && continue
            if close(si, qi)
                assigned[qi] = true
                push!(grp, items[qi])
            end
        end
        push!(groups, grp)
    end
    groups
end

"""
    _prefix_blocks(voc, ids, prefix_len) -> Vector{Vector{UInt32}}

Buckets `ids` by their tokens' first `prefix_len` characters. When linking *requires* a
shared prefix of that length, this blocking is exact -- two tokens in different buckets can
never link -- and it is what makes morphology-first clustering affordable: comparing the
whole vocabulary pairwise is `O(vocsize^2)` (10^10 pairs at 143k tokens), while the sum over
buckets is smaller by orders of magnitude.

`prefix_len <= 0` cannot block, so everything lands in a single bucket -- which also means
`min_common_prefix = 0` gives up the blocking speedup entirely.

Requiring a shared prefix is not only an optimization: character n-gram similarity is
position-blind, so without it `abioticos`/`bioticos` and `abandonadas`/`donadas` link on
sharing nearly every gram despite being different words. It encodes that the target language
inflects by suffix, so set it to `0` for languages where that does not hold.
"""
function _prefix_blocks(voc::Vocabulary, ids, prefix_len::Int)
    prefix_len <= 0 && return [collect(UInt32, ids)]
    blocks = Dict{String,Vector{UInt32}}()
    for tid in ids
        cs = collect(gettoken(voc, tid))
        key = length(cs) >= prefix_len ? String(@view cs[1:prefix_len]) : String(cs)
        push!(get!(() -> UInt32[], blocks, key), UInt32(tid))
    end
    collect(values(blocks))
end

"""
    lemma_clusters(voc::Vocabulary, wordvecs::AbstractDatabase;
                   algorithm::Symbol=:fft, num_clusters::Integer=0,
                   selector::Symbol=:most_frequent, dist=Dist.Cosine(),
                   morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                   qgram::Integer=2, min_common_prefix::Integer=3,
                   order::Symbol=:morphology_first, semantic_threshold::Real=1.0) -> Dict{String,String}

Derives a `token => lemma` map by combining two signals:

1. **Semantic clustering** of `voc`'s tokens by their embeddings in `wordvecs` (column `t` =
   embedding of `gettoken(voc, t)`, e.g. from [`LSI.wordvectors`](@ref)), via one of
   `SimilaritySearch`'s `fft`/`dnet`/`randsel`/`multirandsel`. `num_clusters = 0` defaults
   to `ceil(sqrt(vocsize(voc)))`.
2. **Morphological subclustering** inside each semantic cluster (`morphology`,
   `morphology_threshold`, `qgram` -- see [`_morphology_metric`](@ref)), so only tokens that
   also *look* alike end up sharing a lemma.

Then one canonical token is elected per group (`selector`: `:most_frequent` by default,
`:shortest`, or `:shortest_then_most_frequent`) and every other member maps to it. The
selector also decides seeding order, so it is more consequential than a tie-break:
`:shortest` lets a short misspelling win, and a junk seed fragments the family around it --
measured on 143k Spanish tokens, the typo `guera` seeded a group that swallowed `guerra` and
left `guerras` stranded. `:most_frequent` seeds on the form the corpus actually uses, which
recovered `guerras -> guerra`, `jugadores -> jugador` and `concentraciones -> concentracion`
in the same run.
Subclusters of one are left alone. Returns only non-identity entries -- a lookup miss means
the token is its own lemma.

`order` decides which signal partitions first:

- `:morphology_first` (default): surface-similar families over the whole vocabulary (made
  affordable by blocking on the required shared prefix), then `semantic_threshold` splits a
  family whose members are far apart in embedding space. Whole conjugations collapse
  correctly this way (`abandona`, `abandonado`, `abandonar`, `abandone`, ... -> `abandono`).
- `:semantic_first`: the original order -- cluster by embedding, then split each cluster by
  surface similarity. Retained because it is the only order that respects a caller-supplied
  `algorithm`/`num_clusters`, but it fragments inflection families across clusters.

`semantic_threshold` is a distance under `dist`, so with the default cosine it lives on
`[0, 2]`; the default `1.0` was picked by measurement rather than taste. Tightening it does
not buy precision -- it mostly deletes correct inflections (at `0.9` only 4 of 10 probed
inflections survive, against 9 of 10 at `1.0`), while loosening it past `~1.05` stops
catching anything (the artifacts it legitimately removes are cross-language and truncation
pairs such as `academic`/`academia` and `abstracta`/`abstract`).

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
                         selector::Symbol=:most_frequent, dist=Dist.Cosine(),
                         morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                         qgram::Integer=2, min_common_prefix::Integer=3,
                         order::Symbol=:morphology_first, semantic_threshold::Real=1.0)
    m = vocsize(voc)
    pick = _lemma_pick(selector)
    keyof = _selector_key(selector)
    mp = Int(min_common_prefix)

    # morphological grouping is leader-based, not single-linkage: see `_leader_groups`
    morphgroups(ids) = begin
        toks = [gettoken(voc, tid) for tid in ids]
        reps = [prepare(t) for t in toks]
        order = sortperm(eachindex(ids); by=i -> keyof(voc, ids[i]))
        _leader_groups(ids, order, (i, j) -> morphdist(reps[i], reps[j]) <= morphology_threshold)
    end

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
                    append!(finalgroups, morphgroups(blk))
                end
            end
        end
    elseif order === :morphology_first
        morphdist === nothing &&
            error("order=:morphology_first needs a morphology (:jaccard or :levenshtein), got :none")
        # 1. morphological families over the WHOLE vocabulary, made affordable by blocking
        for blk in _prefix_blocks(voc, 1:m, mp)
            length(blk) <= 1 && continue
            for fam in morphgroups(blk)
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
        chosen_tok = gettoken(voc, pick(voc, group))
        for tid in group
            tok = gettoken(voc, tid)
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

"""
    extend_lemmas_morphological(voc::Vocabulary, lemmas::AbstractDict;
                                candidates=nothing,
                                morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                                qgram::Integer=2, min_common_prefix::Integer=3,
                                selector::Symbol=:most_frequent) -> Dict{String,String}

Derives **additional** `token => lemma` entries for `voc` from surface similarity alone, and
returns only the new ones (merge them into `lemmas` yourself).

This exists because morphology is the signal that actually groups an inflection family:
[`lemma_clusters`](@ref) uses embeddings only to *split* a family whose members mean
different things, never to form one. So a family can be recovered without fitting any
embedding -- which is what makes it usable on a vocabulary that arrived after the model was
trained, e.g. the tokens a refit's sample brings that its base profile never saw
([`refit_profile`](@ref)). Nothing here needs `wordvecs`, an LSI, or a second pass over a
corpus. The tradeoff is that no semantic check can veto a grouping, so two look-alike words
with unrelated meanings will merge where full `lemma_clusters` would have kept them apart.

`candidates` bounds both the cost and the scope. Only prefix blocks containing at least one
candidate token are examined -- the reason this stays cheap when `voc` is a whole base
vocabulary and only a handful of tokens are new -- and only candidate tokens get entries.
That restriction is deliberate: a family may well contain two tokens the base's own
clustering saw and chose *not* to link, and silently overruling that decision is not this
function's business. Pass `nothing` to consider everything.

Tokens already keyed in `lemmas` are skipped, so no chain `token -> lemma -> other lemma` can
be created. Note that under an applied [`LemmaTransformation`](@ref) they are not vocabulary
tokens to begin with.
"""
function extend_lemmas_morphological(voc::Vocabulary, lemmas::AbstractDict;
                                      candidates=nothing,
                                      morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                                      qgram::Integer=2, min_common_prefix::Integer=3,
                                      selector::Symbol=:most_frequent)
    morphology === :none &&
        error("extend_lemmas_morphological needs a morphology (:jaccard or :levenshtein) -- " *
              "surface similarity is the whole signal it has, got :none")

    prepare, morphdist = _morphology_metric(morphology, Int(qgram))
    pick = _lemma_pick(selector)
    keyof = _selector_key(selector)
    mp = Int(min_common_prefix)

    ids = UInt32[UInt32(tid) for tid in eachindex(voc) if !haskey(lemmas, gettoken(voc, tid))]
    length(ids) <= 1 && return Dict{String,String}()

    want = candidates === nothing ? nothing : Set{String}(String(c) for c in candidates)
    out = Dict{String,String}()

    for blk in _prefix_blocks(voc, ids, mp)
        length(blk) <= 1 && continue
        want === nothing || any(tid -> gettoken(voc, tid) in want, blk) || continue

        reps = [prepare(gettoken(voc, tid)) for tid in blk]
        order = sortperm(eachindex(blk); by=i -> keyof(voc, blk[i]))
        groups = _leader_groups(blk, order,
                                (i, j) -> morphdist(reps[i], reps[j]) <= morphology_threshold)

        for fam in groups
            length(fam) <= 1 && continue
            chosen = gettoken(voc, pick(voc, fam))
            for tid in fam
                tok = gettoken(voc, tid)
                tok == chosen && continue
                want === nothing || tok in want || continue
                out[tok] = chosen
            end
        end
    end

    out
end
