# This file is a part of TextSearch.jl

export merge_profiles

# ── config compatibility ─────────────────────────────────────────────────────
#
# `==` is unreliable for these config structs: the default `==` compares heap-allocated
# fields by reference, so two structurally identical `TokenizationConfig`s built separately
# compare UNEQUAL (its `nlist` is a fresh Vector each time), while two default
# `NormalizationConfig`s compare EQUAL only by the accident of sharing the same
# module-level default Regex/Set objects. Merging therefore compares what the fields mean,
# field by field, rather than trusting `==` on the structs.
#
# Only POLICY is compared, and only for equality. There used to be a `_merge_transformations`
# here that had to special-case "these differ only in their stopword set" and union them --
# an artifact-combining rule wedged into an equality check, because artifacts lived inside the
# transformation. The union now happens below with the other artifact rules, where combining
# is the whole point.

function _same_normalization(a::NormalizationConfig, b::NormalizationConfig)
    for f in (:del_diac, :del_dup, :del_punc, :group_num, :group_url, :group_usr, :group_emo, :lc)
        getfield(a, f) === getfield(b, f) || return false
    end
    for f in (:re_user, :re_url, :re_num)
        getfield(a, f).pattern == getfield(b, f).pattern || return false
    end
    a.emojis == b.emojis
end

function _same_tokenization(a::TokenizationConfig, b::TokenizationConfig)
    a.nlist == b.nlist || return false
    a.mark_token_type === b.mark_token_type || return false
    # `save_profile` refuses to serialize custom generators, so any *loaded* profile has
    # none; a non-empty list here means someone built the config in-process.
    isempty(a.generators) && isempty(b.generators)
end

# ── synonym fusion ───────────────────────────────────────────────────────────

"""
    _fuse_synonyms(profiles, voc, k, rrf_k) -> (; synonyms, distances)

Fuses the per-profile synonym networks into one.

Each input profile fit its **own** encoder, so its neighbor *distances* live in its own
embedding space and are not numerically comparable across profiles -- averaging them
directly would be meaningless. What does transfer is the *ranking*: a token that several
independently-fit profiles all place near the same neighbor is far likelier to be a real
relation than one that a single profile ranked highly. So the lists are combined with
Reciprocal Rank Fusion, `score(candidate) = Σ_p 1/(rrf_k + rank_p)`, the standard way to
merge ranked lists produced by incomparable scorers. Fusing by rank is also why merging
needs no distances at all: a profile that carries only its ranking merges perfectly well.

`distances`, when the inputs carry any, holds the mean of the distances the contributing
profiles reported for each surviving pair -- informative (cosine distances share a scale
even across spaces) but, unlike a single profile's, no longer a distance in any one space.
It comes back empty when no input had distances. Candidates are restricted to tokens that
survive in the merged vocabulary.
"""
function _fuse_synonyms(profiles, voc::Vocabulary, k::Integer, rrf_k::Real)
    scores = Dict{String,Dict{String,Float64}}()
    dists = Dict{String,Dict{String,Vector{Float32}}}()
    widest = 0

    # Which inputs held each token, as a bitmask per merged-vocabulary id. A candidate can only
    # appear in a token's neighbour list in an input that held BOTH, so this is how many chances
    # the pair actually had -- see the normalization below.
    nw = cld(length(profiles), 64)
    held = zeros(UInt64, nw, vocsize(voc))
    for (j, p) in enumerate(profiles)
        w, b = fldmod(j - 1, 64)
        pv = p.model.voc
        for i in eachindex(pv)
            id = token2id(voc, gettoken(pv, i))
            id == 0 && continue
            @inbounds held[w + 1, id] |= UInt64(1) << b
        end
    end
    chances(a::Integer, b::Integer) = sum(w -> count_ones(@inbounds(held[w, a] & held[w, b])), 1:nw)

    for p in profiles
        pd = p.synonym_distances
        for (tok, neighbors) in p.synonyms
            token2id(voc, tok) == 0 && continue
            widest = max(widest, length(neighbors))
            s = get!(() -> Dict{String,Float64}(), scores, tok)
            dl = pd === nothing ? nothing : get(pd, tok, nothing)
            for (rank, syn) in enumerate(neighbors)
                token2id(voc, syn) == 0 && continue
                s[syn] = get(s, syn, 0.0) + 1.0 / (rrf_k + rank)
                if dl !== nothing && rank <= length(dl)
                    d = get!(() -> Dict{String,Vector{Float32}}(), dists, tok)
                    push!(get!(() -> Float32[], d, syn), Float32(dl[rank]))
                end
            end
        end
    end

    keep = k > 0 ? Int(k) : widest
    net = Dict{String,Vector{String}}()
    netdist = Dict{String,Vector{Float32}}()

    for (tok, s) in scores
        isempty(s) && continue
        tokid = token2id(voc, tok)
        # a candidate's mean distance, or `nothing` when no input reported one for it
        dtok = get(dists, tok, nothing)
        function meandist(c)
            dtok === nothing && return nothing
            ds = get(dtok, c, nothing)
            ds === nothing ? nothing : sum(ds) / length(ds)
        end

        # Fused score per OPPORTUNITY, not summed. Summing rewards a candidate for having been
        # present in more inputs, and presence is not uniform: `min_ndocs` pruning and per-batch
        # stopword removal leave 71.9% of a merged Portuguese vocabulary and 88.7% of an English
        # one present in only some inputs. Summed, seven rank-1 votes (0.115 at rrf_k=60) lose to
        # eight rank-3 votes (0.127), so a ubiquitous mediocre neighbour displaces a better one
        # that one batch never saw.
        #
        # The `+ 1` is what keeps this from over-correcting into the opposite error: a pure mean
        # would let a single lucky rank-1 sighting tie eight consistent ones, so a candidate
        # confirmed by more inputs keeps a mild edge. Pairs both present everywhere -- the
        # majority -- get the same divisor and are therefore ranked exactly as before.
        fused(c) = s[c] / (chances(tokid, token2id(voc, c)) + 1)

        cands = collect(keys(s))
        # highest fused score first; ties broken deterministically (closer mean distance
        # when known, then lexicographically) so a merge is reproducible regardless of Dict
        # ordering. Candidates without a distance sort after those with one, rather than
        # comparing `nothing` against a number.
        sort!(cands; by=c -> (-fused(c), something(meandist(c), Inf), c))
        resize!(cands, min(keep, length(cands)))
        net[tok] = cands

        # All or nothing per token: a partial list could not stay aligned with the ranking,
        # and a NaN placeholder would be unserializable (JSON rejects it).
        ds = [meandist(c) for c in cands]
        any(isnothing, ds) || (netdist[tok] = Float32[Float32(d) for d in ds])
    end

    (; synonyms=net, distances=netdist)
end

# ── lemma voting ─────────────────────────────────────────────────────────────

function _pick_canonical(tokens, voc::Vocabulary)
    # most frequent wins; ties go to the shorter, then lexicographically smaller token
    best = first(tokens)
    bestkey = (-getoccs(voc, token2id(voc, best)), length(best), best)
    for t in tokens
        key = (-getoccs(voc, token2id(voc, t)), length(t), t)
        key < bestkey && ((best, bestkey) = (t, key))
    end
    best
end

"""
    _vote_lemmas(profiles, voc) -> Dict{String,String}

Merges the per-profile `token => lemma` maps by plurality vote (ties broken by the
canonical-token rule: most frequent, then shortest, then lexicographic), keeping only
tokens and lemmas that survive in the merged vocabulary.

Independent votes can disagree in ways a single clustering never does -- `a => b` in some
profiles and `b => a` in others -- so the winning edges are then followed to a fixed point
so that a whole chain collapses onto one canonical token, and any cycle is resolved by
electing its most frequent member. Without that pass the merged map could contain cycles,
which would make naive lemma lookup non-terminating.
"""
function _vote_lemmas(profiles, voc::Vocabulary)
    votes = Dict{String,Dict{String,Int}}()
    for p in profiles
        for (tok, lemma) in p.lemmas
            (token2id(voc, tok) == 0 || token2id(voc, lemma) == 0) && continue
            v = get!(() -> Dict{String,Int}(), votes, tok)
            v[lemma] = get(v, lemma, 0) + 1
        end
    end

    raw = Dict{String,String}()
    for (tok, v) in votes
        top = maximum(values(v))
        raw[tok] = _pick_canonical([l for (l, c) in v if c == top], voc)
    end

    out = Dict{String,String}()
    for tok in keys(raw)
        chain = [tok]
        cur = tok
        while haskey(raw, cur)
            nxt = raw[cur]
            nxt == cur && break
            if nxt in chain
                cyc = chain[findfirst(==(nxt), chain):end]
                cur = _pick_canonical(cyc, voc)
                break
            end
            push!(chain, nxt)
            cur = nxt
        end
        cur == tok || (out[tok] = cur)
    end

    out
end

"""
    _input_threshold(p::TextProfile, default::Real) -> Float64

The `doc_freq_threshold` `fit` used on `p`, read from its lineage, or `default` when it is not
recorded -- profiles fitted before the threshold was recorded, and merges of merges, whose
summarized lineage drops per-batch params.
"""
function _input_threshold(p::TextProfile, default::Real)
    for s in p.lineage
        s.stage === :fit && haskey(s.params, "doc_freq_threshold") &&
            return Float64(s.params["doc_freq_threshold"])
    end
    Float64(default)
end

"""
    _impute_removed_stopwords(profiles, vocs, voc, pol, doc_freq_threshold) -> Vocabulary

Restores what per-batch stopword removal destroyed, so the merged counters can be read at
corpus scale.

`fit` applies stopwords by tokenizing the batch under `IgnoreStopwords`, so a flagged token
never enters that batch's vocabulary and its counts are simply gone. When *every* input flagged
it there is nothing to do -- it is absent from the merge and stays a stopword. The hard case is
a token some inputs flagged and others did not: the merged counters then hold only the batches
that kept it, which is a fraction of the truth. Measured on Portuguese Wikipedia, 18 of 35
merged stopwords were in that state, `como` among them at df=0.049 against a real corpus df
above 0.5 -- an idf near 3.0 where 0.5 is right.

Neither of the obvious rules works. Dropping such a token deletes content words: on English
Wikipedia 35 of 89 were flagged by at most 2 of 48 batches, `american`, `united`, `states`,
`family` and `history` among them, each made locally ubiquitous by one run of stub articles.
Keeping it with the partial counts is the inflated-idf bug.

So the missing counts are imputed instead. A batch that flagged a token recorded no number, but
it did record a *fact*: the token's document frequency there exceeded that batch's threshold.
`threshold * trainsize` is therefore a real lower bound, and the tightest one available. Only
batches whose vocabulary genuinely lacks the token are imputed for: a profile may *list* a
stopword it never applied, and its counts are then already exact.
Occurrences are scaled by the occs-per-document ratio the batches that kept it observed. Every
token then carries its best available estimate and the corpus-scale threshold decides:
`como` lands at 0.399 and stays as a normal token, `american` at 0.055, `the` was flagged
everywhere and remains a stopword.

The estimate is a lower bound, so a token near the threshold can be judged a normal token when
the truth is just above it. That is the safe direction: idf already drives a
high-document-frequency token's weight toward zero, while deleting a content word is
unrecoverable.
"""
function _impute_removed_stopwords(profiles, vocs, voc::Vocabulary, pol::TextConfig,
                                   doc_freq_threshold::Real)
    extra_ndocs = Dict{String,Int}()
    for (p, v) in zip(profiles, vocs)
        isempty(p.stopwords) && continue
        bound = round(Int, _input_threshold(p, doc_freq_threshold) * gettrainsize(v))
        bound <= 0 && continue
        for t in p.stopwords
            # Impute only where information was actually destroyed. A profile may LIST a
            # stopword without having applied it (`applied.stopwords == false`), and then its
            # own vocabulary still holds the exact counts -- adding a bound on top of those
            # would double-count. Checking the vocabulary tests the fact rather than the
            # declaration, which is also what makes this robust to the two disagreeing.
            token2id(v, t) == 0 || continue
            extra_ndocs[t] = get(extra_ndocs, t, 0) + bound
        end
    end
    isempty(extra_ndocs) && return voc

    N = gettrainsize(voc)
    added_occs = 0
    imputed = Vocabulary(pol, N, getnumtokens(voc))
    for i in eachindex(voc)
        v = voc[i]
        extra = get(extra_ndocs, v.token, 0)
        if extra == 0
            push_token!(imputed, v.token, v.occs, v.ndocs)
        else
            # occurrences per document, as the batches that kept the token measured it
            occs = v.occs + round(Int, extra * (v.occs / max(v.ndocs, 1)))
            added_occs += occs - v.occs
            push_token!(imputed, v.token, occs, min(v.ndocs + extra, N))
        end
    end
    # those occurrences really happened; the batches that removed the token left them out of
    # their own numtokens, so avgdoclen was short by exactly this much
    imputed.numtokens[] = getnumtokens(voc) + added_occs
    imputed
end

# ── merge_profiles ───────────────────────────────────────────────────────────

"""
    merge_profiles(profiles; doc_freq_threshold=0.5, synonyms_k=0, rrf_k=60) -> TextProfile

Merges several [`TextProfile`](@ref)s of one corpus into a single corpus-wide profile:

```julia
p = merge_profiles(load_profile.(paths))
save_profile(dir, p)
```

This is what makes `fit`'s batching usable: batching a large corpus produces one
independent profile per batch, and merging folds them back into the single corpus-wide
profile.

# What is exact, and what is not

- **Vocabulary counts and weights are exact.** `occs`/`ndocs`/`trainsize`/`numtokens` are
  additive across disjoint document batches, and the weighting scheme is *recomputed* from
  the merged counters -- so the merged IDF is the true corpus-wide IDF, not an average of
  per-batch ones. This is the main reason to merge rather than to pick one batch.
- **Synonyms are a rank-fusion consensus, not a recomputation** -- each input's distances
  come from its own embedding space (see [`_fuse_synonyms`](@ref)). Recomputing them
  exactly would need the corpus, or a persisted projection, neither of which a profile
  carries. Fusion scores per *opportunity* rather than summing, because a token absent from an
  input (pruned by `min_ndocs`, or removed as that batch's stopword) cannot appear in any
  neighbour list there, and summing would penalize it for that. Note that no merge can repair
  the missing embedding itself: a token only one input kept has a neighbour list resting on that
  one input's opinion, however it is scored.
- **Lemmas are a plurality vote** over the inputs' clusterings (see [`_vote_lemmas`](@ref)).
- **Stopwords** are recomputed from the merged counters at `doc_freq_threshold`, then
  unioned with the inputs' own sets -- a token every input already removed is absent from the
  merged vocabulary and could not be re-derived, but is still a stopword. A token only *some*
  inputs removed is then dropped from the merged vocabulary: the inputs that removed it never
  recorded its counts, so what survives is a fraction of the truth (measured: `como` at
  df=0.049 against a real corpus df above 0.5), and no merge can reconstruct the rest. What the
  *merged counters* newly flag is only reported, never dropped -- those counts are exact, and
  keeping them is the reason to merge at all. An artifact counts as applied in the merge if any
  input applied it.
- **Lineage** keeps the inputs' stages, one entry per distinct stage with the number of inputs
  that contributed it, followed by the `:merge` step. Merging tuned profiles therefore yields a
  tuned profile; per-batch params are dropped, since they describe batches the merged profile
  no longer has.

Inputs must share their **policy** -- normalization and tokenization -- and their weighting
scheme. Nothing about their artifacts has to match: differing stopword sets union, differing
lemma maps vote, differing networks fuse. That asymmetry is the reason policy and artifacts
are separate concepts. `EntropyWeighting` cannot be merged, since recomputing it needs the
labeled corpus.

`synonyms_k = 0` keeps as many neighbors per token as the richest input had.
"""
function merge_profiles(profiles; doc_freq_threshold::Real=0.5, synonyms_k::Integer=0, rrf_k::Real=60)
    profiles = collect(profiles)
    isempty(profiles) && throw(ArgumentError("merge_profiles: no profiles given"))
    length(profiles) == 1 && @warn "merge_profiles: only one profile given; nothing to merge"

    vocs = [p.model.voc for p in profiles]
    pol = getpolicy(first(profiles))

    for (i, p) in enumerate(profiles)
        q = getpolicy(p)
        _same_normalization(pol.normalization, q.normalization) ||
            error("profile $i has different normalization settings; profiles must share a policy to be merged")
        _same_tokenization(pol.tokenization, q.tokenization) ||
            error("profile $i has different tokenization settings; profiles must share a policy to be merged")
        # Two profiles of different languages have identical normalization and tokenization,
        # so nothing else here can tell them apart: merging Spanish with Portuguese used to
        # succeed silently and produce a model of neither. Only a declared mismatch is
        # refused -- `:unknown` cannot contradict anything.
        (pol.language === :unknown || q.language === :unknown || pol.language === q.language) ||
            error("profile $i is for language :$(q.language) but the first is for " *
                  ":$(pol.language); profiles of different languages cannot be merged")
    end

    gw, lw = first(profiles).model.global_weighting, first(profiles).model.local_weighting
    for (i, p) in enumerate(profiles)
        typeof(p.model.global_weighting) === typeof(gw) && typeof(p.model.local_weighting) === typeof(lw) ||
            error("profile $i uses a different weighting scheme ($(typeof(p.model.global_weighting))/$(typeof(p.model.local_weighting))) than the first ($(typeof(gw))/$(typeof(lw)))")
    end
    gw isa EntropyWeighting &&
        error("cannot merge EntropyWeighting profiles: its weights are supervised and would " *
              "have to be recomputed from the labeled corpus, which a profile does not carry")

    # counts are additive over disjoint batches -- this part of a merge is exact
    voc = Vocabulary(pol, sum(trainsize, vocs), sum(numtokens, vocs))
    for v in vocs
        update_voc!(voc, v)
    end

    voc = _impute_removed_stopwords(profiles, vocs, voc, pol, doc_freq_threshold)

    # What the (imputed) corpus counters flag, plus the tokens no input left any trace of --
    # those cannot be re-derived and are genuinely stopwords. Nothing else: a token whose
    # imputed corpus frequency lands under the threshold is a normal token and must not be
    # listed as a stopword its own vocabulary still contains.
    stopwords = Set{String}(stopword_candidates(voc, doc_freq_threshold))
    for p in profiles, t in p.stopwords
        token2id(voc, t) == 0 && push!(stopwords, t)
    end

    model = VectorModel(gw, lw, voc)   # recomputed from the merged counters

    fused = _fuse_synonyms(profiles, voc, synonyms_k, rrf_k)
    lemmas = _vote_lemmas(profiles, voc)

    # an artifact is applied in the merge if any input applied it
    applied = AppliedArtifacts(
        stopwords = any(p -> p.applied.stopwords, profiles),
        lemmas    = any(p -> p.applied.lemmas, profiles),
        synonyms  = any(p -> p.applied.synonyms, profiles),
    )

    # The inputs' history has to survive the merge, because `istuned` reads nothing else: with
    # only the `:merge` step, merging refitted profiles produced one that reported itself as a
    # base. Their stages are summarized rather than concatenated -- 48 identical `fit` steps
    # are noise, and their per-batch params (trainsize, kappa) describe batches this profile no
    # longer has -- so each distinct stage appears once, in first-appearance order, carrying
    # how many inputs contributed it.
    prior = LineageStep[]
    for p in profiles, step in p.lineage
        any(s -> s.stage === step.stage, prior) && continue
        n = count(q -> any(s -> s.stage === step.stage, q.lineage), profiles)
        push!(prior, LineageStep(step.stage; n_sources=n))
    end
    lineage = push!(prior, LineageStep(:merge; n_sources=length(profiles),
                                              trainsize=gettrainsize(voc)))

    TextProfile(model, stopwords, lemmas, fused.synonyms,
                (isempty(fused.distances) ? nothing : fused.distances),
                applied, lineage)
end
