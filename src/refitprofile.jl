# This file is a part of TextSearch.jl

export refit_profile, refit_textconfig, fold_lemmas, blend_vocabularies

# ── refit ────────────────────────────────────────────────────────────────────
#
# A profile fit from a large generic corpus (Wikipedia, say) is a *bootstrap* model:
# reasonable statistics for a language, not a model for anyone's dataset. Refitting adapts
# one to a specific dataset from a sample of it, and emits a new, self-contained profile.
#
# This is NOT `merge_profiles`. A merge folds disjoint batches of one corpus, where counts
# simply add and the result is exact. A refit combines two views of *different* corpora with
# deliberately unequal authority, so it interpolates rather than sums, and it prunes.
#
# The blend works on the vocabulary's raw counters rather than on the model's derived weight
# vector, which matters: BM25 never reads `VectorModel.weight` -- `tokenscore` computes its
# own idf from `ndocs`/`trainsize` and normalizes by `avgdoclen`. Blending weights alone
# would tune the tf-idf path and leave BM25 with the base corpus' numbers. Blending counters
# tunes both, and the weight follows by recomputation.

"""
    refit_textconfig(base; apply_lemmas::Bool=true, lemmas=nothing) -> TextConfig

The `TextConfig` a refit of `base` runs under, and the one a caller building its own sample
`Vocabulary` **must** tokenize with.

This is public because it is an invariant, not an implementation detail: the blend
interpolates two vocabularies token by token, so both sides have to be produced by the same
normalization, tokenization, stopword set and lemma step. Tokenizing a sample under anything
else silently compares tokens that do not correspond, and the resulting numbers mean nothing.

Everything is inherited from `base` unchanged, with one deliberate exception: when
`apply_lemmas` is set and `base` carries a lemma map it did not itself apply, that map enters
the config's [`TokenPipeline`](@ref), whose lemma stage runs *before* its stopword stage --
the reverse order silently readmits stopwords, since `"las"` is not in a set holding `"la"`
until after it is rewritten. That is the point of a base profile keeping its lemmas
unapplied -- whether to lemmatize belongs to the refit, and a tuned model that declines it
simply does not carry the map. When lemmas are added here, [`refit_profile`](@ref) folds the
base's own counts through the same map so both sides stay comparable.

`lemmas` overrides which map is applied, defaulting to `base.lemmas`. That is what lets a
caller lemmatize under a map *extended* beyond the base's -- see
[`extend_lemmas_morphological`](@ref) -- while keeping everything else about the config
identical.

See also [`refit_profile`](@ref), [`fold_lemmas`](@ref).
"""
function refit_textconfig(base::TextProfile; apply_lemmas::Bool=true, lemmas=nothing)
    p = lemmas === nothing ? base : _with_lemmas(base, lemmas)
    gettextconfig(with_applied(p; lemmas=apply_lemmas))
end

# a profile with a different lemma map, for the extension path (Part: extend_lemmas)
_with_lemmas(p::TextProfile, lemmas) =
    TextProfile(p.model; stopwords=p.stopwords, lemmas,
                query_expansion=p.query_expansion, query_expansion_distances=p.query_expansion_distances,
                applied=p.applied, lineage=p.lineage)

"""
    fold_lemmas(voc::Vocabulary, lemmas) -> (; voc, folded, capped, dropped)

Rewrites `voc`'s tokens through `lemmas`, merging each inflection family's counters into its
lemma. Used to bring a base vocabulary that was built *without* a lemma step onto the same
footing as a sample tokenized *with* one.

The two counters fold differently, and only one is exact:

- `occs` is exact. Occurrences are additive, so a family's total occurrence count is the sum
  of its forms'.
- `ndocs` **overestimates**. A document containing both `"casa"` and `"casas"` counts once
  for each, but once folded it should count once for `"casa"` -- and a vocabulary carries no
  co-occurrence information to correct with.

That overestimate is why every `ndocs` is capped at `trainsize`. The cap is a correctness
requirement, not tidiness: `ndocs > trainsize` makes idf negative
(`log2((0.5+trainsize)/(0.5+ndocs))`) and drives BM25's numerator
(`trainsize - ndocs + 0.5`) below zero. `capped` reports how often it bit, so the
approximation stays visible instead of assumed harmless.

A token whose lemma is absent from `voc` is **dropped** rather than reintroduced: that
happens when the lemma was itself filtered out at fit time (a stopword, or pruned as rare),
and resurrecting it here would smuggle back a token the pipeline deliberately excludes.
`folded` counts remapped tokens, `dropped` the discarded ones.
"""
function fold_lemmas(voc::Vocabulary, lemmas)
    isempty(lemmas) && return (; voc, folded=0, capped=0, dropped=0)

    trainsize = gettrainsize(voc)
    merged = Vocabulary(voc.textconfig, Int64(trainsize), Int64(getnumtokens(voc)))
    folded = 0
    dropped = 0

    for id in eachindex(voc)
        entry = voc[id]
        lemma = get(lemmas, entry.token, entry.token)
        if lemma != entry.token
            if token2id(voc, lemma) == 0
                dropped += 1
                continue
            end
            folded += 1
        end
        push_token!(merged, lemma, entry.occs, entry.ndocs)
    end

    capped = 0
    if trainsize > 0
        @inbounds for i in eachindex(merged.ndocs)
            if merged.ndocs[i] > trainsize
                merged.ndocs[i] = Int32(trainsize)
                capped += 1
            end
        end
    end

    (; voc=merged, folded, capped, dropped)
end

"""
    blend_vocabularies(voc_base, voc_sample; kappa=nothing, min_ndocs::Integer=1,
                       avgdoclen=:blend)
        -> Vocabulary

Interpolates two vocabularies into one, treating `voc_base` as a **prior worth `kappa`
documents** and `voc_sample` as observed evidence.

# The blend

Read `kappa` as "the base is worth this many documents". Both counters are then scaled the
same way -- by the base's average **per document** -- and added to what the sample observed:

```
base_doc_rate(t) = ndocs_base(t) / N_base       # fraction of base documents holding t
base_occ_rate(t) = occs_base(t)  / N_base       # occurrences of t per base document

ndocs(t)  = ndocs_sample(t) + round(kappa * base_doc_rate(t))
occs(t)   = occs_sample(t)  + round(kappa * base_occ_rate(t))
trainsize = N_sample + kappa
numtokens = sum(occs)                           # recomputed from the survivors
```

`kappa = nothing` (the default) means `N_sample`, which weights the two sides equally; halve
it for 1/3 base, double it for 2/3. Expressing the base's authority in documents rather than
as a fraction is what makes the knob mean something concrete -- and it is the only spelling,
since the fraction `w` is just `kappa = N_sample * w / (1 - w)` and one knob with two units
is one knob too many.

The knob is weak, which is worth knowing before reaching for it. Swept against 1,000
known-item queries at three sample sizes, an 18x range of the base's effective weight
(`kappa / (N_sample + kappa)`, from 0.048 to 0.926) moved recall@10 by at most 1.5 points,
and never against the base: more prior was mildly better at every sample size, including one
20x larger than the base's own influence would suggest. `min_ndocs` moves 29 points on the
same measurement. Set `kappa` when you have a reason; the default is not costing you
anything measurable.

`kappa` sets weight, and only weight. It used to decide membership as well -- a base-only
token whose `round(kappa * base_doc_rate)` came out zero simply vanished -- so the
vocabulary shrank hardest
at small samples, which is exactly where the base is the only evidence there is. What the
vocabulary keeps is the gate's decision now.

Using the same per-document denominator for both counters is what keeps the result a
*possible* corpus. Scaling `occs` by the base's share of total tokens instead (`occs_b/T_b`)
looks equally reasonable and is not: the two counters then round against different
denominators, and a token carried from the base lands with `ndocs >= 1` but `occs == 0` --
present in documents yet never occurring. Sharing the denominator preserves each token's
occurrences-per-document ratio, so `occs >= ndocs` holds by construction.

# avgdoclen

By default (`avgdoclen = :blend`) `numtokens` is the sum of the surviving `occs`, so
`avgdoclen` comes out as a weighted mean of the two corpora's average document lengths. That
is the honest reading of the blend -- the pseudo-documents the prior contributes are base
documents, and they are as long as base documents are. But it moves BM25's length
normalization toward the base, and when the two corpora's documents are nothing alike the
effect is large: Wikipedia-es against 400 product reviews lands at 141 tokens/document at
`kappa = N_sample` and 56 at `kappa = N_sample/4`, against the sample's own ~21.

`avgdoclen = :sample` instead sets `numtokens` so the average matches the sample's, and a
positive number sets it to that average directly. This deliberately decouples `numtokens`
from `sum(occs)`, which is safe because that field has exactly one consumer: `avgdoclen`,
and through it `BM25Scorer`'s length normalization. (`TpWeighting` also divides by a
"numtokens", but that one is the *document's* in-vocabulary token count computed per call in
`vectorize!`, not this.) Use it when the profile will index documents shaped like the sample
-- which is the usual reason to refit at all -- and leave it on `:blend` when the base's
documents are representative of what you will index.

# The gate

A token absent from the sample is kept when the base saw it in enough documents:

```
keep(t) = ndocs_sample(t) > 0 || ndocs_base(t) >= min_ndocs
```

`min_ndocs` is the same knob [`fit_profile`](@ref) applies to its own corpus, in the same
unit and with the same meaning: how many documents of evidence a token needs to be in a
vocabulary. There is deliberately only one, and it is a document count rather than a rate,
because that is the unit the evidence arrives in -- "seen in at least 12 base documents" is
something a caller can reason about, where a rate cannot be read at all without knowing the
base's size. It is also what keeps a token seen in one or two documents of a huge corpus --
a typo, an ID -- out of the result.

[`refit_profile`](@ref) defaults it to the bar the base's own fit was run at, read from its
lineage, so everything the base has is kept: the fit already decided what counts as
attested, and a refit silently re-imposing a different bar would overrule that with a number
the caller never chose. Raising it is how a caller asks for a smaller profile; lowering it
below the fit's bar does nothing, because the tokens it would admit were never in the base.

Whatever the gate keeps is then representable: `ndocs` is floored at one document. That floor
is what makes `min_ndocs` a control rather than a suggestion, because without it `kappa`
decides membership, and it decides it backwards.

Measured on a 6,068-token base (16,640 documents) refitted against samples of a different
corpus, scoring 1,000 known-item queries against a 10,000-document index:

| sample | decided by | vocsize | recall@10 | recall@1 | archive |
|---|---|---|---|---|---|
| 100 documents | rounding (old) | 663 | 0.632 | 0.376 | 18 KB |
| 100 documents | the gate | 6,284 | 0.921 | 0.799 | 227 KB |
| 2,000 documents | either | 9,824 | 0.949 | 0.842 | 257 KB |

(Archive sizes are deflated zips; a stored one is about 3x each, and the ratios between them
are what the knob trades, not the absolute figures.)

The two agree from about `kappa > N_base / (2 * min ndocs_base)` upward -- 1,664 for that
base -- since above it the rounding was keeping everything anyway. Below it, the gap is the
difference between a profile that answers and one that cannot.

The cost is real, and trading it is what `min_ndocs` is for: a refitted profile is no longer
automatically sample-sized, and at `kappa = 100` above it costs 12x the bytes. The curve
is steep
at the cheap end, on the same measurement -- `min_ndocs = 12` keeps 84% of the recall gain
for 29% of the bytes, `min_ndocs = 9` keeps 91% for 41% -- so a caller who needs a small
profile raises it and reads what it costs. That is a decision; letting `kappa` make it
silently was not.

Note what needs no rule: a token the base *did* consider important but the sample never
shows keeps only its `kappa`-weighted share, so it survives with reduced weight
automatically.
Lowering importance is arithmetic; dropping is the only part that needs a decision.
"""
function blend_vocabularies(voc_base::Vocabulary, voc_sample::Vocabulary;
                             kappa=nothing, min_ndocs::Integer=1, avgdoclen=:blend)
    N_sample = gettrainsize(voc_sample)
    N_base = gettrainsize(voc_base)
    N_sample > 0 || throw(ArgumentError("blend_vocabularies: the sample vocabulary has trainsize 0"))

    kappa === nothing || kappa > 0 ||
        throw(ArgumentError("kappa must be positive, or `nothing` for the sample's own " *
                            "document count; got $kappa"))
    kappa = kappa === nothing ? Float64(N_sample) : Float64(kappa)
    # a vocabulary's counters are Int32, so a prior larger than that cannot be represented;
    # say so here rather than surfacing an InexactError from a rounding deep in the loop
    kappa <= typemax(Int32) ||
        throw(ArgumentError("kappa=$kappa exceeds what a vocabulary's Int32 counters can " *
                            "hold (max $(typemax(Int32))); a prior that large would in any " *
                            "case leave the sample no influence at all"))
    trainsize = round(Int64, N_sample + kappa)

    # numtokens is a placeholder here and recomputed from the survivors below
    blended = Vocabulary(voc_sample.textconfig, trainsize, Int64(0))

    # The sample goes in first so the output's token order leads with what was observed.
    for id in eachindex(voc_sample)
        entry = voc_sample[id]
        push_token!(blended, entry.token, entry.occs, entry.ndocs)
    end

    for id in eachindex(voc_base)
        entry = voc_base[id]
        # both counters share the per-document denominator, so occs >= ndocs survives
        base_doc_rate = N_base > 0 ? entry.ndocs / N_base : 0.0
        base_occ_rate = N_base > 0 ? entry.occs / N_base : 0.0

        if token2id(voc_sample, entry.token) == 0
            (N_base > 0 && entry.ndocs >= min_ndocs) || continue
            # A token the gate kept has to be representable, and for a small sample the
            # kappa-scaled count of a base-only token is routinely below one document:
            # `round` alone sent it to zero and the token disappeared. Floor it instead.
            ndocs = max(one(Int32), round(Int32, kappa * base_doc_rate))
            push_token!(blended, entry.token,
                        max(ndocs, round(Int32, kappa * base_occ_rate)), ndocs)
        else
            push_token!(blended, entry.token,
                        round(Int32, kappa * base_occ_rate),
                        round(Int32, kappa * base_doc_rate))
        end
    end

    # cap before the check below: the cap can only lower a count, never take one below 1
    @inbounds for i in eachindex(blended.ndocs)
        blended.ndocs[i] > trainsize && (blended.ndocs[i] = Int32(trainsize))
    end

    # Not a prune any more -- the gate above already decided, and nothing reaching here can
    # be at zero. It stays as the structural guarantee that `ndocs == 0` never reaches the
    # weighting, where `log2((0.5 + trainsize) / (0.5 + ndocs))` would make an unobserved
    # token the single heaviest in the model: 10.64 bits against 9.06 for the heaviest real
    # one, a gap that is 1.58 bits at every trainsize.
    voc = filter_tokens(t -> t.ndocs >= 1, blended)
    voc.numtokens[] = _blended_numtokens(avgdoclen, voc, voc_sample)
    voc
end

"""
    _blended_numtokens(avgdoclen, voc, voc_sample) -> Int64

Resolves `blend_vocabularies`' `avgdoclen` option into the `numtokens` to store:
`:blend` sums the surviving occurrences, `:sample` matches the sample's average document
length, and a positive number is used as that average directly.
"""
function _blended_numtokens(avgdoclen, voc::Vocabulary, voc_sample::Vocabulary)
    total = Int64(sum(voc.occs; init=Int64(0)))
    avgdoclen === :blend && return total
    target = if avgdoclen === :sample
        TextSearch.avgdoclen(voc_sample)
    elseif avgdoclen isa Real && avgdoclen > 0
        Float64(avgdoclen)
    else
        throw(ArgumentError("avgdoclen must be :blend, :sample, or a positive number; " *
                            "got $(repr(avgdoclen))"))
    end
    # No `max(vocsize, ...)` floor here, though "at least one occurrence per token" reads like
    # an obvious sanity bound. It is the wrong bound for this field: a blended vocabulary
    # routinely holds more tokens than its nominal corpus could contain (the base contributes
    # tokens whose rate rounds to a single document), so the floor wins for any realistic base
    # and silently turns the override into a no-op -- measured at 18.4 instead of the sample's
    # 9.16 on Wikipedia-es. What describes the corpus is `occs`, which is untouched; in
    # override mode `numtokens` is purely the number `avgdoclen` divides, i.e. a BM25
    # length-normalization parameter, and it is only useful if it is obeyed.
    max(Int64(1), round(Int64, gettrainsize(voc) * target))
end

"""
    _fit_min_ndocs(p::TextProfile, default::Integer) -> Int

The `min_ndocs` the fit that produced `p` was run at, read from its lineage, or `default`
when it is not recorded -- profiles fitted before it was recorded, and bases assembled in
memory rather than fitted.

This is what lets a refit adopt the bar the profile was actually built at instead of a
constant of its own. Note that adopting it can never delete anything: `_fit_vocabulary`
already pruned below it, so every token the base holds clears it by construction. That is the
property worth having in a default -- it is named rather than magic, and it cannot surprise.
"""
function _fit_min_ndocs(p::TextProfile, default::Integer)
    for s in p.lineage
        s.stage === :fit && haskey(s.params, "min_ndocs") &&
            return Int(s.params["min_ndocs"])
    end
    Int(default)
end

"""
    refit_profile(base, sample_voc::Vocabulary; kwargs...) -> NamedTuple
    refit_profile(base, sample_docs; kwargs...) -> NamedTuple

Adapts the bootstrap profile `base` to a dataset, given a sample of it, and returns a new
**self-contained** profile: nothing in the result refers back to `base`, so it can be saved
with [`save_profile`](@ref) and used on its own.

`base` is anything with the fields [`load_profile`](@ref) returns (`model`, `query_expansion`,
`query_expansion_distances`, `lemmas`, `stopword_candidates`, `encoder`) -- a loaded profile, or one
assembled in memory. The return value has that same shape, as [`merge_profiles`](@ref)'s
does.

The first form is the core, and takes a `Vocabulary` the caller built however it liked --
streamed, accumulated across runs with `push_token!`/`update_voc!`, or from a source that is
not a document list at all. It **must** be built under [`refit_textconfig`](@ref)`(base;
apply_lemmas)`, and is checked against it. The second form is a convenience that tokenizes
`sample_docs` for you.

# What is adjusted, and what is not

- **Counters** are interpolated by [`blend_vocabularies`](@ref), which also decides what the
  vocabulary keeps -- `min_ndocs`, the same document count `fit_profile` uses, is that
  control; the weight
  vector is then *recomputed*, which is what makes the tf-idf and BM25 paths tuned by one
  operation rather than only the former.
- **Lemmas** are reused rather than re-derived: the base already paid for them. With
  `apply_lemmas`, they enter the `TextConfig` and the base's counters are folded through the
  same map ([`fold_lemmas`](@ref)) so both sides stay comparable. `extend_lemmas` (corpus
  form only, since it needs to retokenize) additionally recovers families for tokens the base
  never saw, from surface similarity alone -- see [`extend_lemmas_morphological`](@ref).
  Without it those tokens stay unmerged, which is the price of not fitting an embedding.
- **`avgdoclen`** is `:blend` by default and can be pinned to the sample's with `:sample`;
  see [`blend_vocabularies`](@ref) for why that choice matters to BM25.
- **Query expansion** are inherited, restricted to tokens that survived. No embedding is fit here --
  that is exactly what makes a refit cheap next to a fit, and the point of bootstrapping.
- **Stopword candidates** are recomputed from the blended counters, but the *applied* stopword
  set stays the base's. It has to: the base's counts were collected under that set, and
  swapping it mid-blend would compare two incomparable vocabularies. New candidates are
  reported for review, the same detected-versus-applied split the profile format already has.

`EntropyWeighting` is rejected, as it is for a merge: its weights are supervised and cannot
be re-derived from a profile's contents.

Set `verbose` to see the vocabulary sizes, how much of the result the base accounts for, and
the fold/cap counts from any lemma folding.
"""
function refit_profile(base::TextProfile, sample_voc::Vocabulary;
                        kappa=nothing, apply_lemmas::Bool=true, lemmas=nothing,
                        min_ndocs=nothing, avgdoclen=:blend,
                        doc_freq_threshold::Real=0.5, verbose::Bool=true)
    global_weighting, local_weighting = base.model.global_weighting, base.model.local_weighting
    global_weighting isa EntropyWeighting &&
        error("cannot refit an EntropyWeighting profile: its weights are supervised and " *
              "would have to be recomputed from the labeled corpus, which a profile does not carry")

    lemmamap = lemmas === nothing ? base.lemmas : lemmas
    textconfig = refit_textconfig(base; apply_lemmas, lemmas=lemmamap)
    _check_refit_textconfig(textconfig, sample_voc.textconfig)

    base_voc = base.model.voc
    # Fold only when the refit ADDS a lemma step the base did not have. If the base already
    # lemmatized, its counters are exact and folding again would be wrong. The marker says
    # this directly now, instead of being inferred from the shape of the pipeline.
    if apply_lemmas && !isempty(lemmamap) && !base.applied.lemmas
        folding = fold_lemmas(base_voc, lemmamap)
        base_voc = folding.voc
        verbose && println(stderr,
            "refit: folded $(folding.folded) base token(s) into their lemmas " *
            "($(vocsize(base.model.voc)) -> $(vocsize(base_voc)) tokens; " *
            "$(folding.dropped) dropped whose lemma was not in the base vocabulary; " *
            "ndocs capped at trainsize for $(folding.capped))")
    end

    # `nothing` means "the bar this profile was built at", which is the honest default: the
    # refit does not get to invent an evidence threshold the fit never chose.
    fit_min_ndocs = _fit_min_ndocs(base, 1)
    resolved_min_ndocs = min_ndocs === nothing ? fit_min_ndocs : Int(min_ndocs)
    voc = blend_vocabularies(base_voc, sample_voc; kappa, min_ndocs=resolved_min_ndocs, avgdoclen)

    expansion, expansion_distances =
        _restrict_query_expansion(base.query_expansion, base.query_expansion_distances, voc)
    # Restricted to entries whose target survived the prune. No reconciliation step follows:
    # the profile constructor materializes the TextConfig from THIS map, so the applied map
    # and the saved map are the same object by construction. (They used to be assembled
    # separately, and shipped at 110,393 versus 40,320 entries on a real profile.)
    kept_lemmas = Dict{String,String}(
        tok => lemma for (tok, lemma) in lemmamap if token2id(voc, lemma) != 0)

    model = VectorModel(global_weighting, local_weighting, voc)

    stopwords = Set{String}(stopword_candidates(voc, doc_freq_threshold))
    union!(stopwords, base.stopwords)

    prior_docs = kappa === nothing ? Float64(gettrainsize(sample_voc)) : Float64(kappa)
    applied = AppliedArtifacts(stopwords=base.applied.stopwords,
                               lemmas=(apply_lemmas && !isempty(kept_lemmas)),
                               query_expansion=base.applied.query_expansion)
    lineage = LineageStep[base.lineage...,
                          LineageStep(:refit; kappa=prior_docs,
                                              sample_trainsize=gettrainsize(sample_voc),
                                              trainsize=gettrainsize(voc),
                                              lemmas_applied=applied.lemmas)]

    if verbose
        from_sample = count(id -> token2id(sample_voc, gettoken(voc, id)) != 0, eachindex(voc))
        # How many carried tokens sit at the one-document floor says how much of the result
        # rests on the gate rather than on evidence -- the number `min_ndocs` trades against.
        at_floor = count(eachindex(voc)) do id
            getndocs(voc, id) == 1 && token2id(sample_voc, gettoken(voc, id)) == 0
        end
        println(stderr,
            "refit: vocsize $(vocsize(base.model.voc)) (base) + $(vocsize(sample_voc)) (sample) " *
            "-> $(vocsize(voc)); $from_sample token(s) seen in the sample, " *
            "$(vocsize(voc) - from_sample) carried from the base alone " *
            "($at_floor of them at the one-document floor)")
        println(stderr,
            "refit: min_ndocs=$resolved_min_ndocs " *
            (min_ndocs === nothing ? "(the bar the base's own fit used); raise it to carry fewer" :
             resolved_min_ndocs < fit_min_ndocs ?
                "(the base's fit used $fit_min_ndocs, so nothing below that exists to keep)" :
                "(the base's fit used $fit_min_ndocs)"))
        # TextSearch.avgdoclen, qualified deliberately: the `avgdoclen` KEYWORD shadows the
        # function of that name throughout this body, and calling it bare is a MethodError
        # ("objects of type Symbol are not callable") that only fires when verbose is on.
        println(stderr,
            "refit: kappa=$(round(prior_docs; digits=1)) documents of prior against a " *
            "$(gettrainsize(sample_voc))-document sample -> trainsize=$(gettrainsize(voc)), " *
            "avgdoclen=$(round(TextSearch.avgdoclen(voc); digits=2)), " *
            "lemmas=$(applied.lemmas ? "applied" : "carried only")")
    end

    TextProfile(model, stopwords, kept_lemmas, expansion, expansion_distances, applied, lineage)
end

function refit_profile(base::TextProfile, sample_docs; apply_lemmas::Bool=true, extend_lemmas::Bool=false,
                        morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                        qgram::Integer=2, min_common_prefix::Integer=3,
                        lemma_selector::Symbol=:most_frequent,
                        verbose::Bool=true, kwargs...)
    lemmamap = base.lemmas
    textconfig = refit_textconfig(base; apply_lemmas, lemmas=lemmamap)
    sample_voc = Vocabulary(textconfig, sample_docs; verbose=false)

    if extend_lemmas && apply_lemmas
        ext = _extend_lemmas_from_sample(base, sample_voc, lemmamap;
                                        morphology, morphology_threshold, qgram,
                                        min_common_prefix, selector=lemma_selector)
        if !isempty(ext)
            lemmamap = merge(Dict{String,String}(lemmamap), ext)
            # Retokenize rather than folding the vocabulary we already have: the sample is
            # small by definition, so a second pass is cheap and exact, where folding would
            # over-count ndocs for any document holding two forms of a newly-found family.
            textconfig = refit_textconfig(base; apply_lemmas, lemmas=lemmamap)
            sample_voc = Vocabulary(textconfig, sample_docs; verbose=false)
            verbose && println(stderr,
                "refit: extended the lemma map with $(length(ext)) morphological entr" *
                "$(length(ext) == 1 ? "y" : "ies") for token(s) the base had not seen")
        end
    end

    refit_profile(base, sample_voc; apply_lemmas, lemmas=lemmamap, verbose, kwargs...)
end

"""
    _extend_lemmas_from_sample(base, sample_voc, lemmamap; kwargs...) -> Dict{String,String}

Finds lemma entries for the tokens `sample_voc` brings that `base`'s vocabulary never had.

The grouping runs over the base and sample vocabularies **merged**, not over the sample
alone, and that is the point: a new inflected form usually belongs to a family whose lemma
the base already knows, so `"audifonos"` must be able to elect the base's `"audifono"`. The
merged counts also make `:most_frequent` prefer the established form over the newcomer.
Only the new tokens get entries (see [`extend_lemmas_morphological`](@ref)), so the base's
own clustering decisions are never overruled.
"""
function _extend_lemmas_from_sample(base::TextProfile, sample_voc::Vocabulary, lemmamap; kwargs...)
    bvoc = base.model.voc
    new = String[gettoken(sample_voc, id) for id in eachindex(sample_voc)
                 if token2id(bvoc, gettoken(sample_voc, id)) == 0]
    isempty(new) && return Dict{String,String}()
    extend_lemmas_morphological(merge_voc(bvoc, sample_voc), lemmamap; candidates=new, kwargs...)
end

"""
    _check_refit_textconfig(expected::TextConfig, got::TextConfig)

Errors unless `got` is the config a refit requires, comparing field by field via the same
predicates a merge uses -- `==` on these structs is unreliable (see the note atop
`mergeprofiles.jl`).

Worth checking loudly: a sample tokenized under a different config produces tokens that do
not correspond to the base's, and the blend would then quietly interpolate unrelated
counters instead of failing.
"""
function _check_refit_textconfig(expected::TextConfig, got::TextConfig)
    expected.normalization == got.normalization ||
        error("the sample vocabulary was built with different normalization settings than " *
              "the refit requires; build it with refit_textconfig(base; apply_lemmas)")
    _same_tokenization(expected.tokenization, got.tokenization) ||
        error("the sample vocabulary was built with different tokenization settings than " *
              "the refit requires; build it with refit_textconfig(base; apply_lemmas)")
    # The pipeline is plain data -- a lemma map and a stopword set -- so this is a value
    # comparison. It used to need a pair of helpers whose whole job was to normalize over how a
    # `ChainTransformation` happened to be nested; a fixed pipeline has nothing to normalize.
    # Both sides come from one place, `gettextconfig(profile)`, so the only way to fail here is
    # to have tokenized the sample under some other config entirely -- exactly the mistake worth
    # catching loudly.
    expected.pipeline == got.pipeline ||
        error("the sample vocabulary was built with a different lemma map or stopword set " *
              "than the refit requires; build it with refit_textconfig(base; apply_lemmas)")
    nothing
end


"""
    _restrict_query_expansion(query_expansion, distances, voc) -> (query_expansion, distances)

Drops every network entry naming a token absent from `voc`, keeping rank order and the
parallel distances aligned.

Necessary because the refit prunes: an entry left pointing at a dropped token would be
discarded at query time by `expand_query!` without a word (`token2id` returning `0`), so
it would cost file size and tell a reader the network is richer than it is.
"""
function _restrict_query_expansion(query_expansion, distances, voc::Vocabulary)
    out = Dict{String,Vector{String}}()
    outd = Dict{String,Vector{Float32}}()

    for (tok, neighbors) in query_expansion
        token2id(voc, tok) == 0 && continue
        dl = distances === nothing ? nothing : get(distances, tok, nothing)
        words = String[]
        ds = Float32[]
        for (rank, neighbor) in enumerate(neighbors)
            token2id(voc, neighbor) == 0 && continue
            push!(words, neighbor)
            dl !== nothing && rank <= length(dl) && push!(ds, Float32(dl[rank]))
        end
        isempty(words) && continue
        out[tok] = words
        # all or nothing, so the lists can never fall out of alignment
        length(ds) == length(words) && (outd[tok] = ds)
    end

    out, (isempty(outd) ? nothing : outd)
end
