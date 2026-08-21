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
`apply_lemmas` is set and `base` carries a lemma map it did not itself apply, a
[`LemmaTransformation`](@ref) is chained in *first* (see
[`with_lemma_transformation`](@ref)). That is the point of a base profile keeping its lemmas
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
                synonyms=p.synonyms, synonym_distances=p.synonym_distances,
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

    N = gettrainsize(voc)
    out = Vocabulary(voc.textconfig, Int64(N), Int64(getnumtokens(voc)))
    folded = 0
    dropped = 0

    for id in eachindex(voc)
        t = voc[id]
        lemma = get(lemmas, t.token, t.token)
        if lemma != t.token
            if token2id(voc, lemma) == 0
                dropped += 1
                continue
            end
            folded += 1
        end
        push_token!(out, lemma, t.occs, t.ndocs)
    end

    capped = 0
    if N > 0
        @inbounds for i in eachindex(out.ndocs)
            if out.ndocs[i] > N
                out.ndocs[i] = Int32(N)
                capped += 1
            end
        end
    end

    (; voc=out, folded, capped, dropped)
end

"""
    blend_vocabularies(voc_base, voc_sample;
                       kappa::Real=0, keep_rate::Real=1e-5, keep_floor::Integer=3,
                       avgdoclen=:blend)
        -> Vocabulary

Interpolates two vocabularies into one, treating `voc_base` as a **prior worth `kappa`
documents** and `voc_sample` as observed evidence.

# The blend

Read `κ` as "the base is worth this many documents". Both counters are then scaled the same
way -- by the base's average **per document** -- and added to what the sample observed:

```
ndocs(t)  = ndocs_s(t) + round(κ * ndocs_b(t) / N_b)
occs(t)   = occs_s(t)  + round(κ * occs_b(t)  / N_b)
trainsize = N_s + κ
numtokens = sum(occs)                    # recomputed from the survivors
```

`kappa <= 0` defaults to `gettrainsize(voc_sample)`, which weights the two sides equally; halve
it for 1/3 base, double it for 2/3. Expressing the base's authority in documents rather than
as a fraction is what makes the output sample-sized -- so a refitted profile is naturally
lighter than the generic one it came from -- and makes the knob mean something concrete.

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
`κ = N_s` and 56 at `κ = N_s/4`, against the sample's own ~21.

`avgdoclen = :sample` instead sets `numtokens` so the average matches the sample's, and a
positive number sets it to that average directly. This deliberately decouples `numtokens`
from `sum(occs)`, which is safe because that field has exactly one consumer: `avgdoclen`,
and through it `BM25Scorer`'s length normalization. (`TpWeighting` also divides by a
"numtokens", but that one is the *document's* in-vocabulary token count computed per call in
`vectorize!`, not this.) Use it when the profile will index documents shaped like the sample
-- which is the usual reason to refit at all -- and leave it on `:blend` when the base's
documents are representative of what you will index.

# The prune

A token absent from the sample is kept only if the base considered it important:

```
keep(t) = ndocs_s(t) > 0 || (r_b(t) >= keep_rate && ndocs_b(t) >= keep_floor)
```

`keep_rate` is scale-free; `keep_floor` is an absolute floor that stops a token seen in one
or two documents of a huge base corpus -- a typo, an ID -- from clearing a small rate
threshold. Everything surviving must additionally round to `ndocs >= 1`, so a token whose
blended presence is negligible falls out on its own.

Note what needs no rule: a token the base *did* consider important but the sample never
shows keeps only its `κ`-weighted share, so it survives with reduced weight automatically.
Lowering importance is arithmetic; dropping is the only part that needs a decision.
"""
function blend_vocabularies(voc_base::Vocabulary, voc_sample::Vocabulary;
                             kappa::Real=0, keep_rate::Real=1e-5, keep_floor::Integer=3,
                             avgdoclen=:blend)
    N_s = gettrainsize(voc_sample)
    N_b = gettrainsize(voc_base)
    N_s > 0 || throw(ArgumentError("blend_vocabularies: the sample vocabulary has trainsize 0"))

    κ = kappa <= 0 ? Float64(N_s) : Float64(kappa)
    # a vocabulary's counters are Int32, so a prior larger than that cannot be represented;
    # say so here rather than surfacing an InexactError from a rounding deep in the loop
    κ <= typemax(Int32) ||
        throw(ArgumentError("kappa=$kappa exceeds what a vocabulary's Int32 counters can " *
                            "hold (max $(typemax(Int32))); a prior that large would in any " *
                            "case leave the sample no influence at all"))
    N = round(Int64, N_s + κ)

    # numtokens is a placeholder here and recomputed from the survivors below
    out = Vocabulary(voc_sample.textconfig, N, Int64(0))

    # The sample goes in first so the output's token order leads with what was observed.
    for id in eachindex(voc_sample)
        t = voc_sample[id]
        push_token!(out, t.token, t.occs, t.ndocs)
    end

    for id in eachindex(voc_base)
        t = voc_base[id]
        # both counters share the per-document denominator, so occs >= ndocs survives
        r = N_b > 0 ? t.ndocs / N_b : 0.0
        q = N_b > 0 ? t.occs / N_b : 0.0
        insample = token2id(voc_sample, t.token) != 0

        if !insample
            (r >= keep_rate && t.ndocs >= keep_floor) || continue
        end

        nd = round(Int32, κ * r)
        oc = round(Int32, κ * q)
        # a base-only token that contributes nothing measurable is not worth a slot
        (!insample && nd == 0 && oc == 0) && continue
        push_token!(out, t.token, oc, nd)
    end

    # cap before pruning: the cap can only lower a count, never take one below 1
    @inbounds for i in eachindex(out.ndocs)
        out.ndocs[i] > N && (out.ndocs[i] = Int32(N))
    end

    voc = filter_tokens(t -> t.ndocs >= 1, out)
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
    refit_profile(base, sample_voc::Vocabulary; kwargs...) -> NamedTuple
    refit_profile(base, sample_docs; kwargs...) -> NamedTuple

Adapts the bootstrap profile `base` to a dataset, given a sample of it, and returns a new
**self-contained** profile: nothing in the result refers back to `base`, so it can be saved
with [`save_profile`](@ref) and used on its own.

`base` is anything with the fields [`load_profile`](@ref) returns (`model`, `synonyms`,
`synonym_distances`, `lemmas`, `stopword_candidates`, `encoder`) -- a loaded profile, or one
assembled in memory. The return value has that same shape, as [`merge_profiles`](@ref)'s
does.

The first form is the core, and takes a `Vocabulary` the caller built however it liked --
streamed, accumulated across runs with `push_token!`/`update_voc!`, or from a source that is
not a document list at all. It **must** be built under [`refit_textconfig`](@ref)`(base;
apply_lemmas)`, and is checked against it. The second form is a convenience that tokenizes
`sample_docs` for you.

# What is adjusted, and what is not

- **Counters** are interpolated by [`blend_vocabularies`](@ref) and the vocabulary pruned
  there; the weight vector is then *recomputed*, which is what makes the tf-idf and BM25
  paths tuned by one operation rather than only the former.
- **Lemmas** are reused rather than re-derived: the base already paid for them. With
  `apply_lemmas`, they enter the `TextConfig` and the base's counters are folded through the
  same map ([`fold_lemmas`](@ref)) so both sides stay comparable. `extend_lemmas` (corpus
  form only, since it needs to retokenize) additionally recovers families for tokens the base
  never saw, from surface similarity alone -- see [`extend_lemmas_morphological`](@ref).
  Without it those tokens stay unmerged, which is the price of not fitting an embedding.
- **`avgdoclen`** is `:blend` by default and can be pinned to the sample's with `:sample`;
  see [`blend_vocabularies`](@ref) for why that choice matters to BM25.
- **Synonyms** are inherited, restricted to tokens that survived. No embedding is fit here --
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
                        kappa::Real=0, apply_lemmas::Bool=true, lemmas=nothing,
                        keep_rate::Real=1e-5, keep_floor::Integer=3, avgdoclen=:blend,
                        doc_freq_threshold::Real=0.5, verbose::Bool=true)
    gw, lw = base.model.global_weighting, base.model.local_weighting
    gw isa EntropyWeighting &&
        error("cannot refit an EntropyWeighting profile: its weights are supervised and " *
              "would have to be recomputed from the labeled corpus, which a profile does not carry")

    lemmamap = lemmas === nothing ? base.lemmas : lemmas
    tc = refit_textconfig(base; apply_lemmas, lemmas=lemmamap)
    _check_refit_textconfig(tc, sample_voc.textconfig)

    base_voc = base.model.voc
    # Fold only when the refit ADDS a lemma step the base did not have. If the base already
    # lemmatized, its counters are exact and folding again would be wrong. The marker says
    # this directly now, instead of being inferred from the shape of the pipeline.
    if apply_lemmas && !isempty(lemmamap) && !base.applied.lemmas
        f = fold_lemmas(base_voc, lemmamap)
        base_voc = f.voc
        verbose && println(stderr,
            "refit: folded $(f.folded) base gettoken(s) into their lemmas " *
            "($(vocsize(base.model.voc)) -> $(vocsize(base_voc)) tokens; " *
            "$(f.dropped) dropped whose lemma was not in the base vocabulary; " *
            "ndocs capped at trainsize for $(f.capped))")
    end

    voc = blend_vocabularies(base_voc, sample_voc; kappa, keep_rate, keep_floor, avgdoclen)

    syn, sdist = _restrict_synonyms(base.synonyms, base.synonym_distances, voc)
    # Restricted to entries whose target survived the prune. No reconciliation step follows:
    # the profile constructor materializes the TextConfig from THIS map, so the applied map
    # and the saved map are the same object by construction. (They used to be assembled
    # separately, and shipped at 110,393 versus 40,320 entries on a real profile.)
    kept_lemmas = Dict{String,String}(
        tok => lemma for (tok, lemma) in lemmamap if token2id(voc, lemma) != 0)

    model = VectorModel(gw, lw, voc)

    stopwords = Set{String}(stopword_candidates(voc, doc_freq_threshold))
    union!(stopwords, base.stopwords)

    κ = kappa <= 0 ? Float64(gettrainsize(sample_voc)) : Float64(kappa)
    applied = AppliedArtifacts(stopwords=base.applied.stopwords,
                               lemmas=(apply_lemmas && !isempty(kept_lemmas)),
                               synonyms=base.applied.synonyms)
    lineage = LineageStep[base.lineage...,
                          LineageStep(:refit; kappa=κ,
                                              sample_trainsize=gettrainsize(sample_voc),
                                              trainsize=gettrainsize(voc),
                                              lemmas_applied=applied.lemmas)]

    if verbose
        fromsample = count(id -> token2id(sample_voc, gettoken(voc, id)) != 0, eachindex(voc))
        println(stderr,
            "refit: vocsize $(vocsize(base.model.voc)) (base) + $(vocsize(sample_voc)) (sample) " *
            "-> $(vocsize(voc)); $fromsample gettoken(s) seen in the sample, " *
            "$(vocsize(voc) - fromsample) carried from the base alone")
        # TextSearch.avgdoclen, qualified deliberately: the `avgdoclen` KEYWORD shadows the
        # function of that name throughout this body, and calling it bare is a MethodError
        # ("objects of type Symbol are not callable") that only fires when verbose is on.
        println(stderr,
            "refit: kappa=$(round(κ; digits=1)) documents of prior against a " *
            "$(gettrainsize(sample_voc))-document sample -> trainsize=$(gettrainsize(voc)), " *
            "avgdoclen=$(round(TextSearch.avgdoclen(voc); digits=2)), " *
            "lemmas=$(applied.lemmas ? "applied" : "carried only")")
    end

    TextProfile(model, stopwords, kept_lemmas, syn, sdist, applied, lineage)
end

function refit_profile(base::TextProfile, sample_docs; apply_lemmas::Bool=true, extend_lemmas::Bool=false,
                        morphology::Symbol=:jaccard, morphology_threshold::Real=0.3,
                        qgram::Integer=2, min_common_prefix::Integer=3,
                        lemma_selector::Symbol=:most_frequent,
                        verbose::Bool=true, kwargs...)
    lemmamap = base.lemmas
    tc = refit_textconfig(base; apply_lemmas, lemmas=lemmamap)
    sample_voc = Vocabulary(tc, sample_docs; verbose=false)

    if extend_lemmas && apply_lemmas
        ext = _extend_lemmas_from_sample(base, sample_voc, lemmamap;
                                        morphology, morphology_threshold, qgram,
                                        min_common_prefix, selector=lemma_selector)
        if !isempty(ext)
            lemmamap = merge(Dict{String,String}(lemmamap), ext)
            # Retokenize rather than folding the vocabulary we already have: the sample is
            # small by definition, so a second pass is cheap and exact, where folding would
            # over-count ndocs for any document holding two forms of a newly-found family.
            tc = refit_textconfig(base; apply_lemmas, lemmas=lemmamap)
            sample_voc = Vocabulary(tc, sample_docs; verbose=false)
            verbose && println(stderr,
                "refit: extended the lemma map with $(length(ext)) morphological entr" *
                "$(length(ext) == 1 ? "y" : "ies") for gettoken(s) the base had not seen")
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
    _same_normalization(expected.normalization, got.normalization) ||
        error("the sample vocabulary was built with different normalization settings than " *
              "the refit requires; build it with refit_textconfig(base; apply_lemmas)")
    _same_tokenization(expected.tokenization, got.tokenization) ||
        error("the sample vocabulary was built with different tokenization settings than " *
              "the refit requires; build it with refit_textconfig(base; apply_lemmas)")
    # The transformation is compared by its ARTIFACTS, which is all it can hold now: a lemma
    # map and a stopword set. Both come from one place -- `gettextconfig(profile)` -- so the only
    # way to fail this is to have tokenized the sample under some other config entirely, which
    # is exactly the mistake worth catching loudly.
    _same_artifacts(expected.transformation, got.transformation) ||
        error("the sample vocabulary was built with a different lemma map or stopword set " *
              "than the refit requires; build it with refit_textconfig(base; apply_lemmas)")
    nothing
end

"""
    _same_artifacts(a, b) -> Bool

Whether two materialized transformations apply the same artifacts. Compares the lemma map and
the stopword set by value, ignoring how the chain happens to be nested -- a lone
`IgnoreStopwords` and a one-element chain around it are the same pipeline.
"""
_same_artifacts(a::AbstractTokenTransformation, b::AbstractTokenTransformation) =
    _artifact_pair(a) == _artifact_pair(b)

_artifact_pair(::IdentityTokenTransformation) = (nothing, nothing)
_artifact_pair(t::LemmaTransformation) = (t.lemmas, nothing)
_artifact_pair(t::IgnoreStopwords) = (nothing, t.stopwords)
function _artifact_pair(t::ChainTransformation)
    lem = nothing
    sw = nothing
    for s in t.list
        l, w = _artifact_pair(s)
        l === nothing || (lem = l)
        w === nothing || (sw = w)
    end
    (lem, sw)
end

"""
    _restrict_synonyms(synonyms, distances, voc) -> (synonyms, distances)

Drops every network entry naming a token absent from `voc`, keeping rank order and the
parallel distances aligned.

Necessary because the refit prunes: an entry left pointing at a dropped token would be
discarded at query time by `expand_synonyms!` without a word (`token2id` returning `0`), so
it would cost file size and tell a reader the network is richer than it is.
"""
function _restrict_synonyms(synonyms, distances, voc::Vocabulary)
    out = Dict{String,Vector{String}}()
    outd = Dict{String,Vector{Float32}}()

    for (tok, neighbors) in synonyms
        token2id(voc, tok) == 0 && continue
        dl = distances === nothing ? nothing : get(distances, tok, nothing)
        words = String[]
        ds = Float32[]
        for (rank, syn) in enumerate(neighbors)
            token2id(voc, syn) == 0 && continue
            push!(words, syn)
            dl !== nothing && rank <= length(dl) && push!(ds, Float32(dl[rank]))
        end
        isempty(words) && continue
        out[tok] = words
        # all or nothing, so the lists can never fall out of alignment
        length(ds) == length(words) && (outd[tok] = ds)
    end

    out, (isempty(outd) ? nothing : outd)
end
