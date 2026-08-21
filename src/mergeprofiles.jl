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

_same_transformation(::IdentityTokenTransformation, ::IdentityTokenTransformation) = true
_same_transformation(a::IgnoreStopwords, b::IgnoreStopwords) = a.stopwords == b.stopwords
_same_transformation(a::LemmaTransformation, b::LemmaTransformation) = a.lemmas == b.lemmas

function _same_transformation(a::SnowballTokenTransformation, b::SnowballTokenTransformation)
    (hasproperty(a.stemmer, :alg) && hasproperty(b.stemmer, :alg)) || return false
    a.stemmer.alg == b.stemmer.alg && a.stemmer.enc == b.stemmer.enc
end

_same_transformation(a::ChainTransformation, b::ChainTransformation) =
    length(a.list) == length(b.list) && all(_same_transformation(x, y) for (x, y) in zip(a.list, b.list))

_same_transformation(_, _) = false

"""
    _merge_transformations(tts) -> AbstractTokenTransformation

Resolves the token transformation of a merged profile. Identical transformations are kept
as-is. Transformations that differ *only* in their stopword set -- the normal case when
each batch detected its own stopwords from its own slice -- collapse to the union of those
sets, since a token any batch removed is absent from that batch's vocabulary and so is
already excluded from the merged counts. Anything else (differing stemmers, differing
chains) cannot be reconciled and is an error.
"""
function _merge_transformations(tts)
    tt1 = first(tts)
    all(t -> _same_transformation(tt1, t), tts) && return tt1

    if all(t -> t isa IgnoreStopwords || t isa IdentityTokenTransformation, tts)
        u = Set{String}()
        for t in tts
            t isa IgnoreStopwords && union!(u, t.stopwords)
        end
        return IgnoreStopwords(u)
    end

    error("cannot merge profiles with incompatible token transformations: " *
          "$(unique(typeof.(tts))). Only identical transformations, or ones differing " *
          "solely in their IgnoreStopwords set, can be merged.")
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

    for p in profiles
        pd = get(p, :synonym_distances, nothing)
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
        # a candidate's mean distance, or `nothing` when no input reported one for it
        dtok = get(dists, tok, nothing)
        function meandist(c)
            dtok === nothing && return nothing
            ds = get(dtok, c, nothing)
            ds === nothing ? nothing : sum(ds) / length(ds)
        end

        cands = collect(keys(s))
        # highest fused score first; ties broken deterministically (closer mean distance
        # when known, then lexicographically) so a merge is reproducible regardless of Dict
        # ordering. Candidates without a distance sort after those with one, rather than
        # comparing `nothing` against a number.
        sort!(cands; by=c -> (-s[c], something(meandist(c), Inf), c))
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
    bestkey = (-occs(voc, token2id(voc, best)), length(best), best)
    for t in tokens
        key = (-occs(voc, token2id(voc, t)), length(t), t)
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

# ── merge_profiles ───────────────────────────────────────────────────────────

"""
    merge_profiles(profiles; doc_freq_threshold=0.5, synonyms_k=0, rrf_k=60)
        -> (; model, synonyms, synonym_distances, lemmas, stopword_candidates, encoder)

Merges several profiles (each a [`load_profile`](@ref) result) into one, returning exactly
the pieces [`save_profile`](@ref) takes as keywords, so a merged profile is written with

```julia
p = merge_profiles(load_profile.(paths))
save_profile(dir, p.model; p.synonyms, p.synonym_distances, p.lemmas,
             p.stopword_candidates, p.encoder)
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
  carries.
- **Lemmas are a plurality vote** over the inputs' clusterings (see [`_vote_lemmas`](@ref)).
- **Stopword candidates** are recomputed from the merged counters at `doc_freq_threshold`,
  then unioned with the inputs' recorded candidates -- a token every input already removed
  is absent from the merged vocabulary and could not be re-derived, but is still a stopword
  and stays recorded as one.

Inputs must share their normalization and tokenization settings, and their weighting
scheme; transformations may differ only in their stopword set. `EntropyWeighting` cannot be
merged, since recomputing it needs the labeled corpus.

`synonyms_k = 0` keeps as many neighbors per token as the richest input had.
"""
function merge_profiles(profiles; doc_freq_threshold::Real=0.5, synonyms_k::Integer=0, rrf_k::Real=60)
    profiles = collect(profiles)
    isempty(profiles) && throw(ArgumentError("merge_profiles: no profiles given"))
    length(profiles) == 1 && @warn "merge_profiles: only one profile given; nothing to merge"

    vocs = [p.model.voc for p in profiles]
    tc1 = first(vocs).textconfig

    for (i, voc) in enumerate(vocs)
        _same_normalization(tc1.normalization, voc.textconfig.normalization) ||
            error("profile $i has different normalization settings; profiles must be fit with the same TextConfig to be merged")
        _same_tokenization(tc1.tokenization, voc.textconfig.tokenization) ||
            error("profile $i has different tokenization settings; profiles must be fit with the same TextConfig to be merged")
        tc1.expand_query_synonyms === voc.textconfig.expand_query_synonyms ||
            error("profile $i has a different expand_query_synonyms setting than the first profile")
    end

    gw, lw = first(profiles).model.global_weighting, first(profiles).model.local_weighting
    for (i, p) in enumerate(profiles)
        typeof(p.model.global_weighting) === typeof(gw) && typeof(p.model.local_weighting) === typeof(lw) ||
            error("profile $i uses a different weighting scheme ($(typeof(p.model.global_weighting))/$(typeof(p.model.local_weighting))) than the first ($(typeof(gw))/$(typeof(lw)))")
    end
    gw isa EntropyWeighting &&
        error("cannot merge EntropyWeighting profiles: its weights are supervised and would " *
              "have to be recomputed from the labeled corpus, which a profile does not carry")

    transformation = _merge_transformations([voc.textconfig.transformation for voc in vocs])
    textconfig = TextConfig(tc1; transformation)

    # counts are additive over disjoint batches -- this part of a merge is exact
    voc = Vocabulary(textconfig, sum(trainsize, vocs), sum(numtokens, vocs))
    for v in vocs
        update_voc!(voc, v)
    end

    model = VectorModel(gw, lw, voc)   # recomputed from the merged counters

    fused = _fuse_synonyms(profiles, voc, synonyms_k, rrf_k)
    lemmas = _vote_lemmas(profiles, voc)

    candidates = Set{String}(stopword_candidates(voc, doc_freq_threshold))
    for p in profiles
        union!(candidates, p.stopword_candidates)
    end

    source_kinds = unique(String[
        p.encoder === nothing ? "unknown" : String(get(p.encoder, "kind", "unknown"))
        for p in profiles
    ])
    encoder = (; kind=:merged, n_sources=length(profiles), source_kinds=join(sort(source_kinds), ","))

    (; model, synonyms=fused.synonyms,
       synonym_distances=(isempty(fused.distances) ? nothing : fused.distances),
       lemmas, stopword_candidates=sort!(collect(candidates)), encoder)
end
