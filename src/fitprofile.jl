# This file is a part of TextSearch.jl

export fit_profile

"""
    _fit_vocabulary(tc, corpus, min_ndocs; label="", verbose=true) -> Vocabulary

Builds a vocabulary under `tc` and prunes it to tokens in at least `min_ndocs` documents.

The pruning is not cosmetic: the expansion network is an all-pairs kNN over the vocabulary, so
this cuts the most expensive stage of a fit quadratically. Pruning to nothing is an error rather
than an empty model, because it is always a mistake in the threshold and silently returning
nothing wastes whatever comes after.
"""
function _fit_vocabulary(tc::TextConfig, corpus, min_ndocs::Integer;
                         label::AbstractString="", verbose::Bool=true)
    voc = Vocabulary(tc, corpus; verbose=false)
    min_ndocs > 1 || return voc

    before = vocsize(voc)
    voc = filter_tokens(t -> t.ndocs >= min_ndocs, voc)
    verbose && (println("  $(label)vocabulary pruned by min_ndocs=$min_ndocs: $before -> $(vocsize(voc)) tokens");
                flush(stdout))   # long runs are usually watched through a redirected log
    vocsize(voc) > 0 ||
        error("min_ndocs=$min_ndocs pruned the entire vocabulary ($before tokens, none in >= $min_ndocs documents); lower it")
    voc
end

"""
    _remap_expansion_to_lemmas(network, distances, lemmas) -> (; query_expansion, distances)

Rewrites an expansion network's keys and values through `lemmas`.

Needed because the network is derived from the *unlemmatized* vocabulary -- it has to be, since
the lemma map itself comes from embeddings over that vocabulary -- while a profile that applies
lemmas no longer has those tokens. Left alone, every inflected entry would be dropped at query
time in silence.

Entries that collapse onto the same lemma are merged, keeping each candidate's best rank or
distance, and a lemma pointing at itself is dropped. Distances are kept for a token only if every
one of its candidates has one, so the two lists can never fall out of alignment.
"""
function _remap_expansion_to_lemmas(synmap, syndists, lemmas)
    lem(t) = get(lemmas, t, t)
    hasdist = syndists !== nothing
    acc = Dict{String,Dict{String,Tuple{Float64,Union{Nothing,Float32}}}}()

    for (tok, syns) in synmap
        k = lem(tok)
        d = get!(() -> Dict{String,Tuple{Float64,Union{Nothing,Float32}}}(), acc, k)
        dl = hasdist ? get(syndists, tok, nothing) : nothing
        for (rank, syn) in enumerate(syns)
            s = lem(syn)
            s == k && continue
            dist = (dl !== nothing && rank <= length(dl)) ? Float32(dl[rank]) : nothing
            key = dist === nothing ? Float64(rank) : Float64(dist)
            prev = get(d, s, nothing)
            (prev === nothing || key < prev[1]) && (d[s] = (key, dist))
        end
    end

    out = Dict{String,Vector{String}}()
    outd = Dict{String,Vector{Float32}}()
    for (k, d) in acc
        isempty(d) && continue
        cands = sort!(collect(keys(d)); by=c -> (d[c][1], c))
        out[k] = cands
        ds = [d[c][2] for c in cands]
        any(isnothing, ds) || (outd[k] = Float32[Float32(x) for x in ds])
    end

    (; query_expansion=out, distances=(isempty(outd) ? nothing : outd))
end

"""
    fit_profile(textconfig::TextConfig, corpus; kwargs...) -> TextProfile

Distills `corpus` into a portable [`TextProfile`](@ref): vocabulary and counters, weights, a
stopword set, an expansion network and a lemma map, with the lineage recording how.

This exists because the ordering is not obvious and getting it wrong is silent. Three passes over
the corpus, and each one has to be where it is:

1. **Stopwords before the vocabulary.** They are detected from an unfiltered pass whose only
   product is the list of tokens above the threshold, and then removed *while building the
   vocabulary the encoder trains on* -- so a filtered token never enters the counters, the
   factorization, or the network. It is also the single largest stage of a fit: measured on
   272,466 Spanish Wikipedia paragraphs, 36.5s of 123.2s.
2. **The encoder, then the lemmas.** Lemma families are found by clustering token embeddings, so
   the embeddings have to exist first.
3. **The vocabulary again, under the lemma map**, when `lemmas.apply` is set. A lemma is a
   normalization, so it belongs in the `TextConfig` where every consumer applies it to documents
   and queries alike and the idf counts an inflection family together instead of splitting it
   across forms. This pass cannot be folded into an earlier one -- the map is derived from
   embeddings over the vocabulary it rewrites. LSI is deliberately *not* redone afterwards: the
   embeddings' job was to find the families and they did.

# Keywords, grouped as the concerns they belong to

- `min_ndocs = 1` -- drop tokens in fewer documents than this, before the encoder runs.
- `stopwords = (; doc_freq_threshold=0.0, reuse=nothing)` -- `0` disables detection. `reuse`
  takes a set another batch already detected, which is how batches of one corpus end up with
  identical sets and therefore an exact merge (nothing to impute).
- `encoder = (; outdim=256, scaling=:none, factorization=:auto, wordvectors=nothing)` -- LSI
  unless `wordvectors` hands over external embeddings, in which case they are used as they are.
  Reading them from a file is the caller's business; this takes vectors.
- `expansion = (; k=8, head_df=0.0, max_target_ratio=50.0, approx=:auto, construction_recall=0.97,
  search_recall=0.9)` -- see [`query_expansion`](@ref).
- `lemmas = (; apply=false, algorithm=:fft, ...)` -- see [`lemma_clusters`](@ref). `apply=false`
  is the default because a *base* profile computes the map and leaves the choice to whoever tunes
  from it.
- `verbose = true`.

# Example

```julia
julia> p = fit_profile(TextConfig(), corpus; min_ndocs=5, stopwords=(; doc_freq_threshold=0.5));

julia> isbase(p)
true
```
"""
function fit_profile(textconfig::TextConfig, corpus;
                     min_ndocs::Integer=1,
                     stopwords=NamedTuple(),
                     encoder=NamedTuple(),
                     expansion=NamedTuple(),
                     lemmas=NamedTuple(),
                     verbose::Bool=true)
    threshold = Float64(get(stopwords, :doc_freq_threshold, 0.0))
    reuse = get(stopwords, :reuse, nothing)

    # Pass 1
    if threshold > 0 || reuse !== nothing
        candidates = reuse === nothing ?
            stopword_candidates(Vocabulary(textconfig, corpus; verbose=false), threshold) :
            collect(reuse)
        fit_tc = TextConfig(textconfig; pipeline=TokenPipeline(stopwords=Set(candidates)))
    else
        candidates = String[]
        fit_tc = textconfig
    end
    stopwordset = Set{String}(candidates)

    voc = _fit_vocabulary(fit_tc, corpus, min_ndocs; verbose)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)

    # Pass 2
    outdim = Int(get(encoder, :outdim, 256))
    scaling = Symbol(get(encoder, :scaling, :none))
    external = get(encoder, :wordvectors, nothing)
    expopts = (k = Int(get(expansion, :k, 8)),
               approx = Symbol(get(expansion, :approx, :auto)),
               construction_recall = Float64(get(expansion, :construction_recall, 0.97)),
               search_recall = Float64(get(expansion, :search_recall, 0.9)),
               head_df = Float64(get(expansion, :head_df, 0.0)),
               max_target_ratio = Float64(get(expansion, :max_target_ratio, 50.0)))

    wordvecs, net = if external === nothing
        lsi = LatentSemanticIndexing(model, corpus; maxoutdim=outdim, scaling, verbose=false,
                                     factorization=Symbol(get(encoder, :factorization, :auto)))
        wordvectors(lsi), query_expansion(lsi, expopts.k; verbose=false,
            approx=expopts.approx, construction_recall=expopts.construction_recall,
            search_recall=expopts.search_recall, head_df=expopts.head_df,
            max_target_ratio=expopts.max_target_ratio)
    else
        external, query_expansion(voc, external, expopts.k; verbose=false,
            approx=expopts.approx, construction_recall=expopts.construction_recall,
            search_recall=expopts.search_recall, head_df=expopts.head_df,
            max_target_ratio=expopts.max_target_ratio)
    end
    synmap, syndists = net.query_expansion, net.distances

    lemmamap = lemma_clusters(voc, wordvecs;
        algorithm=Symbol(get(lemmas, :algorithm, :fft)),
        num_clusters=Int(get(lemmas, :num_clusters, 0)),
        selector=Symbol(get(lemmas, :selector, :most_frequent)),
        morphology=Symbol(get(lemmas, :morphology, :jaccard)),
        morphology_threshold=Float64(get(lemmas, :morphology_threshold, 0.3)),
        qgram=Int(get(lemmas, :qgram, 2)),
        min_common_prefix=Int(get(lemmas, :min_common_prefix, 3)),
        order=Symbol(get(lemmas, :order, :morphology_first)),
        semantic_threshold=Float64(get(lemmas, :semantic_threshold, 1.0)))

    # Pass 3
    apply_lemmas = Bool(get(lemmas, :apply, false)) && !isempty(lemmamap)
    applied = AppliedArtifacts(stopwords=!isempty(stopwordset), lemmas=apply_lemmas)

    if apply_lemmas
        # The chain order -- lemmas before the stopword filter -- is not decided here: a profile
        # materializes its own config, so this asks one for the config rather than assembling it.
        probe = TextProfile(model; stopwords=stopwordset, lemmas=lemmamap, applied)
        voc = _fit_vocabulary(gettextconfig(probe), corpus, min_ndocs; label="lemmatized ", verbose)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        remapped = _remap_expansion_to_lemmas(synmap, syndists, lemmamap)
        synmap, syndists = remapped.query_expansion, remapped.distances
        verbose && (println("  lemmas applied: $(length(lemmamap)) remapped tokens -> " *
                            "vocsize=$(vocsize(voc)), query_expansion=$(length(synmap))");
                    flush(stdout))
    end

    TextProfile(model; stopwords=stopwordset, lemmas=lemmamap,
                query_expansion=synmap, query_expansion_distances=syndists, applied,
                lineage=[LineageStep(:fit;
                    encoder=(external === nothing ? "lsi" : "external"),
                    outdim, scaling=String(scaling),
                    # where external vectors came from is the caller's knowledge, not this
                    # function's: it takes vectors, not a path
                    source_path=String(get(encoder, :source_path, "")),
                    trainsize=gettrainsize(model.voc),
                    # `merge` needs this: a batch that removed a stopword recorded no count for
                    # it, and threshold*trainsize is the bound that lets the merge impute one
                    doc_freq_threshold=threshold)])
end
