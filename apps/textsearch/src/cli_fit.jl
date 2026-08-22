function parse_fit_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch fit",
        description="Fit a TextSearch profile (vocabulary, weights, query_expansion, lemmas, " *
                     "stopword candidates) from a corpus. Options are edited as a TOML " *
                     "config file, visudo-style, rather than passed as flags -- pass " *
                     "--config to skip the \$EDITOR flow and read a config file directly.")
    @add_arg_table! s begin
        "--config"
            help = "path to a fit config TOML file; skips launching \$EDITOR"
    end
    parse_args(args, s)
end

"""
    _each_batch(f, itr, batch_size::Integer) -> Int

Calls `f(batch_index, docs::Vector{String})` for each chunk of at most `batch_size`
document texts pulled from an `each_record`-style iterator (`batch_size <= 0` means one
unbounded chunk), and returns how many chunks were produced.

Batches are yielded and released **as they fill**, never accumulated: a corpus far larger
than memory (all of Wikipedia, say) only ever costs one batch's worth of documents at a
time, since each batch's profile is independent anyway.
"""
function _each_batch(f::Function, itr, batch_size::Integer)
    nbatches = 0
    buf = String[]
    for p in itr
        push!(buf, first(p))
        if batch_size > 0 && length(buf) == batch_size
            nbatches += 1
            f(nbatches, buf)
            buf = String[]   # a fresh buffer: `f` may retain the one it was handed
        end
    end
    if !isempty(buf)
        nbatches += 1
        f(nbatches, buf)
    end
    nbatches
end

"""
    _load_external_embeddings(path::AbstractString, voc::Vocabulary) -> (MatrixDatabase, oov::Int)

Loads a `token -> vector` JSON mapping (`{"cat": [0.1, 0.2, ...], ...}`) and builds a
`(dim, vocsize(voc))` `MatrixDatabase` by looking up each of `voc`'s tokens by name
(the external vocabulary generally isn't the same as `voc`'s, built fresh from this
corpus). Missing tokens get a zero vector; `oov` counts them.
"""
function _load_external_embeddings(path::AbstractString, voc)
    mapping = JSON3.read(read(path))
    dim = length(first(v for (_, v) in pairs(mapping)))
    m = vocsize(voc)
    X = zeros(Float32, dim, m)
    oov = 0
    for tid in 1:m
        key = Symbol(gettoken(voc, tid))
        if haskey(mapping, key)
            X[:, tid] .= Float32.(mapping[key])
        else
            oov += 1
        end
    end
    MatrixDatabase(X), oov
end

"""
    _query_expansion_approx(v::AbstractString) -> Union{Symbol,Bool}

Maps the config's `[query_expansion] approx` string onto what `TextSearch.query_expansion` expects:
`"auto"` -> `:auto` (approximate only once the vocabulary is big enough to need it),
`"always"` -> `true`, `"never"` -> `false`.
"""
function _query_expansion_approx(v::AbstractString)
    v == "auto"   && return :auto
    v == "always" && return true
    v == "never"  && return false
    error("invalid [query_expansion] approx = $(repr(v)); expected \"auto\", \"always\", or \"never\"")
end

function _fit_textconfig(cfg)
    norm = cfg["normalization"]
    tok = cfg["tokenization"]
    TextConfig(
        normalization=NormalizationConfig(;
            del_diac=norm["del_diac"], del_dup=norm["del_dup"], del_punc=norm["del_punc"],
            group_num=norm["group_num"], group_url=norm["group_url"], group_usr=norm["group_usr"],
            group_emo=norm["group_emo"], lc=norm["lc"],
        ),
        tokenization=TokenizationConfig(nlist=Int8.(tok["nlist"]), mark_token_type=tok["mark_token_type"]),
        language=Symbol(get(tok, "language", "unknown")),
    )
end

"""
    _build_vocabulary(tc, docs, min_ndocs; label="") -> Vocabulary

Tokenizes `docs` under `tc` into a `Vocabulary`, then drops tokens appearing in
fewer than `min_ndocs` documents.

Pruning happens before anything expensive touches the vocabulary: the query_expansion network is an
all-pairs search over it, so this is a quadratic saving, and a token seen in one or two
documents has no usable embedding to begin with. `label` distinguishes the passes in the
progress output.
"""
function _build_vocabulary(tc, docs::Vector{String}, min_ndocs::Int; label::AbstractString="")
    voc = Vocabulary(tc, docs; verbose=false)
    min_ndocs > 1 || return voc

    before = vocsize(voc)
    voc = filter_tokens(t -> t.ndocs >= min_ndocs, voc)
    println("  $(label)vocabulary pruned by min_ndocs=$min_ndocs: $before -> $(vocsize(voc)) tokens")
    flush(stdout)   # long runs are usually watched through a redirected log
    vocsize(voc) > 0 ||
        error("min_ndocs=$min_ndocs pruned the entire vocabulary ($before tokens, none in >= $min_ndocs documents); lower it")
    voc
end

"""
    _remap_query_expansion_to_lemmas(synmap, syndists, lemmas) -> (; query_expansion, distances)

Rewrites a query_expansion network's keys and values through `lemmas`, for use when the lemma map
is baked into the profile's `TextConfig` and the vocabulary is therefore lemmatized.

Without this the network would silently stop working: its entries name unlemmatized forms,
which are no longer tokens of the vocabulary, and `expand_query!` drops an out-of-
vocabulary query_expansion without complaint (`token2id` returning `0`) -- a quiet loss of every
expansion whose surface form happened to be inflected.

Two source tokens can share a lemma, so entries are merged rather than overwritten. What
"best" means depends on what the network carries: with `syndists`, the smallest distance
wins; without it, the smallest *rank* does -- the rank-based analogue, and the reason the
network stays usable when distances were never stored. A query_expansion that lemmatizes onto its
own key is dropped, since a token is not its own query_expansion.

Each list comes back in rank order (nearest first), matching how `TextSearch.query_expansion`
produces them. `distances` is `nothing` when the input had none.
"""
function _remap_query_expansion_to_lemmas(synmap, syndists, lemmas)
    lem(t) = get(lemmas, t, t)
    hasdist = syndists !== nothing
    # per lemma key: candidate => (ordering key, distance-or-nothing)
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
        # all or nothing per token, so the two lists can never fall out of alignment
        any(isnothing, ds) || (outd[k] = Float32[Float32(x) for x in ds])
    end

    (; query_expansion=out, distances=(isempty(outd) ? nothing : outd))
end

"""
    _fit_one_batch(docs::Vector{String}, cfg, batch_dir::AbstractString; reuse=nothing)
        -> (vocsize::Int, model, stopwords::Vector{String})

Runs the full `fit` pipeline over one batch of document texts and saves the resulting
profile (uncompressed) into `batch_dir`. See `cmd_fit`'s docstring / the project plan for
the stopword-before-vocabulary ordering rationale.
"""
function _fit_one_batch(docs::Vector{String}, cfg, batch_dir::AbstractString; reuse=nothing)
    sw = cfg["stopwords"]
    enc = cfg["encoder"]
    syn = cfg["query_expansion"]
    lem = cfg["lemmas"]

    base_textconfig = _fit_textconfig(cfg)

    if sw["enabled"]
        # `reuse` is the stopword set a previous part already detected, or `nothing` to detect
        # here. Detection needs a full tokenization pass whose only product is the list of tokens
        # above the threshold, and it is the largest single stage of a fit -- measured on 10,000
        # Spanish articles, 35.6s of 90.9s (39%), and on their 272,466 paragraphs 36.5s of 123.2s.
        # See `cmd_fit` for why reusing one part's set across the rest is sound.
        candidates = reuse === nothing ?
            stopword_candidates(Vocabulary(base_textconfig, docs; verbose=false),
                                Float64(sw["doc_freq_threshold"])) :
            collect(reuse)
        fit_tc = TextConfig(base_textconfig; transformation=IgnoreStopwords(Set(candidates)))
    else
        fit_tc = base_textconfig
        candidates = String[]
    end

    min_ndocs = Int(get(get(cfg, "vocabulary", Dict()), "min_ndocs", 1))
    voc = _build_vocabulary(fit_tc, docs, min_ndocs)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)

    kind = Symbol(enc["kind"])
    outdim = Int(enc["outdim"])
    scaling = Symbol(enc["scaling"])
    external_path = get(enc, "external_path", "")

    synopts = (
        approx = _query_expansion_approx(get(syn, "approx", "auto")),
        construction_recall = Float64(get(syn, "construction_recall", 0.97)),
        search_recall = Float64(get(syn, "search_recall", 0.9)),
        head_df = Float64(get(syn, "head_df", 0.0)),
        max_target_ratio = Float64(get(syn, "max_target_ratio", 50.0)),
    )

    lsiopts = (factorization = Symbol(get(enc, "factorization", "auto")),)

    wordvecs, net = if kind === :lsi
        lsi = LatentSemanticIndexing(model, docs; maxoutdim=outdim, scaling, verbose=false, lsiopts...)
        wordvectors(lsi), query_expansion(lsi, Int(syn["k"]); verbose=false, synopts...)
    elseif kind === :external
        wv, oov = _load_external_embeddings(external_path, voc)
        oov > 0 && @warn "textsearch fit: $oov / $(vocsize(voc)) vocabulary tokens missing from external embeddings; using zero vectors for them"
        wv, query_expansion(voc, wv, Int(syn["k"]); verbose=false, synopts...)
    else
        error("unknown encoder kind: $(enc["kind"]); supported: lsi, external")
    end
    synmap, syndists = net.query_expansion, net.distances

    lemmas = lemma_clusters(voc, wordvecs;
        algorithm=Symbol(lem["algorithm"]), num_clusters=Int(lem["num_clusters"]),
        selector=Symbol(lem["selector"]),
        morphology=Symbol(get(lem, "morphology", "jaccard")),
        morphology_threshold=Float64(get(lem, "morphology_threshold", 0.3)),
        qgram=Int(get(lem, "qgram", 2)),
        min_common_prefix=Int(get(lem, "min_common_prefix", 3)),
        order=Symbol(get(lem, "order", "morphology_first")),
        semantic_threshold=Float64(get(lem, "semantic_threshold", 1.0)))

    # Third pass: bake the lemma map into the TextConfig and rebuild vocabulary/weights
    # under it. A lemma is a normalization, so this is where it belongs -- once it is in the
    # TextConfig, every consumer (vectorize, bagofwords, the inverted files, search) applies
    # it to documents and queries alike, and the idf counts a whole inflection family
    # together instead of splitting it across its forms.
    #
    # The lemma map cannot be known before this point: it is derived from embeddings over
    # the vocabulary it now rewrites, so this pass cannot be folded into an earlier one. LSI
    # is deliberately NOT redone on the lemmatized vocabulary -- the embeddings' job was to
    # discover the families, and they have; re-deriving them would only shift query_expansion
    # neighbours slightly for the cost of a full factorization.
    apply_lemmas = Bool(get(lem, "apply", false)) && !isempty(lemmas)
    stopwords = Set(candidates)
    applied = AppliedArtifacts(stopwords=sw["enabled"], lemmas=apply_lemmas)

    if apply_lemmas
        # Rebuild the vocabulary under the lemma map. The chain order (lemmas before the
        # stopword filter) is not decided here: a TextProfile materializes its own config, so
        # this asks a profile for the config rather than assembling one.
        probe = TextProfile(model; stopwords, lemmas, applied)
        voc = _build_vocabulary(gettextconfig(probe), docs, min_ndocs; label="lemmatized ")
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        # the network's entries name unlemmatized forms, which are no longer vocabulary
        # tokens; left alone, every inflected entry would be silently dropped at query time
        remapped = _remap_query_expansion_to_lemmas(synmap, syndists, lemmas)
        synmap, syndists = remapped.query_expansion, remapped.distances
        println("  lemmas applied: $(length(lemmas)) remapped tokens -> vocsize=$(vocsize(voc)), query_expansion=$(length(synmap))")
        flush(stdout)
    end

    profile = TextProfile(model; stopwords, lemmas,
                          query_expansion=synmap, query_expansion_distances=syndists, applied,
                          lineage=[LineageStep(:fit; encoder=String(kind), outdim, scaling=String(scaling),
                                                     source_path=external_path,
                                                     trainsize=gettrainsize(model.voc),
                                                     # `merge` needs this: a batch that removed
                                                     # a stopword recorded no count for it, and
                                                     # threshold*trainsize is the bound that
                                                     # lets the merge impute one
                                                     doc_freq_threshold=(sw["enabled"] ?
                                                        Float64(sw["doc_freq_threshold"]) : 0.0))])
    save_profile(batch_dir, profile)

    vocsize(voc), model, candidates
end

function cmd_fit(args::Vector{String})
    o = parse_fit_args(args)
    cfg = load_fit_config(o["config"])

    input = cfg["input"]
    output = cfg["output"]
    format = Symbol(input["format"])

    mkpath(output["dir"])

    resume = Bool(get(output, "resume", false))

    # The first part detects stopwords with a full pass; every later part reuses that set.
    #
    # Sound because a stopword is a property of the language, not of the batch, and verified:
    # across five 400k-paragraph batches drawn from two different Spanish shards, the first
    # batch's 31 tokens are a SUPERSET of every other batch's set and equal their union exactly
    # -- no later batch would have flagged anything the first one missed, and reuse removes at
    # most five extra function words (`entre este lo son sus`).
    #
    # Wikipedia makes that easy by ordering articles longest-first: longer paragraphs hold more
    # function words, so the first batch maximizes their document frequency by construction. A
    # corpus in ARBITRARY order has no such guarantee, and one whose first batch is topically
    # narrow could miss a genuine stopword. The remedy is not another knob but a bigger
    # `batch_size` (equivalently, fewer parts): a larger first batch is a better sample of the
    # corpus, and the detection cost is paid once regardless of how many parts follow.
    #
    # It also makes merging exact. When every part removes the same set, no token is removed in
    # some parts and kept in others, so no counts are partial and `merge_profiles` has nothing to
    # impute -- which is the entire bug class its imputation exists to paper over.
    shared_stopwords::Union{Nothing,Vector{String}} = nothing

    n = _each_batch(each_record(format, input["path"], input["text_key"]), Int(output["batch_size"])) do i, docs
        zippath = joinpath(output["dir"], "$(output["prefix"])-$(lpad(i, 4, '0')).zip")

        # Each part is written as soon as it is fitted, so an interrupted run leaves the
        # finished ones on disk. `resume` then skips refitting those -- the batch's documents
        # are still read (cheap) so later parts keep the same boundaries, only the fit
        # (the expensive part) is skipped. Off by default: silently reusing a profile fitted
        # under different settings would be worse than redoing the work.
        if resume && isfile(zippath)
            println("part $i already present, skipping fit (resume=true) -> $zippath")
            flush(stdout)
            return
        end

        batch_dir = joinpath(output["dir"], "_textsearch_fit_batch_$(lpad(i, 4, '0'))")
        try
            m, _, sw = _fit_one_batch(docs, cfg, batch_dir; reuse=shared_stopwords)
            if shared_stopwords === nothing
                shared_stopwords = sw
                isempty(sw) || println("  detected $(length(sw)) stopwords; later parts reuse them")
            end
            zip_profile(batch_dir, zippath)
            println("saved profile $i ($(length(docs)) docs, vocsize=$m) -> $zippath")
            flush(stdout)
        finally
            rm(batch_dir; recursive=true, force=true)
        end
    end

    n == 0 && error("no documents found in $(input["path"]) (format=$(input["format"]))")
    # NB: return the exit code, not the batch count -- `main` uses an Integer return value
    # as the process exit status, so returning `n` here reported success as failure
    # (1 batch -> exit 1), which `set -e` callers such as corpora/wikipedia.sh treat as a
    # failed fit.
    0
end
