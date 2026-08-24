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

    kind = Symbol(enc["kind"])
    kind in (:lsi, :external) ||
        error("unknown encoder kind: $(enc["kind"]); supported: lsi, external")

    # The app's job is to turn a config file into arguments; the pipeline itself is
    # `fit_profile`, in the library, so anyone using TextSearch gets the same three passes in
    # the same order without reading this file.
    external_path = get(enc, "external_path", "")
    wordvecs = nothing
    if kind === :external
        voc = Vocabulary(_fit_textconfig(cfg), docs; verbose=false)
        wordvecs, oov = _load_external_embeddings(external_path, voc)
        oov > 0 && @warn "textsearch fit: $oov / $(vocsize(voc)) vocabulary tokens missing from external embeddings; using zero vectors for them"
    end

    profile = fit_profile(_fit_textconfig(cfg), docs;
        min_ndocs = Int(get(get(cfg, "vocabulary", Dict()), "min_ndocs", 1)),
        stopwords = (doc_freq_threshold = sw["enabled"] ? Float64(sw["doc_freq_threshold"]) : 0.0,
                     reuse = sw["enabled"] ? reuse : nothing),
        encoder   = (outdim = Int(enc["outdim"]),
                     scaling = Symbol(enc["scaling"]),
                     factorization = Symbol(get(enc, "factorization", "auto")),
                     wordvectors = wordvecs,
                     source_path = external_path),
        expansion = (k = Int(syn["k"]),
                     approx = _query_expansion_approx(get(syn, "approx", "auto")),
                     construction_recall = Float64(get(syn, "construction_recall", 0.97)),
                     search_recall = Float64(get(syn, "search_recall", 0.9)),
                     head_df = Float64(get(syn, "head_df", 0.0)),
                     max_target_ratio = Float64(get(syn, "max_target_ratio", 50.0))),
        lemmas    = (apply = Bool(get(lem, "apply", false)),
                     algorithm = Symbol(lem["algorithm"]),
                     num_clusters = Int(lem["num_clusters"]),
                     selector = Symbol(lem["selector"]),
                     morphology = Symbol(get(lem, "morphology", "jaccard")),
                     morphology_threshold = Float64(get(lem, "morphology_threshold", 0.3)),
                     qgram = Int(get(lem, "qgram", 2)),
                     min_common_prefix = Int(get(lem, "min_common_prefix", 3)),
                     order = Symbol(get(lem, "order", "morphology_first")),
                     semantic_threshold = Float64(get(lem, "semantic_threshold", 1.0))))

    save_profile(batch_dir, profile)
    vocsize(profile.model.voc), profile.model, sort!(collect(profile.stopwords))
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
            # Zipped under a temporary name and renamed only on success. `resume` above decides
            # by existence, and existence cannot tell finished from interrupted: a run killed
            # while writing the zip would leave a truncated part that the next run skips and the
            # merge then reads. Same reason `prepare` renames its JSONL into place.
            partial = zippath * ".partial"
            rm(partial; force=true)
            zip_profile(batch_dir, partial)
            mv(partial, zippath; force=true)
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
