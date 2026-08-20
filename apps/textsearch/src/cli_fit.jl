function parse_fit_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch fit",
        description="Fit a TextSearch profile (vocabulary, weights, synonyms, lemmas, " *
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
        key = Symbol(token(voc, tid))
        if haskey(mapping, key)
            X[:, tid] .= Float32.(mapping[key])
        else
            oov += 1
        end
    end
    MatrixDatabase(X), oov
end

"""
    _synonyms_approx(v::AbstractString) -> Union{Symbol,Bool}

Maps the config's `[synonyms] approx` string onto what `TextSearch.synonyms` expects:
`"auto"` -> `:auto` (approximate only once the vocabulary is big enough to need it),
`"always"` -> `true`, `"never"` -> `false`.
"""
function _synonyms_approx(v::AbstractString)
    v == "auto"   && return :auto
    v == "always" && return true
    v == "never"  && return false
    error("invalid [synonyms] approx = $(repr(v)); expected \"auto\", \"always\", or \"never\"")
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
    )
end

"""
    _fit_one_batch(docs::Vector{String}, cfg, batch_dir::AbstractString) -> (vocsize::Int, model)

Runs the full `fit` pipeline over one batch of document texts and saves the resulting
profile (uncompressed) into `batch_dir`. See `cmd_fit`'s docstring / the project plan for
the stopword-before-vocabulary ordering rationale.
"""
function _fit_one_batch(docs::Vector{String}, cfg, batch_dir::AbstractString)
    sw = cfg["stopwords"]
    enc = cfg["encoder"]
    syn = cfg["synonyms"]
    lem = cfg["lemmas"]

    base_textconfig = _fit_textconfig(cfg)

    if sw["enabled"]
        voc0 = Vocabulary(base_textconfig, docs; verbose=false)
        candidates = stopword_candidates(voc0, Float64(sw["doc_freq_threshold"]))
        textconfig = TextConfig(base_textconfig; transformation=IgnoreStopwords(Set(candidates)))
    else
        textconfig = base_textconfig
        candidates = String[]
    end

    voc = Vocabulary(textconfig, docs; verbose=false)

    # Prune rare tokens before anything expensive touches the vocabulary: the synonym
    # network is an all-pairs search over it, so this is quadratic savings, and tokens seen
    # in one or two documents have no usable embedding to begin with.
    min_ndocs = Int(get(get(cfg, "vocabulary", Dict()), "min_ndocs", 1))
    if min_ndocs > 1
        before = vocsize(voc)
        voc = filter_tokens(t -> t.ndocs >= min_ndocs, voc)
        println("  vocabulary pruned by min_ndocs=$min_ndocs: $before -> $(vocsize(voc)) tokens")
        flush(stdout)   # long runs are usually watched through a redirected log
        vocsize(voc) > 0 ||
            error("min_ndocs=$min_ndocs pruned the entire vocabulary ($before tokens, none in >= $min_ndocs documents); lower it")
    end

    model = VectorModel(IdfWeighting(), TfWeighting(), voc)

    kind = Symbol(enc["kind"])
    outdim = Int(enc["outdim"])
    scaling = Symbol(enc["scaling"])
    external_path = get(enc, "external_path", "")

    synopts = (
        approx = _synonyms_approx(get(syn, "approx", "auto")),
        construction_recall = Float64(get(syn, "construction_recall", 0.97)),
        search_recall = Float64(get(syn, "search_recall", 0.9)),
    )

    lsiopts = (factorization = Symbol(get(enc, "factorization", "auto")),)

    wordvecs, synmap = if kind === :lsi
        lsi = LatentSemanticIndexing(model, docs; maxoutdim=outdim, scaling, verbose=false, lsiopts...)
        wordvectors(lsi), synonyms(lsi, Int(syn["k"]); verbose=false, synopts...)
    elseif kind === :external
        wv, oov = _load_external_embeddings(external_path, voc)
        oov > 0 && @warn "textsearch fit: $oov / $(vocsize(voc)) vocabulary tokens missing from external embeddings; using zero vectors for them"
        wv, synonyms(voc, wv, Int(syn["k"]); verbose=false, synopts...)
    else
        error("unknown encoder kind: $(enc["kind"]); supported: lsi, external")
    end

    lemmas = lemma_clusters(voc, wordvecs;
        algorithm=Symbol(lem["algorithm"]), num_clusters=Int(lem["num_clusters"]),
        selector=Symbol(lem["selector"]),
        morphology=Symbol(get(lem, "morphology", "jaccard")),
        morphology_threshold=Float64(get(lem, "morphology_threshold", 0.3)),
        qgram=Int(get(lem, "qgram", 2)),
        min_common_prefix=Int(get(lem, "min_common_prefix", 3)))

    save_profile(batch_dir, model;
        synonyms=synmap, lemmas, stopword_candidates=candidates,
        encoder=(; kind, outdim, scaling, source_path=external_path))

    vocsize(voc), model
end

function cmd_fit(args::Vector{String})
    o = parse_fit_args(args)
    cfg = load_fit_config(o["config"])

    input = cfg["input"]
    output = cfg["output"]
    format = Symbol(input["format"])

    mkpath(output["dir"])

    n = _each_batch(each_record(format, input["path"], input["text_key"]), Int(output["batch_size"])) do i, docs
        batch_dir = joinpath(output["dir"], "_textsearch_fit_batch_$(lpad(i, 4, '0'))")
        try
            m, _ = _fit_one_batch(docs, cfg, batch_dir)
            zippath = joinpath(output["dir"], "$(output["prefix"])-$(lpad(i, 4, '0')).zip")
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
