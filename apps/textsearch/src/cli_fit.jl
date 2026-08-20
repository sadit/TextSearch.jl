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
    _batches(itr, batch_size::Integer) -> Vector{Vector{String}}

Splits an `each_record`-style iterator of `(text, record) => ...` pairs into chunks of at
most `batch_size` document texts (`batch_size <= 0` means one unbounded chunk).
"""
function _batches(itr, batch_size::Integer)
    batches = Vector{String}[]
    buf = String[]
    for p in itr
        push!(buf, first(p))
        if batch_size > 0 && length(buf) == batch_size
            push!(batches, buf)
            buf = String[]
        end
    end
    isempty(buf) || push!(batches, buf)
    batches
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
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)

    kind = Symbol(enc["kind"])
    outdim = Int(enc["outdim"])
    scaling = Symbol(enc["scaling"])
    external_path = get(enc, "external_path", "")

    wordvecs, synmap = if kind === :lsi
        lsi = LatentSemanticIndexing(model, docs; maxoutdim=outdim, scaling, verbose=false)
        wordvectors(lsi), synonyms(lsi, Int(syn["k"]); verbose=false)
    elseif kind === :external
        wv, oov = _load_external_embeddings(external_path, voc)
        oov > 0 && @warn "textsearch fit: $oov / $(vocsize(voc)) vocabulary tokens missing from external embeddings; using zero vectors for them"
        wv, synonyms(voc, wv, Int(syn["k"]); verbose=false)
    else
        error("unknown encoder kind: $(enc["kind"]); supported: lsi, external")
    end

    lemmas = lemma_clusters(voc, wordvecs;
        algorithm=Symbol(lem["algorithm"]), num_clusters=Int(lem["num_clusters"]), selector=Symbol(lem["selector"]))

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

    batches = _batches(each_record(format, input["path"], input["text_key"]), Int(output["batch_size"]))
    isempty(batches) && error("no documents found in $(input["path"]) (format=$(input["format"]))")

    mkpath(output["dir"])
    n = length(batches)

    for (i, docs) in enumerate(batches)
        batch_dir = joinpath(output["dir"], "_textsearch_fit_batch_$(lpad(i, 4, '0'))")
        try
            m, _ = _fit_one_batch(docs, cfg, batch_dir)
            zippath = joinpath(output["dir"], "$(output["prefix"])-$(lpad(i, 4, '0')).zip")
            zip_profile(batch_dir, zippath)
            println("saved profile $i/$n ($(length(docs)) docs, vocsize=$m) -> $zippath")
        finally
            rm(batch_dir; recursive=true, force=true)
        end
    end
end
