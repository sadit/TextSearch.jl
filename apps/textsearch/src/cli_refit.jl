function parse_refit_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch refit",
        description="Adapt a bootstrap profile to a dataset, given a sample of it, and write " *
                     "a new self-contained profile. Statistics are ADJUSTED, not replaced: " *
                     "the base acts as a prior worth --kappa documents against the sample's " *
                     "evidence, so a word the base considered important but the sample never " *
                     "shows survives with reduced weight, while one that mattered in neither " *
                     "is dropped. No embedding is fit here -- lemmas and synonyms come from " *
                     "the base -- which is what makes a refit cheap next to a fit. The result " *
                     "is typically smaller and more accurate for the dataset than the generic " *
                     "profile it came from.")
    @add_arg_table! s begin
        "profile"
            help = "base profile: installed nickname, or path to a .zip/directory"
            required = true
        "--sample"
            help = "path to a sample of the target dataset"
            required = true
        "--out"
            help = "output profile path (.zip)"
            required = true
        "--format"
            help = "sample format: plaintext | csv | jsonl | json | parquet"
            default = "jsonl"
        "--text-key"
            help = "column/JSON-key holding the document text"
            default = "text"
        "--kappa"
            help = "the base's authority, in documents (0 = as many as the sample has, " *
                   "weighting the two sides equally). Mutually exclusive with --base-weight."
            arg_type = Float64
            default = 0.0
        "--base-weight"
            help = "the base's share of the blend as a fraction in (0,1), converted to a " *
                   "kappa relative to the sample size; 0.5 matches the --kappa default"
            arg_type = Float64
            default = 0.0
        "--no-lemmas"
            help = "do not apply the base's lemma map; it is still carried as an artifact. " *
                   "Whether to lemmatize is the refit's decision, which is why a base " *
                   "profile normally leaves the map unapplied."
            action = :store_true
        "--keep-rate"
            help = "a token absent from the sample is kept only if its base document-frequency " *
                   "rate is at least this"
            arg_type = Float64
            default = 1e-5
        "--keep-floor"
            help = "...and only if it was seen in at least this many base documents, so a " *
                   "single-document typo cannot clear a small rate threshold"
            arg_type = Int
            default = 3
        "--doc-freq-threshold"
            help = "document-frequency ratio above which a token is reported as a stopword " *
                   "candidate (the APPLIED stopword set stays the base's)"
            arg_type = Float64
            default = 0.5
        "--drop-distances"
            help = "omit the synonym network's distances from the output, for the smallest " *
                   "possible profile; only the ranking is used on the normal query path"
            action = :store_true
        "--chunk"
            help = "documents per batch while streaming the sample into a vocabulary; bounds " *
                   "memory and does not affect the result"
            arg_type = Int
            default = 10000
    end
    parse_args(args, s)
end

"""
    _stream_sample_vocabulary(textconfig, format, path, text_key, chunk) -> Vocabulary

Builds a `Vocabulary` from a sample by folding it in batches of `chunk` documents, never
holding more than one batch.

This is why `refit_profile`'s core takes a `Vocabulary` rather than a corpus: a sample can be
far larger than memory, and the counters are all a refit needs. `trainsize`/`numtokens` are
accumulated as batches arrive, since `update_voc!` only merges the per-token counts.
"""
function _stream_sample_vocabulary(textconfig, format::Symbol, path::AbstractString,
                                    text_key::AbstractString, chunk::Int)
    acc = Vocabulary(textconfig, Int64(0), Int64(0))
    ndocs_total = Int64(0)
    ntokens_total = Int64(0)

    _each_batch(each_record(format, path, text_key), chunk) do _, docs
        v = Vocabulary(textconfig, docs; verbose=false)
        update_voc!(acc, v)
        ndocs_total += trainsize(v)
        ntokens_total += numtokens(v)
    end

    acc.trainsize[] = ndocs_total
    acc.numtokens[] = ntokens_total
    acc
end

function cmd_refit(args::Vector{String})
    o = parse_refit_args(args)
    out = o["out"]
    endswith(out, ".zip") || error("--out must end in .zip, got '$out'")
    o["chunk"] >= 1 || error("--chunk must be >= 1, got $(o["chunk"])")
    o["keep-floor"] >= 0 || error("--keep-floor must be >= 0, got $(o["keep-floor"])")

    kappa = o["kappa"]
    bw = o["base-weight"]
    if bw != 0.0
        kappa == 0.0 || error("pass either --kappa or --base-weight, not both")
        0.0 < bw < 1.0 || error("--base-weight must be in (0,1), got $bw")
    end

    base = load_profile(_resolve_profile_path(o["profile"]))
    apply_lemmas = !o["no-lemmas"]

    println("base: vocsize=$(vocsize(base.model.voc)) trainsize=$(trainsize(base.model.voc)) " *
            "lemmas=$(length(base.lemmas)) synonyms=$(length(base.synonyms))")
    flush(stdout)

    # The sample must be tokenized under exactly the config the refit runs under, or the two
    # sides' counters do not correspond; refit_profile re-checks this and errors if not.
    textconfig = refit_textconfig(base; apply_lemmas)
    sample_voc = _stream_sample_vocabulary(textconfig, Symbol(o["format"]), o["sample"],
                                           o["text-key"], o["chunk"])
    trainsize(sample_voc) > 0 || error("the sample at '$(o["sample"])' yielded no documents")

    # --base-weight is expressed relative to the sample, so it needs the sample's size first
    bw != 0.0 && (kappa = trainsize(sample_voc) * bw / (1 - bw))

    r = refit_profile(base, sample_voc;
                      kappa, apply_lemmas,
                      keep_rate=o["keep-rate"], keep_floor=o["keep-floor"],
                      doc_freq_threshold=o["doc-freq-threshold"],
                      verbose=true)

    syndists = o["drop-distances"] ? nothing : r.synonym_distances

    mkpath(dirname(abspath(out)))
    tmpdir = out * ".tmpdir"
    try
        save_profile(tmpdir, r.model;
                     r.synonyms, synonym_distances=syndists, r.lemmas,
                     r.stopword_candidates, r.encoder)
        zip_profile(tmpdir, out)
    finally
        rm(tmpdir; recursive=true, force=true)
    end

    println("refit -> $out")
    println("  vocsize=$(vocsize(r.model.voc))  trainsize=$(trainsize(r.model.voc))  " *
            "numtokens=$(numtokens(r.model.voc))")
    println("  synonyms=$(length(r.synonyms)) tokens  " *
            "distances=$(syndists === nothing ? "dropped" : "$(length(syndists)) tokens")  " *
            "lemmas=$(length(r.lemmas)) remapped " *
            "($(r.encoder.lemmas_applied ? "applied" : "carried only"))  " *
            "stopword_candidates=$(length(r.stopword_candidates))")
    0
end
