function parse_info_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch info",
        description="Show details for a profile: an installed nickname, or a path to a " *
                     "profile .zip/directory. Taking a path matters for the step this command " *
                     "exists for -- looking at a freshly fitted or merged profile before " *
                     "deciding whether to install it at all.")
    @add_arg_table! s begin
        "profile"
            help = "installed nickname or path to a profile .zip/directory"
            required = true
    end
    parse_args(args, s)
end

function cmd_info(args::Vector{String})
    o = parse_info_args(args)
    path = _resolve_profile_path(o["profile"])

    p = load_profile(path)
    voc = p.model.voc

    println("profile:   ", o["profile"])
    println("path:      ", path)
    println("trainsize: ", gettrainsize(voc))
    println("vocsize:   ", vocsize(voc))
    println("numtokens: ", getnumtokens(voc))
    println("avgdoclen: ", avgdoclen(voc))
    println("kind:      ", istuned(p) ? "tuned" : "base")
    println("lineage:   ", lineage_summary(p))

    # For each artifact: how much of it there is, and whether the profile APPLIES it or merely
    # carries it. That second half is the difference between a base model and a tuned one, and
    # it used to be invisible here.
    mark(n, applied) = "$n " * (applied ? "(applied)" : "(carried, not applied)")
    println("stopwords: ", mark(length(p.stopwords), p.applied.stopwords))
    println("lemmas:    ", mark(length(p.lemmas), p.applied.lemmas), " remapped tokens")
    println("query_expansion:  ", mark(length(p.query_expansion), p.applied.query_expansion), " tokens",
            p.query_expansion_distances === nothing ? ", ranking only" :
            ", with $(length(p.query_expansion_distances)) distance lists")
    # Not a stored artifact: derived from the vocabulary on demand, and reported here because a
    # profile that folds case and diacritics has nothing to bridge and it is worth seeing which
    # case this is before wondering why a query was not corrected.
    nvar = length(derive_variants(voc))
    println("variants:  ", nvar == 0 ?
            "none derivable (the profile folds what a query would fold)" :
            "$nvar folded spellings derivable from the vocabulary")
    println()
    show(stdout, gettextconfig(p))
end
