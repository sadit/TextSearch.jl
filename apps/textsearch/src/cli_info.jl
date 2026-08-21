function parse_info_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch info", description="Show details for an installed profile.")
    @add_arg_table! s begin
        "nickname"
            help = "installed profile nickname (see 'textsearch list')"
            required = true
    end
    parse_args(args, s)
end

function cmd_info(args::Vector{String})
    o = parse_info_args(args)
    path = profile_path(o["nickname"])
    isfile(path) || error("no installed profile named '$(o["nickname"])'; run 'textsearch list' to see installed profiles")

    p = load_profile(path)
    voc = p.model.voc

    println("nickname:  ", o["nickname"])
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
    println("synonyms:  ", mark(length(p.synonyms), p.applied.synonyms), " tokens",
            p.synonym_distances === nothing ? ", ranking only" :
            ", with $(length(p.synonym_distances)) distance lists")
    println()
    show(stdout, gettextconfig(p))
end
