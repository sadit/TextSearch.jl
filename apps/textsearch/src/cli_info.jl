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
    println("trainsize: ", trainsize(voc))
    println("vocsize:   ", vocsize(voc))
    println("numtokens: ", numtokens(voc))
    println("avgdoclen: ", avgdoclen(voc))
    println("synonyms:  ", length(p.synonyms), " tokens")
    println("lemmas:    ", length(p.lemmas), " remapped tokens")
    println("stopword_candidates: ", length(p.stopword_candidates), " tokens")
    if p.encoder !== nothing
        println("encoder:   ", p.encoder["kind"], " (", join(["$k=$v" for (k, v) in p.encoder if k != "kind"], ", "), ")")
    else
        println("encoder:   (none saved)")
    end
    println()
    show(stdout, voc.textconfig)
end
