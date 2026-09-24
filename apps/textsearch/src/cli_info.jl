function parse_info_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch info",
        description="Show details for a profile: an installed nickname, or a path to a " *
                     "profile .zip/directory. Taking a path matters for the step this command " *
                     "exists for -- looking at a freshly fitted or merged profile before " *
                     "deciding whether to install it at all. An LSI projection is shown instead " *
                     "with --lsi (the one installed for a nickname), or when the path is one.")
    @add_arg_table! s begin
        "profile"
            help = "installed nickname, or path to a profile or LSI .zip/directory"
            required = true
        "--lsi"
            help = "show the LSI projection installed for this nickname instead of the profile"
            action = :store_true
    end
    parse_args(args, s)
end

function cmd_info(args::Vector{String})
    o = parse_info_args(args)
    spec = o["profile"]
    if o["lsi"]
        path = (isfile(spec) || isdir(spec)) ? spec : lsi_path(spec)
        ispath(path) || error("no LSI projection installed for '$spec'; " *
                              "run 'textsearch list --lsi' to see installed ones")
        return _lsi_info(spec, path)
    end
    (isfile(spec) || isdir(spec)) && _is_lsi_artifact(spec) && return _lsi_info(spec, spec)
    path = _resolve_profile_path(spec)

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
    # Only an installed nickname has an installed LSI to look for; a path is a profile on its own.
    if path == profile_path(o["profile"])
        b = lsi_binding(o["profile"])
        println("lsi:       ", b === nothing ?
                "not installed (textsearch download $(o["profile"]) --lsi)" :
                b.bound ? "installed, outdim=$(b.outdim) -> $(b.path)" :
                "installed at $(b.path) but bound to profile id=$(b.profile_id), not this one " *
                "(id=$(profile_id(p))) -- load_lsi will refuse it")
    end
    println()
    show(stdout, gettextconfig(p))
end

# What an LSI artifact says about itself, and whether the profile it names is here. "Here" is
# the installed profile of the same nickname; for a loose path, the installed profile named in
# its manifest, which is what `download --lsi` and `install` would bind it to.
function _lsi_info(spec, path)
    s = lsi_summary(path)
    installed = path == lsi_path(spec)
    nick = installed ? spec : s.name
    ppath = isempty(nick) ? "" : profile_path(nick)
    bound = if isempty(ppath) || !isfile(ppath)
        "profile " * (isempty(nick) ? "(unnamed)" : "'$nick'") * " is not installed"
    else
        id = profile_id(load_profile(ppath))
        id == s.profile_id ? "yes, to the installed '$nick'" :
            "NO -- the installed '$nick' has id=$id; load_lsi will refuse it"
    end
    println("lsi:       ", spec)
    println("path:      ", path)
    println("size:      ", isfile(path) ? Base.format_bytes(filesize(path)) : "(directory)")
    println("profile:   ", isempty(s.name) ? "(unnamed)" : s.name, "  id=", s.profile_id,
            isempty(s.repo) ? "" : "  ($(s.repo) @ $(s.tag))")
    println("bound:     ", bound)
    println("outdim:    ", s.outdim, s.maxoutdim == s.outdim ? "" : " (fitted at $(s.maxoutdim))",
            " -- load_lsi(...; outdim) serves any smaller one")
    println("scaling:   ", s.scaling)
    return 0
end
