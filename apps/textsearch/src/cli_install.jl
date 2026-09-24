function parse_install_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch install",
        description="Install a profile .zip, or an LSI projection .zip, under a nickname. An LSI " *
                    "is recognized by its manifest and goes to ~/.textsearch/lsi/, bound to the " *
                    "installed profile of the same nickname.")
    @add_arg_table! s begin
        "path"
            help = "path to a profile .zip (e.g. produced by 'textsearch fit') or an LSI .zip " *
                   "(e.g. es-lsi.zip from a release, or one written by save_lsi)"
            required = true
        "nickname"
            help = "nickname to install under (default: the zip's filename without extension, " *
                   "and without -lsi for an LSI)"
            required = false
        "--force"
            help = "overwrite an existing profile (or LSI) installed under the same nickname"
            action = :store_true
    end
    parse_args(args, s)
end

function cmd_install(args::Vector{String})
    o = parse_install_args(args)
    path = o["path"]
    isfile(path) || error("no such file: $path")
    _is_lsi_artifact(path) && return _install_lsi(path, o["nickname"], o["force"])

    nickname = o["nickname"] === nothing ? default_nickname(path) : o["nickname"]
    dest = profile_path(nickname)
    (isfile(dest) && !o["force"]) &&
        error("a profile named '$nickname' already exists at $dest; pass --force to overwrite, or choose a different nickname")
    cp(path, dest; force=true)
    println("installed '$nickname' -> $dest")
    # replacing a profile can orphan the LSI installed for its nickname; say so rather than
    # leave a projection that `load_lsi` will refuse
    b = lsi_binding(nickname)
    b === nothing || b.bound ||
        println(stderr, "note: the LSI installed for '$nickname' is bound to profile id=$(b.profile_id), " *
                        "not this one; replace it, or remove it with 'textsearch uninstall $nickname --lsi'")
    return 0
end

# An LSI goes in only against the profile it names, when that profile is installed: bound to
# another it would answer wrongly rather than fail. With no profile installed yet it is accepted,
# and the note says which profile it waits for.
function _install_lsi(path, nickname, force)
    nickname = nickname === nothing ? default_lsi_nickname(path) : nickname
    dest = lsi_path(nickname)
    (isfile(dest) && !force) &&
        error("an LSI for '$nickname' already exists at $dest; pass --force to overwrite, or choose a different nickname")
    s = lsi_summary(path)
    ppath = profile_path(nickname)
    if isfile(ppath)
        id = profile_id(load_profile(ppath))
        id == s.profile_id ||
            error("this LSI is bound to profile id=$(s.profile_id)$(isempty(s.name) ? "" : " ('$(s.name)')"), " *
                  "and the installed '$nickname' has id=$id; install it under the nickname of " *
                  "the profile it was fitted against")
    end
    cp(path, dest; force=true)
    println("installed LSI for '$nickname' (outdim=$(s.outdim)) -> $dest")
    isfile(ppath) ||
        println(stderr, "note: no profile named '$nickname' is installed; this LSI needs profile " *
                        "id=$(s.profile_id)", isempty(s.repo) ? "" : " (from $(s.repo) @ $(s.tag))")
    return 0
end
