function parse_uninstall_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch uninstall",
        description="Remove an installed profile, and its LSI projection with it. Without " *
                     "--force nothing is deleted: the paths and sizes are printed along with the " *
                     "command that would delete them. A profile zip can represent hours of " *
                     "compute, and the copy under ~/.textsearch may be the only one left, so the " *
                     "destructive reading of the word 'uninstall' has to be asked for.")
    @add_arg_table! s begin
        "nickname"
            help = "installed profile nickname (see 'textsearch list')"
            required = true
        "--lsi"
            help = "remove only the LSI projection, keeping the profile"
            action = :store_true
        "--force"
            help = "actually delete the installed file(s)"
            action = :store_true
    end
    parse_args(args, s)
end

"""
    cmd_uninstall(args)

Deletes the installed copy when `--force` is given, and otherwise only reports what would be
deleted. The default is the reporting one because the name is ambiguous in a way that matters:
"uninstall" suggests removing a registration, while here the registration *is* the file.

Without `--lsi` the profile's LSI projection goes with it: it is bound to exactly that profile,
so left behind it is a file nothing can load. With `--lsi` only the projection is removed.
"""
function cmd_uninstall(args::Vector{String})
    o = parse_uninstall_args(args)
    nick = o["nickname"]
    only_lsi = o["lsi"]
    candidates = only_lsi ? [lsi_path(nick)] : [profile_path(nick), lsi_path(nick)]
    paths = filter(isfile, candidates)
    isempty(paths) && error(only_lsi ?
        "no LSI projection installed for '$nick'; run 'textsearch list --lsi' to see installed ones" :
        "no installed profile named '$nick'; run 'textsearch list' to see installed profiles")
    label(path) = path == lsi_path(nick) ? "LSI for '$nick'" : "'$nick'"

    if o["force"]
        for path in paths
            size = Base.format_bytes(filesize(path))
            rm(path)
            println("deleted ", label(path), " ($size) from $path")
        end
    else
        println(only_lsi ? "the LSI for '$nick' is installed at:" : "'$nick' is installed at:")
        for path in paths
            println("  $path  ($(Base.format_bytes(filesize(path))))")
        end
        println("nothing was deleted -- pass --force to remove ", length(paths) == 1 ? "the file:" : "them:")
        println("  textsearch uninstall $nick", only_lsi ? " --lsi" : "", " --force")
    end
end
