function parse_uninstall_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch uninstall",
        description="Remove an installed profile. Without --force nothing is deleted: the " *
                     "path and size are printed along with the command that would delete it. " *
                     "A profile zip can represent hours of compute, and the copy under " *
                     "~/.textsearch may be the only one left, so the destructive reading of " *
                     "the word 'uninstall' has to be asked for.")
    @add_arg_table! s begin
        "nickname"
            help = "installed profile nickname (see 'textsearch list')"
            required = true
        "--force"
            help = "actually delete the installed file"
            action = :store_true
    end
    parse_args(args, s)
end

"""
    cmd_uninstall(args)

Deletes the installed copy when `--force` is given, and otherwise only reports what would be
deleted. The default is the reporting one because the name is ambiguous in a way that matters:
"uninstall" suggests removing a registration, while here the registration *is* the file.
"""
function cmd_uninstall(args::Vector{String})
    o = parse_uninstall_args(args)
    nick = o["nickname"]
    # the profile and, if installed, its LSI: removing only the profile would leave a projection
    # that nothing can load, since it is bound to exactly that profile
    paths = filter(isfile, [profile_path(nick), lsi_path(nick)])
    isempty(paths) && error("no installed profile named '$nick'; run 'textsearch list' to see installed profiles")

    if o["force"]
        for path in paths
            size = Base.format_bytes(filesize(path))
            rm(path)
            println("deleted ", path == lsi_path(nick) ? "LSI for '$nick'" : "'$nick'", " ($size) from $path")
        end
    else
        println("'$nick' is installed at:")
        for path in paths
            println("  $path  ($(Base.format_bytes(filesize(path))))")
        end
        println("nothing was deleted -- pass --force to remove ", length(paths) == 1 ? "the file:" : "them:")
        println("  textsearch uninstall $nick --force")
    end
end
