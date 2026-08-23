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
    path = profile_path(o["nickname"])
    isfile(path) || error("no installed profile named '$(o["nickname"])'; run 'textsearch list' to see installed profiles")
    size = Base.format_bytes(filesize(path))

    if o["force"]
        rm(path)
        println("deleted '$(o["nickname"])' ($size) from $path")
    else
        println("'$(o["nickname"])' is installed at:")
        println("  $path  ($size)")
        println("nothing was deleted -- pass --force to remove the file:")
        println("  textsearch uninstall $(o["nickname"]) --force")
    end
end
