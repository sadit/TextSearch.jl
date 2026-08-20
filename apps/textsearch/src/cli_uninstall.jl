function parse_uninstall_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch uninstall",
        description="Print an installed profile's file path -- does NOT delete the file. " *
                     "textsearch never silently destroys a profile zip you may have spent " *
                     "significant compute producing; remove it yourself if you're sure.")
    @add_arg_table! s begin
        "nickname"
            help = "installed profile nickname (see 'textsearch list')"
            required = true
    end
    parse_args(args, s)
end

"""
    cmd_uninstall(args)

NOTE: despite the name, this does NOT delete anything -- it only prints the installed
profile's path so you can review/back up/delete it yourself.
"""
function cmd_uninstall(args::Vector{String})
    o = parse_uninstall_args(args)
    path = profile_path(o["nickname"])
    isfile(path) || error("no installed profile named '$(o["nickname"])'; run 'textsearch list' to see installed profiles")
    println("'$(o["nickname"])' is installed at:")
    println(path)
    println("textsearch does not delete profile files automatically -- remove it yourself if you're sure, e.g.:")
    println("  rm '$path'")
end
