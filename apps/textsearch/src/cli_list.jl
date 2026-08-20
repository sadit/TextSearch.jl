function parse_list_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch list", description="List installed profile nicknames.")
    parse_args(args, s)
end

function cmd_list(args::Vector{String})
    parse_list_args(args)
    foreach(println, list_nicknames())
end
