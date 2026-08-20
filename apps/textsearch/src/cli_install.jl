function parse_install_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch install", description="Install a profile .zip under a nickname.")
    @add_arg_table! s begin
        "path"
            help = "path to a profile .zip (e.g. produced by 'textsearch fit')"
            required = true
        "nickname"
            help = "nickname to install under (default: the zip's filename without extension)"
            required = false
        "--force"
            help = "overwrite an existing profile installed under the same nickname"
            action = :store_true
    end
    parse_args(args, s)
end

function cmd_install(args::Vector{String})
    o = parse_install_args(args)
    isfile(o["path"]) || error("no such file: $(o["path"])")
    nickname = o["nickname"] === nothing ? default_nickname(o["path"]) : o["nickname"]
    dest = profile_path(nickname)
    (isfile(dest) && !o["force"]) &&
        error("a profile named '$nickname' already exists at $dest; pass --force to overwrite, or choose a different nickname")
    cp(o["path"], dest; force=true)
    println("installed '$nickname' -> $dest")
end
