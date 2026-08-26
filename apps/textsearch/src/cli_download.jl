function parse_download_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch download", description="Download and install pre-computed linguistic profiles from GitHub releases.")
    @add_arg_table! s begin
        "nicknames"
            help = "one or more profile nicknames to download (e.g. en, es, eu, fr, it, pt)"
            nargs = '+'
            required = true
        "--tag"
            help = "GitHub release tag"
            default = "v1.1.0"
        "--repo"
            help = "GitHub repository (owner/repo)"
            default = "sadit/TextSearch.jl"
        "--force"
            help = "overwrite existing installed profile"
            action = :store_true
    end
    parse_args(args, s)
end

function cmd_download(args::Vector{String})
    o = parse_download_args(args)
    tag = o["tag"]
    repo = o["repo"]
    force = o["force"]
    for nickname in o["nicknames"]
        dest = profile_path(nickname)
        if isfile(dest) && !force
            println(stderr, "profile '$nickname' already exists at $dest (pass --force to overwrite)")
            continue
        end
        url = "https://github.com/$repo/releases/download/$tag/$nickname.zip"
        println("downloading '$nickname' from $url ...")
        tmppath = tempname() * ".zip"
        try
            Downloads.download(url, tmppath)
            mv(tmppath, dest; force=true)
            println("installed '$nickname' -> $dest")
        catch e
            rm(tmppath; force=true)
            println(stderr, "failed to download profile '$nickname' from $url: $e")
            return 1
        end
    end
    return 0
end
