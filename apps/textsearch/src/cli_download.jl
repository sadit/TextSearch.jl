function parse_download_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch download", description="Download and install pre-computed linguistic profiles from GitHub releases or custom URLs.")
    @add_arg_table! s begin
        "targets"
            help = "one or more profile nicknames (e.g. en, es, eu, fr, it, pt) or direct URLs"
            nargs = '+'
            required = true
        "--tag"
            help = "GitHub release tag"
            default = "v1.1.0"
        "--repo"
            help = "GitHub repository (owner/repo)"
            default = "sadit/TextSearch.jl"
        "--url"
            help = "custom base download URL or direct archive URL"
            default = nothing
        "--as"
            help = "custom nickname to install under (only valid with a single download target)"
            default = nothing
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
    targets = o["targets"]
    custom_as = o["as"]
    custom_url = o["url"]

    if custom_as !== nothing && length(targets) > 1
        println(stderr, "--as can only be used when downloading a single profile target")
        return 1
    end

    for target in targets
        is_url = startswith(target, "http://") || startswith(target, "https://")
        nickname = if custom_as !== nothing
            custom_as
        elseif is_url
            default_nickname(target)
        else
            target
        end

        dest = profile_path(nickname)
        if isfile(dest) && !force
            println(stderr, "profile '$nickname' already exists at $dest (pass --force to overwrite)")
            continue
        end

        download_url = if is_url
            target
        elseif custom_url !== nothing
            endswith(custom_url, ".zip") ? custom_url : joinpath(custom_url, "$nickname.zip")
        else
            "https://github.com/$repo/releases/download/$tag/$nickname.zip"
        end

        println("downloading '$nickname' from $download_url ...")
        tmppath = tempname() * ".zip"
        try
            Downloads.download(download_url, tmppath; headers=["User-Agent" => "TextSearch.jl"])
            mv(tmppath, dest; force=true)
            println("installed '$nickname' -> $dest")
        catch e
            rm(tmppath; force=true)
            println(stderr, "failed to download profile '$nickname' from $download_url: $e")
            return 1
        end
    end
    return 0
end

