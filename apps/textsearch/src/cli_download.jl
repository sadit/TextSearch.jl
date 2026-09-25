function parse_download_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch download", description="Download and install pre-computed linguistic profiles from GitHub releases or custom URLs.")
    @add_arg_table! s begin
        "targets"
            help = "one or more profile nicknames (e.g. en, es, eu, fr, it, pt) or direct URLs"
            nargs = '+'
            required = true
        "--tag"
            help = "GitHub release tag"
            default = PROFILES_RELEASE_TAG
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
            help = "overwrite existing installed profile (and LSI, with --lsi)"
            action = :store_true
        "--lsi"
            help = "also download the profile's LSI projection (<nickname>-lsi.zip in the " *
                   "release) into ~/.textsearch/lsi/, and check it is bound to the profile"
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
            # not `continue` under --lsi: an installed profile is exactly what an LSI is added to
            println(stderr, "profile '$nickname' already exists at $dest (pass --force to overwrite)")
        else
            download_url = is_url ? target : TextSearch._release_url("$nickname.zip", repo, tag, custom_url)
            println("downloading '$nickname' from $download_url ...")
            try
                download_profile(download_url; dest, force=true)
                println("installed '$nickname' -> $dest")
            catch e
                println(stderr, "failed to download profile '$nickname' from $download_url: $e")
                return 1
            end
        end

        if o["lsi"]
            # the release names the LSI after the profile it belongs to, which is `target` even
            # when --as installs that profile under another nickname
            is_url && (println(stderr, "--lsi needs a profile nickname, not a URL: $target"); return 1)
            _download_lsi_for(nickname, target, repo, tag, custom_url, force) || return 1
        end
    end
    return 0
end

# Downloads `<remote>-lsi.zip` as the LSI of installed profile `nickname`, and removes it again
# if it does not name that profile: an LSI bound to another profile answers wrongly rather than
# failing, so an unbound one is not left installed where `info` and a consumer would find it.
function _download_lsi_for(nickname, remote, repo, tag, custom_url, force)
    dest = lsi_path(nickname)
    if isfile(dest) && !force
        println(stderr, "LSI for '$nickname' already exists at $dest (pass --force to overwrite)")
        return true
    end
    lsi_url = TextSearch._release_url("$remote-lsi.zip", repo, tag, custom_url)
    println("downloading LSI for '$nickname' from $lsi_url ...")
    try
        download_lsi(lsi_url; dest, force=true)
    catch e
        println(stderr, "failed to download the LSI for '$nickname' from $lsi_url: $e")
        return false
    end
    b = lsi_binding(nickname)
    if !b.bound
        rm(dest; force=true)
        println(stderr, "the LSI at $lsi_url is bound to profile id=$(b.profile_id), not to the " *
                        "installed '$nickname'; removed it. Download the matching profile " *
                        "(same --tag) with --force.")
        return false
    end
    println("installed LSI for '$nickname' (outdim=$(b.outdim)) -> $dest")
    true
end

