function parse_list_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch list", description="List locally installed profile nicknames or query remote releases.")
    @add_arg_table! s begin
        "--remote", "-r"
            help = "list remote profiles available in GitHub releases or custom URL"
            action = :store_true
        "--tag"
            help = "GitHub release tag when listing remote profiles"
            default = PROFILES_RELEASE_TAG
        "--repo"
            help = "GitHub repository (owner/repo) when listing remote profiles"
            default = "sadit/TextSearch.jl"
        "--url"
            help = "custom URL or API endpoint for remote profiles"
            default = nothing
        "--lsi"
            help = "list LSI projections instead of profiles: installed ones, or with --remote " *
                   "the <nickname>-lsi.zip assets of the release"
            action = :store_true
    end
    parse_args(args, s)
end

function cmd_list(args::Vector{String})
    o = parse_list_args(args)
    lsi = o["lsi"]
    if !o["remote"] && o["url"] === nothing
        if lsi
            for nick in list_lsi_nicknames()
                b = lsi_binding(nick)
                println(nick, "  outdim=", b.outdim, b.bound ? "" :
                        "  (NOT bound to the installed '$nick' profile)")
            end
        else
            foreach(println, list_nicknames())
        end
        return 0
    end

    tag = o["tag"]
    repo = o["repo"]
    url = o["url"]
    remote_items = try
        lsi ? list_remote_lsi(; repo, tag, url) : list_remote_profiles(; repo, tag, url)
    catch e
        println(stderr, "error fetching remote profiles: $e")
        return 1
    end

    what = lsi ? "LSI projections" : "profiles"
    if isempty(remote_items)
        println("no remote $what found.")
        return 0
    end

    source_desc = url !== nothing ? url : "$repo @ $tag"
    println("Remote $what ($source_desc):")
    for r in remote_items
        sz_mb = round(r.size / (1024^2); digits=1)
        println("  $(r.name) ($(sz_mb) MB) -> $(r.url)")
    end
    return 0
end
