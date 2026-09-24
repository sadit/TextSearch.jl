function parse_list_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch list",
        description="List installed profiles and LSI projections, or those of a remote release. " *
                    "The two are listed in separate sections: an LSI is not a profile, it is " *
                    "bound to one.")
    @add_arg_table! s begin
        "--remote", "-r"
            help = "list what a GitHub release (or --url) offers instead of what is installed"
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
        "--profiles"
            help = "list profiles only"
            action = :store_true
        "--lsi"
            help = "list LSI projections only"
            action = :store_true
    end
    parse_args(args, s)
end

# Which sections to print: both unless one of --profiles/--lsi narrows it (both flags = both).
_list_sections(o) = (profiles = o["profiles"] || !o["lsi"], lsi = o["lsi"] || !o["profiles"])

function _print_section(title, lines)
    println(title)
    isempty(lines) ? println("  (none)") : foreach(l -> println("  ", l), lines)
end

function cmd_list(args::Vector{String})
    o = parse_list_args(args)
    sel = _list_sections(o)

    if !o["remote"] && o["url"] === nothing
        first_section = true
        if sel.profiles
            _print_section("profiles ($(profiles_dir())):", list_nicknames())
            first_section = false
        end
        if sel.lsi
            first_section || println()
            lines = map(list_lsi_nicknames()) do nick
                b = lsi_binding(nick)
                string(nick, "  outdim=", b.outdim, b.bound ? "" :
                       isfile(profile_path(nick)) ? "  (NOT bound to the installed '$nick' profile)" :
                       "  (profile '$nick' is not installed)")
            end
            _print_section("LSI projections ($(lsi_dir())):", lines)
        end
        return 0
    end

    tag = o["tag"]
    repo = o["repo"]
    url = o["url"]
    source_desc = url !== nothing ? url : "$repo @ $tag"
    fmt(r) = "$(r.name) ($(round(r.size / (1024^2); digits=1)) MB) -> $(r.url)"

    sections = Tuple{String,Function}[]
    sel.profiles && push!(sections, ("Remote profiles", list_remote_profiles))
    sel.lsi && push!(sections, ("Remote LSI projections", list_remote_lsi))
    for (i, (title, lister)) in enumerate(sections)
        items = try
            lister(; repo, tag, url)
        catch e
            println(stderr, "error fetching remote profiles: $e")
            return 1
        end
        i > 1 && println()
        _print_section("$title ($source_desc):", fmt.(items))
    end
    return 0
end
