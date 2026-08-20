module TextSearchApp

using ArgParse
using TOML
using JSON3
using CSV
using Parquet2
using Tables
using TextSearch

include("home.jl")
include("corpusio.jl")
include("config.jl")
include("cli_fit.jl")
include("cli_merge.jl")
include("cli_search.jl")
include("cli_list.jl")
include("cli_info.jl")
include("cli_install.jl")
include("cli_uninstall.jl")

const SUBCOMMANDS = Dict(
    "fit" => cmd_fit,
    "merge" => cmd_merge,
    "search" => cmd_search,
    "list" => cmd_list,
    "info" => cmd_info,
    "install" => cmd_install,
    "uninstall" => cmd_uninstall,
)

function print_top_help(io::IO)
    println(io, """
    textsearch -- fit, search, and manage TextSearch.jl profiles.

    Usage: textsearch <subcommand> [options]

    Subcommands:
      fit         fit a profile (vocabulary, weights, synonyms, lemmas, stopword
                  candidates) from a corpus -- opens \$EDITOR on a TOML config
      merge       merge several profiles into one corpus-wide profile
      search      grep-like search over a collection using a profile's tokenization
                  (not fast to start -- see 'textsearch search --help')
      list        list installed profile nicknames
      info        show details for an installed profile
      install     install a profile .zip under a nickname
      uninstall   print an installed profile's path (does NOT delete the file)

    Run 'textsearch <subcommand> --help' for subcommand-specific options.
    """)
end

"""
    main(args=ARGS) -> Int

Entry point: dispatches to the requested subcommand. Returns a process exit code.
"""
function (@main)(args::Vector{String}=ARGS)
    if isempty(args) || first(args) in ("-h", "--help")
        print_top_help(stdout)
        return 0
    end

    fn = get(SUBCOMMANDS, args[1], nothing)
    if fn === nothing
        println(stderr, "unknown subcommand '$(args[1])'. Use one of: $(join(sort(collect(keys(SUBCOMMANDS))), ", "))")
        return 1
    end

    result = fn(args[2:end])
    result isa Integer ? result : 0
end

end # module
