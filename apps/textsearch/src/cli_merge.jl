function parse_merge_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch merge",
        description="Merge multiple profiles into one. NOT YET IMPLEMENTED -- the CLI " *
                     "surface below is stable so scripts can be written against it, but " *
                     "combining weights/synonyms/lemmas across profiles needs more design.")
    @add_arg_table! s begin
        "profiles"
            help = "installed nicknames or paths of the profiles to merge (2 or more)"
            nargs = '+'
            required = true
        "--out"
            help = "output profile path"
            required = true
    end
    parse_args(args, s)
end

"""
    cmd_merge(args) -> Int

STUB: not yet implemented. Merging vocabulary/weights/synonyms/lemmas across profiles is
deferred to a future design session -- for now, re-run 'textsearch fit' with
batch_size=0 over the combined corpus instead. Prints a message to stderr and returns 1.
"""
function cmd_merge(args::Vector{String})
    parse_merge_args(args)
    println(stderr, "textsearch merge: not yet implemented. Merging weights/synonyms/lemmas " *
                     "across profiles needs more design -- for now, re-run 'textsearch fit' " *
                     "with batch_size=0 over the combined corpus instead.")
    1
end
