function parse_search_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch search",
        description="Grep-like search: prints every record of a collection whose text " *
                     "field shares at least --threshold tokens (after the profile's " *
                     "normalization/tokenization) with the query, one JSONL line per hit. " *
                     "Unlike grep, this is NOT fast to start -- loading the profile and " *
                     "opening the collection has real cost. Prefer this over grep only " *
                     "when you need corpus-consistent tokenization/normalization, not raw " *
                     "byte-level matching.")
    @add_arg_table! s begin
        "profile"
            help = "installed nickname or path to a profile .zip/directory"
            required = true
        "query"
            help = "query text"
            required = true
        "--collection"
            help = "path to the collection to search"
            required = true
        "--format"
            help = "collection format: plaintext | csv | jsonl | json | parquet"
            default = "jsonl"
        "--text-key"
            help = "column/JSON-key holding the document text"
            default = "text"
        "-t", "--threshold"
            help = "minimum token-set intersection size for a match (t=1: any shared token; " *
                   "raise toward the query's token count for stricter, AND-like matching)"
            arg_type = Int
            default = 1
    end
    parse_args(args, s)
end

"""
    _resolve_profile_path(spec::AbstractString) -> String

`spec` is used as a literal path if it exists (file or directory); otherwise it's looked
up as an installed nickname under `~/.textsearch/profiles/`.
"""
function _resolve_profile_path(spec::AbstractString)
    (isfile(spec) || isdir(spec)) && return spec
    path = profile_path(spec)
    isfile(path) || error("no profile at '$spec' and no installed profile named '$spec'; " *
                           "run 'textsearch list' to see installed profiles")
    path
end

function cmd_search(args::Vector{String})
    o = parse_search_args(args)
    o["threshold"] >= 1 || error("--threshold must be >= 1, got $(o["threshold"])")

    p = load_profile(_resolve_profile_path(o["profile"]))
    tc = p.model.voc.textconfig
    qtokens = Set(collect(tokenize(tc, o["query"])))
    t = o["threshold"]

    for (text, record) in each_record(Symbol(o["format"]), o["collection"], o["text-key"])
        rtokens = Set(collect(tokenize(tc, text)))
        length(intersect(qtokens, rtokens)) >= t && println(JSON3.write(record))
    end
end
