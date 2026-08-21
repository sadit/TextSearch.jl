function parse_search_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch search",
        description="Grep-like search that runs a collection through the profile's whole " *
                     "pipeline -- normalization, tokenization, lemma normalization and " *
                     "synonym expansion -- and prints every matching record as one JSONL " *
                     "line, in corpus order. This is the way to exercise a profile's " *
                     "artifacts end to end: --no-lemmas / --no-synonyms turn each one off, " *
                     "so you can see what it contributes, and the effective query tokens " *
                     "are reported on stderr. Unlike grep it is NOT fast to start (loading " *
                     "a large profile takes seconds), so prefer grep for raw byte matching.")
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
        "--no-lemmas"
            help = "do not map tokens through the profile's lemmas on either side"
            action = :store_true
        "--no-synonyms"
            help = "do not expand the query with the profile's synonym network"
            action = :store_true
        "--synonyms-k"
            help = "use at most this many synonyms per query token (0 = all the profile stored)"
            arg_type = Int
            default = 0
        "--chunk"
            help = "records per parallel batch; matches are printed in corpus order after " *
                   "each batch, so this bounds memory rather than changing results"
            arg_type = Int
            default = 4096
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

"""
    _query_tokens(p, query, tc, uselemmas, usesynonyms, synk) -> (Set{String}, report)

Builds the query's token set by running it through the same pipeline a document goes
through, plus the expansion only a query gets:

1. tokenize under the profile's `TextConfig`
2. map each token through `lemmas`, so a query written `casas` reaches documents that said
   `casa`
3. add each token's synonyms -- themselves lemma-mapped, so they can meet document tokens on
   the same footing

`report` carries the intermediate sets for the stderr summary, which is the point of this
being a probe command: it shows which artifact contributed what.
"""
function _query_tokens(p, query::AbstractString, tc, uselemmas::Bool, usesynonyms::Bool, synk::Int)
    lemma(tok) = uselemmas ? get(p.lemmas, tok, tok) : tok

    raw = collect(tokenize(tc, query))
    base = Set{String}(lemma(t) for t in raw)

    expanded = Set{String}()
    if usesynonyms
        for tok in raw
            neighbors = get(p.synonyms, tok, nothing)
            neighbors === nothing && continue
            for (i, (syn, _)) in enumerate(neighbors)
                synk > 0 && i > synk && break
                push!(expanded, lemma(syn))
            end
        end
        setdiff!(expanded, base)
    end

    union(base, expanded), (; raw, base, expanded)
end

"""
    _matches(qtokens, text, tc, lemmas, t) -> Bool

True when the document shares at least `t` tokens with the query, comparing on the same
lemma-mapped footing. Counts with an early exit instead of materializing an intersection.
"""
function _matches(qtokens::Set{String}, text::AbstractString, tc, lemmas, t::Int)
    c = 0
    for tok in tokenize(tc, text)
        tk = lemmas === nothing ? tok : get(lemmas, tok, tok)
        if tk in qtokens
            c += 1
            c >= t && return true
        end
    end
    false
end

function cmd_search(args::Vector{String})
    o = parse_search_args(args)
    o["threshold"] >= 1 || error("--threshold must be >= 1, got $(o["threshold"])")
    o["chunk"] >= 1 || error("--chunk must be >= 1, got $(o["chunk"])")

    p = load_profile(_resolve_profile_path(o["profile"]))
    tc = p.model.voc.textconfig
    uselemmas = !o["no-lemmas"]
    lemmas = uselemmas ? p.lemmas : nothing

    qtokens, rep = _query_tokens(p, o["query"], tc, uselemmas, !o["no-synonyms"], o["synonyms-k"])
    isempty(qtokens) && error("the query has no tokens under this profile's TextConfig " *
                              "(every term may have been a stopword); nothing could match")

    # what the pipeline actually did with the query -- on stderr, so stdout stays pure JSONL
    basestr = join(sort(collect(rep.base)), " ")
    expstr = join(sort(collect(rep.expanded)), " ")
    println(stderr, "query: $(length(rep.raw)) token(s) -> $basestr")
    isempty(rep.expanded) ||
        println(stderr, "  + $(length(rep.expanded)) synonym(s) -> $expstr")
    println(stderr, "  matching with threshold=$(o["threshold"]) over $(length(qtokens)) token(s), " *
                    "lemmas=$(uselemmas ? "on" : "off"), threads=$(Threads.nthreads())")

    t = o["threshold"]
    chunk = o["chunk"]
    texts = String[]
    records = Any[]
    # one slot per record in the chunk: each task writes only its own index (no shared
    # state, no locking), and printing happens afterwards in index order -- so parallelism
    # cannot reorder or interleave the output
    out = Union{Nothing,String}[]
    nhits = 0

    function flush_chunk!()
        n = length(texts)
        n == 0 && return
        resize!(out, n)
        fill!(out, nothing)
        Threads.@threads for i in 1:n
            if _matches(qtokens, texts[i], tc, lemmas, t)
                out[i] = JSON3.write(records[i])
            end
        end
        for i in 1:n
            line = out[i]
            if line !== nothing
                println(line)
                nhits += 1
            end
        end
        empty!(texts); empty!(records)
    end

    for (text, record) in each_record(Symbol(o["format"]), o["collection"], o["text-key"])
        push!(texts, text)
        push!(records, record)
        length(texts) >= chunk && flush_chunk!()
    end
    flush_chunk!()

    println(stderr, "$nhits match(es)")
    0
end
