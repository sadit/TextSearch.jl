function parse_search_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch search",
        description="Grep-like search that runs a collection through the profile's whole " *
                     "pipeline -- normalization, tokenization, lemma normalization and " *
                     "query expansion -- and prints every matching record as one JSONL " *
                     "line, in corpus order. This is the way to exercise a profile's " *
                     "artifacts end to end: --no-lemmas / --no-query_expansion turn each one off, " *
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
            help = "drop the profile's lemma step from the tokenization pipeline, so tokens " *
                   "are matched in their surface forms on both sides (no effect on a profile " *
                   "fitted without an applied lemma map)"
            action = :store_true
        "--no-query_expansion"
            help = "do not expand the query with the profile's query_expansion network"
            action = :store_true
        "--correction"
            help = "orthographic correction of the query: auto (default; correct a token only " *
                   "where the evidence says the typed spelling is wrong -- absent from the " *
                   "vocabulary, or far rarer than another spelling of the same word) | off " *
                   "(search exactly what was typed: this is the \"search instead for ...\" " *
                   "escape, and it is what the report points you to) | always (bridge every " *
                   "token, the only way to reach an accented alternative of a spelling that is " *
                   "itself common)"
            default = "auto"
        "--correction-ratio"
            help = "how much rarer than another spelling of the same word a typed spelling has " *
                   "to be before --correction=auto treats it as wrong (1 = anything but the " *
                   "commonest, Inf = never)"
            arg_type = Float64
            default = 50.0
        "--query_expansion-k"
            help = "use at most this many query_expansion per query token (0 = all the profile stored)"
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
    _query_report(p, query, tc, policy::QueryPolicy, variants) -> (Set{String}, report)

Runs the library's [`query_tokens`](@ref) and keeps the pieces this command reports on.

The pipeline itself is not here and must not be: it is the same one the inverted files use, so
`textsearch search` cannot drift from what an index would have done with the same query. What is
left is presentation -- which set came from where, for the stderr summary that is the point of
this being a probe command.
"""
function _query_report(p, query::AbstractString, tc, policy::QueryPolicy, variants)
    raw = collect(tokenize(tc, query))
    rq = query_tokens(p.model.voc, raw,
                      QueryPipeline(; policy, variants,
                                    expansion = policy.expansion ? p.query_expansion : nothing,
                                    distances = policy.expansion ? p.query_expansion_distances : nothing))
    base = Set{String}(t.token for t in rq.terms if t.reason !== :expansion)
    expanded = setdiff(Set{String}(t.token for t in rq.terms if t.reason === :expansion), base)
    (union(base, expanded),
     (; raw, base, expanded, bridged=setdiff(base, Set(raw)), resolution=rq.resolution))
end

"""
    _matches(qtokens, text, tc, buff, t) -> Bool

True when the document shares at least `t` tokens with the query. Counts with an early exit
instead of materializing an intersection.

`buff` is a borrowed `TokenizerBuffer` reused across the documents one task handles, so
matching a document allocates nothing per document -- the tokenizer writes into the buffer
instead of returning a fresh token vector each time. Each task must own its buffer; passing
one shared across tasks would corrupt it.
"""
function _matches(qtokens::Set{String}, text::AbstractString, tc, buff, t::Int)
    empty!(buff)
    c = 0
    for tok in tokenize(borrowtokenizedtext, tc, text, buff).tokens
        if tok in qtokens
            c += 1
            c >= t && return true
        end
    end
    false
end

"""
    _partition(n::Int, k::Int) -> Vector{UnitRange{Int}}

Splits `1:n` into at most `k` contiguous, non-overlapping ranges covering it exactly, the
first `n % k` of them one element longer. Contiguity is what keeps a task's writes confined
to its own slice of the output, and covering `1:n` exactly is what guarantees no record is
matched twice or skipped.
"""
function _partition(n::Int, k::Int)
    k = min(k, n)
    k <= 1 && return [1:n]
    q, r = divrem(n, k)
    ranges = Vector{UnitRange{Int}}(undef, k)
    start = 1
    for j in 1:k
        len = q + (j <= r ? 1 : 0)
        ranges[j] = start:(start + len - 1)
        start += len
    end
    ranges
end

function cmd_search(args::Vector{String})
    o = parse_search_args(args)
    o["threshold"] >= 1 || error("--threshold must be >= 1, got $(o["threshold"])")
    o["chunk"] >= 1 || error("--chunk must be >= 1, got $(o["chunk"])")

    base = load_profile(_resolve_profile_path(o["profile"]))
    # `--no-lemmas` means "do not apply the lemma map", which is a marker on the profile, not
    # surgery on a pipeline: flipping it re-materializes the TextConfig from the artifacts the
    # profile still carries. A profile that never applied them is unaffected.
    p = o["no-lemmas"] ? with_applied(base; lemmas=false) : base
    tc = gettextconfig(p)
    lemmas_on = p.applied.lemmas

    correction = Symbol(o["correction"])
    correction in (:off, :auto, :always) ||
        error("unknown --correction: $(o["correction"]); supported: off, auto, always")
    o["correction-ratio"] >= 1 ||
        error("--correction-ratio must be >= 1, got $(o["correction-ratio"])")
    # every query-time choice in one value, so the query arrives marked with how to treat it
    policy = QueryPolicy(; correction, expansion=!o["no-query_expansion"],
                           expansion_k=o["query_expansion-k"],
                           negligible_ratio=o["correction-ratio"])
    # Derived here rather than read from the profile: the map is a pure function of the
    # vocabulary, so storing it would be a second copy of something already in the file -- and a
    # stale one, since a merge cannot union per-part maps correctly. 0.24s over the 479,245-token
    # Portuguese Wikipedia vocabulary, once per invocation.
    variants = policy.correction === :off ? nothing : derive_variants(p.model.voc)
    qtokens, rep = _query_report(p, o["query"], tc, policy, variants)
    isempty(qtokens) && error("the query has no tokens under this profile's TextConfig " *
                              "(every term may have been a stopword); nothing could match")

    # what the pipeline actually did with the query -- on stderr, so stdout stays pure JSONL
    basestr = join(sort(collect(rep.base)), " ")
    expstr = join(sort(collect(rep.expanded)), " ")
    println(stderr, "query: $(length(rep.raw)) token(s) -> $basestr")
    # Correcting a query silently is not an option: say what was searched, and where a typed
    # spelling was replaced, point at the way to get it back -- the CLI's "search instead for ...".
    for line in explain(rep.resolution)
        println(stderr, "  ~ $line")
    end
    any(t -> !t.kept, rep.resolution.resolved) &&
        println(stderr, "  ~ to search as typed instead: --correction off")
    isempty(rep.expanded) ||
        println(stderr, "  + $(length(rep.expanded)) query_expansion(s) -> $expstr")
    # report what the profile actually carries, not what was requested: --no-lemmas on a
    # profile that never applied them changes nothing, and a probe command should say so
    println(stderr, "  matching with threshold=$(o["threshold"]) over $(length(qtokens)) token(s), " *
                    "lemmas=$(lemmas_on ? "on" : "off") " *
                    "(profile carries $(length(base.lemmas)), " *
                    "$(base.applied.lemmas ? "applied" : "not applied")), " *
                    "threads=$(Threads.nthreads())")

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
        # Partitioned explicitly rather than looping over 1:n so each task borrows ONE
        # tokenizer buffer and holds it across all its documents. A buffer must not be
        # shared between tasks, and the pool holds 2*nthreads()+4 of them, so this never
        # blocks. Slot assignment is unchanged: a task only ever writes out[i] for i in
        # its own range.
        ranges = _partition(n, Threads.nthreads())
        Threads.@threads for r in ranges
            tokenizerbuffer() do buff
                for i in r
                    if _matches(qtokens, texts[i], tc, buff, t)
                        out[i] = JSON3.write(records[i])
                    end
                end
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
