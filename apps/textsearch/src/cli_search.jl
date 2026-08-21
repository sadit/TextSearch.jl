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
            help = "drop the profile's lemma step from the tokenization pipeline, so tokens " *
                   "are matched in their surface forms on both sides (no effect on a profile " *
                   "fitted without an applied lemma map)"
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
    _strip_lemmas(tt) -> AbstractTokenTransformation

Returns `tt` with every `LemmaTransformation` step removed, for `--no-lemmas`.

Lemmas live in the profile's `TextConfig` (a lemma is a normalization, so the tokenizer
applies it to documents and queries alike), which means turning them off is a matter of
taking that step out of the pipeline rather than declining to apply it afterwards.
"""
_strip_lemmas(tt::AbstractTokenTransformation) = tt
_strip_lemmas(::LemmaTransformation) = IdentityTokenTransformation()
function _strip_lemmas(tt::ChainTransformation)
    kept = AbstractTokenTransformation[s for s in tt.list if !(s isa LemmaTransformation)]
    length(kept) == 1 ? only(kept) : ChainTransformation(kept)
end

"""
    _has_lemmas(tt) -> Bool

Whether `tt` applies a lemma map, so the stderr summary can report what the profile is
actually doing rather than what was asked for (`--no-lemmas` on a profile fitted with
`[lemmas] apply = false` changes nothing, and saying so is the point of a probe command).
"""
_has_lemmas(tt::AbstractTokenTransformation) = false
_has_lemmas(::LemmaTransformation) = true
_has_lemmas(tt::ChainTransformation) = any(_has_lemmas, tt.list)

"""
    _query_tokens(p, query, tc, usesynonyms, synk) -> (Set{String}, report)

Builds the query's token set by running it through the same `tc` a document goes through --
so normalization, stopwords and lemmas all apply identically to both sides -- plus the one
step only a query gets: synonym expansion.

Each synonym is itself tokenized through `tc`, which is what lets it meet document tokens
on the same footing: a synonym stored in an inflected form arrives lemmatized, and one that
is a stopword drops out. `report` carries the intermediate sets for the stderr summary,
which is the point of this being a probe command: it shows which artifact contributed what.
"""
function _query_tokens(p, query::AbstractString, tc, usesynonyms::Bool, synk::Int)
    raw = collect(tokenize(tc, query))
    base = Set{String}(raw)

    expanded = Set{String}()
    if usesynonyms
        for tok in raw
            neighbors = get(p.synonyms, tok, nothing)
            neighbors === nothing && continue
            for (i, (syn, _)) in enumerate(neighbors)
                synk > 0 && i > synk && break
                for st in tokenize(tc, syn)
                    push!(expanded, st)
                end
            end
        end
        setdiff!(expanded, base)
    end

    union(base, expanded), (; raw, base, expanded)
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

    p = load_profile(_resolve_profile_path(o["profile"]))
    # Lemmas are part of the profile's TextConfig, so the tokenizer applies them to
    # documents and queries alike and there is nothing for this command to apply itself.
    # A profile fitted with [lemmas] apply = false simply has no such step.
    profile_tc = p.model.voc.textconfig
    tc = o["no-lemmas"] ?
        TextConfig(profile_tc; transformation=_strip_lemmas(profile_tc.transformation)) :
        profile_tc
    lemmas_on = _has_lemmas(tc.transformation)

    qtokens, rep = _query_tokens(p, o["query"], tc, !o["no-synonyms"], o["synonyms-k"])
    isempty(qtokens) && error("the query has no tokens under this profile's TextConfig " *
                              "(every term may have been a stopword); nothing could match")

    # what the pipeline actually did with the query -- on stderr, so stdout stays pure JSONL
    basestr = join(sort(collect(rep.base)), " ")
    expstr = join(sort(collect(rep.expanded)), " ")
    println(stderr, "query: $(length(rep.raw)) token(s) -> $basestr")
    isempty(rep.expanded) ||
        println(stderr, "  + $(length(rep.expanded)) synonym(s) -> $expstr")
    # report what the profile actually carries, not what was requested: --no-lemmas on a
    # profile that never had them changes nothing, and a probe command should say so
    println(stderr, "  matching with threshold=$(o["threshold"]) over $(length(qtokens)) token(s), " *
                    "lemmas=$(lemmas_on ? "on" : "off") (profile " *
                    "$(_has_lemmas(profile_tc.transformation) ? "carries" : "has no") lemma map), " *
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
