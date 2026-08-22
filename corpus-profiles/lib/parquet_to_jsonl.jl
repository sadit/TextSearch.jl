#!/usr/bin/env julia
#
# Generic parquet -> JSONL converter for corpus-profile builds.
#
#   julia --project=apps/textsearch lib/parquet_to_jsonl.jl OUT.jsonl SHARD.parquet... \
#         [--text-column text] [--limit N] [--min-chars N] [--min-tokens N] \
#         [--split-paragraphs] [--title-column title] [--keep-columns id,title,url]
#
# Streams row by row (Parquet2 reads a row group at a time, so peak memory stays bounded by
# the largest row group, not the corpus) and writes one JSON object per line with a "text"
# key -- the shape `textsearch fit --config` expects for `format = "jsonl"`.
#
# Not Wikipedia-specific: any parquet corpus with a text column works, which is why this
# lives in lib/ rather than in a per-corpus driver.
#
# With `--split-paragraphs`, one record is emitted per paragraph instead of per document, each
# prefixed with the document's title. Measured on 10,000 Spanish Wikipedia articles that turns
# 10,000 documents into ~293,000: 36 blocks per article on average (median 18, p90 90), of which
# 18.5% fall under `--min-tokens 4`. Those discards are almost entirely bare section headings --
# 9.9% of all blocks are a single word -- which is the point rather than a side effect: a
# document frequency computed over paragraphs separates a real stopword from a Wikipedia
# artifact, because `de` is in nearly every paragraph while `referencias` is a one-word heading
# block that never survives the filter, though it sits in 82% of articles.

using Parquet2, JSON3, Tables

function parse_args(argv)
    argv = copy(argv)
    opts = Dict{String,Any}(
        "text-column" => "text", "limit" => 0, "min-chars" => 0, "keep-columns" => String[],
        # A minimum in TOKENS, not characters: with paragraphs as documents the useful floor is
        # "does this have any content at all", and 4 words is that floor. Characters cannot
        # express it -- "Referencias" is 11 characters and no content, while "no fue así" is 10
        # and a sentence.
        "min-tokens" => 0, "split-paragraphs" => false, "title-column" => "title",
    )
    positional = String[]
    while !isempty(argv)
        a = popfirst!(argv)
        if a == "--text-column"
            opts["text-column"] = popfirst!(argv)
        elseif a == "--limit"
            opts["limit"] = parse(Int, popfirst!(argv))
        elseif a == "--min-chars"
            opts["min-chars"] = parse(Int, popfirst!(argv))
        elseif a == "--min-tokens"
            opts["min-tokens"] = parse(Int, popfirst!(argv))
        elseif a == "--split-paragraphs"
            opts["split-paragraphs"] = true
        elseif a == "--title-column"
            opts["title-column"] = popfirst!(argv)
        elseif a == "--keep-columns"
            opts["keep-columns"] = split(popfirst!(argv), ',', keepempty=false)
        elseif startswith(a, "--")
            error("unknown option: $a")
        else
            push!(positional, a)
        end
    end
    length(positional) >= 2 ||
        error("usage: parquet_to_jsonl.jl OUT.jsonl SHARD.parquet... [--text-column C] " *
              "[--limit N] [--min-chars N] [--min-tokens N] [--split-paragraphs] " *
              "[--title-column C] [--keep-columns a,b]")
    opts["out"] = positional[1]
    opts["shards"] = positional[2:end]
    opts
end

"""
    paragraphs(text) -> Vector{SubString}

Splits one document into its paragraph blocks on blank lines, trimmed, empties dropped.

Wikipedia's plain-text rendering separates paragraphs by `\n\n` and leaves a section heading as
the FIRST line of its section's first block, joined to the prose by a single `\n`
("Nacionalidades y evolución demográfica \nSegún el censo de 2008, ..."). Splitting on blank
lines therefore keeps a heading attached to the paragraph it introduces, which is where it
belongs -- and leaves a heading with no prose under it as its own tiny block, which
`--min-tokens` then drops.
"""
paragraphs(text::AbstractString) =
    [b for b in (strip(x) for x in split(text, "\n\n")) if !isempty(b)]

"whitespace-separated word count -- a cheap proxy for tokens, no tokenizer needed here"
nwords(s::AbstractString) = count(!isempty, split(s))

"""
    looks_like_heading(block; maxwords=12) -> Bool

Whether a paragraph block is probably a section heading rather than prose: short, and not
ending in sentence punctuation. Deliberately loose -- the action taken on a positive is to glue
the block to the next paragraph, which costs nothing when the guess is wrong (a short list item
simply travels with the paragraph under it) and preserves real information when it is right.
"""
looks_like_heading(b::AbstractString; maxwords::Int=12) =
    nwords(b) <= maxwords && !endswith(rstrip(b), ('.', '!', '?', ':', ';', ',', ')', '"'))

"""
    paragraph_units(text; maxheadwords=12) -> Vector{String}

Paragraph blocks with heading-looking blocks folded FORWARD into the paragraph they introduce.

Wikipedia already glues a heading to its section's first paragraph when prose follows it
immediately, but a heading whose section opens with a list, a table or an infobox lands as its
own block -- 9.9% of all blocks on 10,000 Spanish articles are a single word. Dropping those on
`--min-tokens` throws away the one label that says what the following text is about, so they are
carried forward instead, and several in a row accumulate ("Historia" then "Siglo XIX"). A
heading with nothing after it is dropped, since there is no paragraph left to label.
"""
function paragraph_units(text::AbstractString; maxheadwords::Int=12)
    out = String[]
    pending = String[]
    for b in paragraphs(text)
        if looks_like_heading(b; maxwords=maxheadwords)
            push!(pending, String(b))
        elseif isempty(pending)
            push!(out, String(b))
        else
            push!(out, string(join(pending, "\n"), "\n", b))
            empty!(pending)
        end
    end
    out
end

function main(argv)
    opts = parse_args(argv)
    textcol = Symbol(opts["text-column"])
    titlecol = Symbol(opts["title-column"])
    keepcols = Symbol.(opts["keep-columns"])
    limit = opts["limit"]::Int
    minchars = opts["min-chars"]::Int
    mintokens = opts["min-tokens"]::Int
    split_paras = opts["split-paragraphs"]::Bool

    written = 0
    skipped_short = 0
    skipped_missing = 0
    sources = 0
    t0 = time()

    mkpath(dirname(abspath(opts["out"])))
    open(opts["out"], "w") do io
        for (si, shard) in enumerate(opts["shards"])
            ds = Parquet2.Dataset(shard)
            nshard = 0
            for row in Tables.rows(ds)
                # Checked before the row is touched, so `--limit` always means whole source
                # documents: with --split-paragraphs a mid-document stop would emit a truncated
                # article and skew every statistic derived from it.
                if limit > 0 && sources >= limit
                    @info "reached --limit source documents" limit sources written
                    @goto done
                end
                text = getproperty(row, textcol)
                if text === missing || text === nothing
                    skipped_missing += 1
                    continue
                end
                text = String(text)
                if length(text) < minchars
                    skipped_short += 1
                    continue
                end
                sources += 1

                # Common fields, resolved once per source document rather than per paragraph.
                extra = Dict{String,Any}()
                for c in keepcols
                    v = getproperty(row, c)
                    extra[String(c)] = v === missing ? nothing : v
                end

                # The title goes in front of every paragraph on purpose: a paragraph on its own
                # loses what it is about ("cuenta con 81.588 habitantes" names no place), and the
                # title is the one piece of context every paragraph of the document shares.
                title = if split_paras
                    tv = getproperty(row, titlecol)
                    (tv === missing || tv === nothing) ? "" : String(tv)
                else
                    ""
                end

                units = split_paras ? paragraph_units(text) : [text]
                for (pi, unit) in enumerate(units)
                    if mintokens > 0 && nwords(unit) < mintokens
                        skipped_short += 1
                        continue
                    end
                    body = isempty(title) ? String(unit) : string(title, "\n", unit)
                    rec = Dict{String,Any}("text" => body)
                    merge!(rec, extra)
                    split_paras && (rec["paragraph"] = pi)
                    println(io, JSON3.write(rec))

                    written += 1
                    nshard += 1
                    written % 100_000 == 0 && @info "progress" written sources elapsed_s=round(time() - t0; digits=1)
                end
            end
            @info "shard done" shard=basename(shard) index=si of=length(opts["shards"]) rows=nshard total=written
        end
        @label done
    end

    @info "wrote JSONL" out=opts["out"] records=written source_documents=sources skipped_short skipped_missing elapsed_s=round(time() - t0; digits=1)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
