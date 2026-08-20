#!/usr/bin/env julia
#
# Generic parquet -> JSONL converter for corpus-profile builds.
#
#   julia --project=apps/textsearch lib/parquet_to_jsonl.jl OUT.jsonl SHARD.parquet... \
#         [--text-column text] [--limit N] [--min-chars N] [--keep-columns id,title,url]
#
# Streams row by row (Parquet2 reads a row group at a time, so peak memory stays bounded by
# the largest row group, not the corpus) and writes one JSON object per line with a "text"
# key -- the shape `textsearch fit --config` expects for `format = "jsonl"`.
#
# Not Wikipedia-specific: any parquet corpus with a text column works, which is why this
# lives in lib/ rather than in a per-corpus driver.

using Parquet2, JSON3, Tables

function parse_args(argv)
    argv = copy(argv)
    opts = Dict{String,Any}(
        "text-column" => "text", "limit" => 0, "min-chars" => 0, "keep-columns" => String[],
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
        elseif a == "--keep-columns"
            opts["keep-columns"] = split(popfirst!(argv), ',', keepempty=false)
        elseif startswith(a, "--")
            error("unknown option: $a")
        else
            push!(positional, a)
        end
    end
    length(positional) >= 2 ||
        error("usage: parquet_to_jsonl.jl OUT.jsonl SHARD.parquet... [--text-column C] [--limit N] [--min-chars N] [--keep-columns a,b]")
    opts["out"] = positional[1]
    opts["shards"] = positional[2:end]
    opts
end

function main(argv)
    opts = parse_args(argv)
    textcol = Symbol(opts["text-column"])
    keepcols = Symbol.(opts["keep-columns"])
    limit = opts["limit"]::Int
    minchars = opts["min-chars"]::Int

    written = 0
    skipped_short = 0
    skipped_missing = 0
    t0 = time()

    mkpath(dirname(abspath(opts["out"])))
    open(opts["out"], "w") do io
        for (si, shard) in enumerate(opts["shards"])
            ds = Parquet2.Dataset(shard)
            nshard = 0
            for row in Tables.rows(ds)
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

                rec = Dict{String,Any}("text" => text)
                for c in keepcols
                    v = getproperty(row, c)
                    rec[String(c)] = v === missing ? nothing : v
                end
                println(io, JSON3.write(rec))

                written += 1
                nshard += 1
                if limit > 0 && written >= limit
                    @info "reached --limit" limit written
                    @goto done
                end
                written % 100_000 == 0 && @info "progress" written elapsed_s=round(time() - t0; digits=1)
            end
            @info "shard done" shard=basename(shard) index=si of=length(opts["shards"]) rows=nshard total=written
        end
        @label done
    end

    @info "wrote JSONL" out=opts["out"] records=written skipped_short skipped_missing elapsed_s=round(time() - t0; digits=1)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
