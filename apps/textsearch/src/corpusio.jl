"""
    _row_to_dict(row) -> Dict{String,Any}

Converts any Tables.jl-compliant row (a `CSV.Row`, a Parquet2 row, ...) into a plain
`Dict{String,Any}`, so `search`'s hits can always be re-serialized with `JSON3.write`
regardless of the source format's native row type.
"""
_row_to_dict(row) = Dict{String,Any}(String(c) => Tables.getcolumn(row, c) for c in Tables.columnnames(row))

"""
    each_record(format::Symbol, path::AbstractString, text_key::AbstractString)

Yields `(text::String, record::Dict{String,Any})` pairs one at a time, without
materializing the whole file, for `format in (:plaintext, :jsonl, :csv, :parquet, :json)`.
`text_key` names the column/JSON-key holding the document text (ignored for `:plaintext`,
which instead splits the file into paragraph-level documents via `tokenize_paragraphs`).

Streaming properties, by format:
- `:jsonl`/`:csv` are genuinely bounded-memory (`eachline`/`CSV.Rows` never hold more than
  one row in memory at a time).
- `:parquet` streams at row-group granularity (lazy column-chunk reads via Parquet2) --
  not strictly one-row-at-a-time, but not whole-file either.
- `:json` (a single top-level JSON array) CANNOT stream with JSON3: the closing `]` must
  be seen before any element can be trusted as complete. This is an inherent format
  limitation, not a bug -- avoid `:json` for large corpora, prefer `:jsonl`.
"""
function each_record(format::Symbol, path::AbstractString, text_key::AbstractString)
    key = Symbol(text_key)
    if format === :plaintext
        (p => Dict{String,Any}("text" => p) for p in tokenize_paragraphs(read(path, String)))
    elseif format === :jsonl
        (
            let obj = JSON3.read(line)
                String(obj[key]) => Dict{String,Any}(String(k) => v for (k, v) in pairs(obj))
            end
            for line in eachline(path) if !isempty(strip(line))
        )
    elseif format === :csv
        (String(Tables.getcolumn(row, key)) => _row_to_dict(row) for row in CSV.Rows(path))
    elseif format === :parquet
        (String(Tables.getcolumn(row, key)) => _row_to_dict(row) for row in Tables.rows(Parquet2.readfile(path)))
    elseif format === :json
        (
            String(rec[key]) => Dict{String,Any}(String(k) => v for (k, v) in pairs(rec))
            for rec in JSON3.read(read(path))
        )
    else
        error("unsupported corpus format: $format; supported: plaintext, jsonl, csv, parquet, json")
    end
end
