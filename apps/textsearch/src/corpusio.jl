# ── readers loaded on first use ──────────────────────────────────────────────
#
# Parquet2 costs 1.84 s to load and CSV 0.21 s, against ~4 s for a whole `textsearch --help`,
# and only a corpus in one of those two formats needs them. So they stay dependencies (installed
# and precompiled with the app) but are `require`d where they are used, not at module load.
#
# Loading a package mid-command defines its methods in a world newer than the command's own, so
# every call into it goes through `invokelatest` -- opening the table, each `iterate` on it, and
# the per-row conversion. Per row that is one dynamic call against a row decode, which is noise.
const _PARQUET2 = Base.PkgId(Base.UUID("98572fba-bba0-415d-956f-fa77e587d26d"), "Parquet2")
const _CSV = Base.PkgId(Base.UUID("336ed68f-0bac-5ca0-87d4-7b16caf5d00b"), "CSV")

struct _LatestWorld{T}
    it::T
end
Base.IteratorSize(::Type{<:_LatestWorld}) = Base.SizeUnknown()
Base.iterate(w::_LatestWorld, s...) = Base.invokelatest(iterate, w.it, s...)

# The VALUES have to come out as types the command's world already knows, not only the calls:
# CSV hands back `PosLenString`s and `InlineStrings`, whose methods are as new as the package, so
# printing or JSON-writing one later, back in the command's world, fails with "method too new".
# Strings are converted; numbers, `missing` and dates are Base types already. A column type outside
# those (a nested Parquet list, say) would need the same treatment, and would fail loudly if not.
_plain(v) = v isa AbstractString ? String(v) : v

"""
    _table_record(row, key) -> Pair{String,Dict{String,Any}}

A Tables.jl row (a `CSV.Row`, a Parquet2 row, ...) as `text => fields`, with every field in a
plain `Dict{String,Any}` so `search`'s hits can be re-serialized with `JSON3.write` whatever the
source format's native row type -- and, per `_plain`, from the command's own world.
"""
_table_record(row, key::Symbol) =
    String(Tables.getcolumn(row, key)) =>
        Dict{String,Any}(String(c) => _plain(Tables.getcolumn(row, c)) for c in Tables.columnnames(row))

function _lazy_rows(pkg::Base.PkgId, open_rows, path::AbstractString, key::Symbol)
    mod = Base.require(pkg)
    rows = Base.invokelatest(open_rows, mod, path)
    (Base.invokelatest(_table_record, row, key) for row in _LatestWorld(rows))
end

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
            # bytes, not the String: JSON3.read(::String) first asks `isfile(line)` of anything
            # under 255 bytes -- a stat per record, which on a network filesystem is a round trip
            let obj = JSON3.read(codeunits(line))
                String(obj[key]) => Dict{String,Any}(String(k) => v for (k, v) in pairs(obj))
            end
            for line in eachline(path) if !isempty(strip(line))
        )
    elseif format === :csv
        _lazy_rows(_CSV, (m, p) -> m.Rows(p), path, key)
    elseif format === :parquet
        _lazy_rows(_PARQUET2, (m, p) -> Tables.rows(m.readfile(p)), path, key)
    elseif format === :json
        (
            String(rec[key]) => Dict{String,Any}(String(k) => v for (k, v) in pairs(rec))
            for rec in JSON3.read(read(path))
        )
    else
        error("unsupported corpus format: $format; supported: plaintext, jsonl, csv, parquet, json")
    end
end
