# AGENTIC.md

Guidance for AI coding agents (Claude Code and similar) working in this repository.

## What this package is

`TextSearch.jl` is a Julia package that turns text into vector/sparse representations
(BOW, TF, TF-IDF, entropy-based weightings, BM25) and provides an inverted-file index
(via [`InvertedFiles.jl`](https://github.com/sadit/InvertedFiles.jl)) for searching them.
It is meant to be paired with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl).
The package was previously named `TextModel.jl`.

## Processing pipeline (mental model)

Understanding the data flow makes it much easier to find where a change belongs:

```
raw text/corpus
  → TextConfig (textconfig.jl)         preprocessing options (lowercase, diacritics, url/user/number
                                        grouping, q-grams, n-grams, skip-grams, token transforms)
  → normalize_text (normalize.jl)      character-level normalization
  → tokenize (tokenize.jl)             produces a TokenizedText (list of token strings)
  → Vocabulary (voc.jl, updatevoc.jl)  token ⇄ id mapping, occurrence/doc-frequency counters
  → BOW / bagofwords (bow.jl)          Dict{UInt32,Int32} bag-of-words per document
  → VectorModel (vmodel.jl, emodel.jl) local/global weighting schemes → SVEC (sparse Dict vector)
       or
  → BM25 (bm25.jl) + BM25InvertedFile (bm25invfile.jl, bm25invfilesearch.jl)  index + kNN search
```

`sparseconversions.jl` / `dvec.jl` bridge between the package's `Dict`-based sparse
vectors (`BOW = Dict{UInt32,Int32}`, `SVEC = Dict{UInt32,Float32}`) and
`SparseArrays`/`SimilaritySearch` vector types (arithmetic, norms, distances).

## Source file map

| File | Responsibility |
|---|---|
| `TextSearch.jl` | Module entry point, includes, `BOW`/`SVEC` type aliases |
| `textconfig.jl` | `TextConfig`, `Skipgram` — tokenization/preprocessing configuration |
| `tokentrans.jl` | `AbstractTokenTransformation` hooks (stemming, stopwords, chaining) |
| `normalize.jl` | Character-level text normalization, emoji detection |
| `tokenize.jl` | `TokenizedText`, `tokenize`/`tokenize_corpus`, q-grams/n-grams/skip-grams |
| `voc.jl` | `Vocabulary` type: token↔id table, occurrence/ndocs counters |
| `updatevoc.jl` | Merging/updating `Vocabulary` instances |
| `approxvoc.jl` | Approximate vocabulary lookup (`QgramsLookup`) for fuzzy/OOV matching |
| `bow.jl` | Bag-of-words construction from tokenized text |
| `sparseconversions.jl` | Conversions between `Dict` sparse vectors and `SparseArrays` |
| `dvec.jl` | Arithmetic/distance operations (`+`, `-`, `dot`, `norm`, centroid, Cosine/Angle) on `SVEC` |
| `vmodel.jl` | `VectorModel`, local/global weighting schemes (TF, IDF, TP, binary), `vectorize` |
| `emodel.jl` | Entropy-based weighting schemes (`EntropyWeighting`, `CombineWeighting`) |
| `multi.jl` | Merging/joining `VectorModel`s (`update!`, `joinmodel`) — requires `KCenters` |
| `bm25.jl` | `BM25` scoring struct and `bm25score`/`tokenscore` |
| `bm25invfile.jl` | `BM25InvertedFile` — the inverted-file index built on `InvertedFiles.jl` |
| `bm25invfilesearch.jl` | kNN search over `BM25InvertedFile` (uses `Intersections.jl`) |
| `deprecated.jl` | Backwards-compatible shims for renamed/removed APIs |

## Dev environment

```julia
# from repo root
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the test suite:

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
# or, from a REPL with the project active:
] test
```

Individual test files live under `test/` and are included from `test/runtests.jl`:
`tok.jl` (tokenization), `voc.jl` (vocabulary), `vec.jl` (vector models/weighting),
`search.jl` (BM25 inverted file search).

Build the docs (Documenter.jl):

```julia
julia --project=docs docs/make.jl
```

CI (`.github/workflows/ci.yml`) runs on Julia 1.10 for Ubuntu and Windows and uploads
coverage to Codecov. `documentation.yml` builds and deploys docs on pushes/tags to `main`.

## Conventions and gotchas

- **Julia version floor is 1.10** (`Project.toml` `[compat] julia = "^1.10"`). Don't use
  syntax/stdlib features newer than that.
- **`Aqua.jl` runs in the test suite** (`test/runtests.jl`): ambiguity and type-piracy
  checks (`Aqua.test_ambiguities`, `Aqua.test_piracies`). If you add methods that
  extend `Base`/`LinearAlgebra`/`SparseArrays` functions on non-owned types, add them to
  the `treat_as_own` list in `runtests.jl` if they're intentional, otherwise avoid the
  piracy.
- **Sparse vectors are plain `Dict`s**, not `SparseVector`: `BOW = Dict{UInt32,Int32}`,
  `SVEC = Dict{UInt32,Float32}` (defined in `TextSearch.jl`). Arithmetic/distance
  operators for them live in `dvec.jl` — extend there, not ad hoc elsewhere.
- **Buffer pooling**: separate channel-based per-thread pools avoid allocations —
  `BOW_CACHES` (a `Channel{BOW}`) in `voc.jl` backs `Vocabulary` building only;
  `bagofwords`/`bagofwords!`/`bagofwords_corpus` (`bow.jl`) take/create a plain
  [`BOW`](@ref) directly (no wrapper struct), `sizehint!`ed up front from
  `voc`'s [`avgdoclen`](@ref) via `_bow_sizehint`; `VectorizeBuffer`/`VECTORIZE_CACHES`
  in `vmodel.jl` backs the performance-sensitive `vectorize!` path (sorts/RLEs token ids
  directly, no `Dict` involved); and `TokenizerBuffer`/`TOKENIZER_CACHES` inside the
  `Tokenizer` module (use `tokenizerbuffer(f)`) backs tokenization scratch space, borrowed
  on demand by the others rather than duplicated.
- **Parallelism uses SimilaritySearch's `@BATCHES`** (v1.0+; no `Polyester` dependency
  anywhere in this package anymore). Simple per-item loops use the one-argument form
  (`@BATCHES getminbatch(n) for i in 1:n ... end`); `voc.jl`'s `tokenize_and_append!` uses
  the 5-section form (`@BEGINBATCH`/`@LOOP`/`@ENDBATCH`) to merge per-batch counters into
  `Vocabulary` once per batch instead of once per document. Always call `getminbatch(n)`
  with **one** argument — `getminbatch(n, nt)`'s second argument is a thread count, not a
  second corpus-size-like quantity; passing anything corpus-derived there silently returns
  a useless batch size (this was a real, previously-unnoticed bug in this codebase before
  the v1.0 migration, since the loops using it were `Threads.@threads`, which doesn't even
  take a `minbatch`).
  - **Known residual risk**: `InvertedFileContext`'s search-time scratch buffers
    (`positions`, `cont_u32`, `cont_iw`, `cont_iiw`, `knns`, sized by `Threads.maxthreadid()`)
    are still indexed by `Threads.threadid()` (`getcontainer`/`getpositions` in
    `invertedfiles/invfile.jl`). This is unchanged by the v1.0 migration (SimilaritySearch's
    own internal state moved to `@batchid()`-indexing instead, precisely to avoid this class
    of bug — see its `AGENTS.md`/`set_batch_scheduler!` docs), but TextSearch's version
    wasn't ported because `InvertedFileContext` is a single global cached singleton
    (`DEFAULT_CACHE_INVFILES[]`, returned by `getcontext`) shared across whatever
    parallelizes concurrent `search`/`push_item!` calls *outside* TextSearch's control — a
    clean `@batchid()` fix needs that caller to tag a per-batch context copy (`@set ctx.batchid
    = @batchid()`) before calling in, which isn't a contract TextSearch can enforce today.
    Only touch this if you're prepared to either add that per-call-site tagging contract, or
    redesign away from one-shared-context-with-N-thread-slots entirely (e.g. one context per
    concurrent caller, no internal arrays at all).
- **Struct immutability**: most core types (`TextConfig`, `Vocabulary`, `BM25`, …) are
  immutable `struct`s with explicit copy-constructors (e.g. `TextConfig(c::TextConfig; kwargs...)`)
  for "update a field" patterns — follow that pattern instead of adding mutability.
- **Custom `Base.show` methods** exist for most structs with a `prefix`/`indent` keyword
  convention for nested pretty-printing — match this style for new types.
- **Docstrings** use standard Julia docstring format with a signature block followed by
  a description and `- field: description` bullet lists; keep new public API documented
  the same way and cross-reference with `` [`Type`](@ref) ``.
- This package depends on unreleased/companion packages from the same author
  (`InvertedFiles.jl`, `Intersections.jl`, `SimilaritySearch.jl`) — check their APIs
  when a search/index-related change looks like it needs upstream changes too.

## Where to look for more context

- `README.md` — install/usage overview, links to Pluto notebook examples in `examples/`.
- `docs/src/index.md`, `docs/src/api.md` — narrative docs and API reference source.
- `examples/invindex.jl`, `examples/searchgraph.jl` — Pluto notebooks demonstrating
  end-to-end usage.
