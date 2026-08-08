# AGENTS.md

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
  → VectorModel (vmodel.jl, emodel.jl) local/global weighting schemes → SparseVector{Float32,Int32}
       or
  → BM25 (bm25.jl) + BM25InvertedFile (bm25invfile.jl, bm25invfilesearch.jl)  index + kNN search
```

`dvec.jl` provides utilities and fast operations (`sparsedot`, `centroid`)
on `SparseVector{Float32,Int32}` and conversions with `BOW`.

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
| `dvec.jl` | Optimized operations (`sparsedot`, `centroid`) on `SparseVector` |
| `vmodel.jl` | `VectorModel`, local/global weighting schemes (TF, IDF, TP, binary), `vectorize` |
| `emodel.jl` | Entropy-based weighting schemes (`EntropyWeighting`, `CombineWeighting`) |
| `bm25.jl` | `BM25` scoring struct and `bm25score`/`tokenscore` |
| `intersections/` | Search and set intersection algorithms (`doublingsearch`, `binarysearch`, `bk!`, `bkt!`, `umerge!`, `imerge2!`, `svs`, `xmerge!`) |
| `invertedfiles/` | `PostingList`, `SortedIntSet`, `IdWeight`, `WeightedInvertedFile`, `BinaryInvertedFile`, and search routines |
| `deprecated.jl` | Backwards-compatible shims for renamed/removed APIs |

## Dev environment

```julia
# from repo root
julia -t 3 --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the test suite:

```julia
julia -t 3 --project=. -e 'using Pkg; Pkg.test()'
# or, from a REPL with the project active:
] test
```

Individual test files live under `test/` and are included from `test/runtests.jl`:
`tok.jl` (tokenization), `voc.jl` (vocabulary), `vec.jl` (vector models/weighting),
`intersections.jl` (intersection algorithms), `search.jl` (BM25 inverted file search).

Build the docs (Documenter.jl):

```julia
julia --project=docs docs/make.jl
```

CI (`.github/workflows/ci.yml`) runs on Julia 1.12 for Ubuntu and Windows and uploads
coverage to Codecov. `documentation.yml` builds and deploys docs on pushes/tags to `main`.

## Conventions and gotchas

- **Julia version floor is 1.10** (`Project.toml` `[compat] julia = "^1.10"`). Don't use
  syntax/stdlib features newer than that.
- **Always run Julia commands with `-t 3`** (e.g. `julia -t 3 --project=.`) for multithreaded evaluation and test runs.
- **`Aqua.jl` runs in the test suite** (`test/runtests.jl`): ambiguity and type-piracy
  checks (`Aqua.test_ambiguities`, `Aqua.test_piracies`). If you add methods that
  extend `Base`/`LinearAlgebra`/`SparseArrays` functions on non-owned types, add them to
  the `treat_as_own` list in `runtests.jl` if they're intentional, otherwise avoid the
  piracy.
- **Sparse vectors are `SparseVector{Float32,Int32}`** (from `SparseArrays.jl`). `vectorize`/`vectorize!` return `SparseVector`. `BOW = Dict{UInt32,Int32}` is a raw count dictionary. `SVEC = Dict{UInt32,Float32}` exists only as a legacy type alias. Fast operations like `sparsedot` and `centroid` live in `dvec.jl`.
- **Indexing and Key Extraction**: `PostingList` and `SortedIntSet` implement standard `Base.getindex` (`plist[i]`) returning the primary key (`UInt32`), eliminating the legacy `getkey` abstraction. Search and intersection algorithms in `Intersections` (`binarysearch`, `doublingsearch`) operate directly on `A[i]` using semi-open ranges `[sp, ep)` and bitshift division `(sp + ep) >>> 1`. `_dot_gallop` in `dvec.jl` uses `doublingsearch`.
- **Buffer pooling**: `bagofwords`/`bagofwords!`/`bagofwords_corpus` (`bow.jl`) do **not** use buffer pooling or global caches; they allocate/fill a plain `BOW` (`Dict{UInt32,Int32}`) per document, pre-sized up front from `voc`'s `avgdoclen` via `sizehint!(bow, _bow_sizehint(voc))`. Channel-based per-thread pools are restricted to `voc.jl` (`BOW_CACHES` for `Vocabulary` building), `vmodel.jl` (`VectorizeBuffer`/`VECTORIZE_CACHES` for `vectorize!`), and `Tokenizer` (`TokenizerBuffer`/`TOKENIZER_CACHES` for tokenization scratch space borrowed on demand).
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
  - **Contexts & Batching**: `InvertedFileContext` scratch buffers (`positions`, `cont_u32`, `cont_iw`, `cont_iiw`, `knns`) are sized to `maxbatches` and indexed by `ctx.batchid`. Contexts are created on demand for each operation (`getcontext(invfile)` returns a fresh `InvertedFileContext`), with no global cached singleton.
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
