[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/TextSearch.jl/dev)
[![Build Status](https://github.com/sadit/TextSearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/TextSearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/TextSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/TextSearch.jl)

# TextSearch.jl

`TextSearch.jl` is a Julia library for text preprocessing, tokenization, vocabulary management, vector-space modeling (BOW, TF, TF-IDF, entropy-based weightings), BM25 ranking, and full-text inverted indexes. It is designed to work seamlessly with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl) for high-performance, multithreaded similarity search over large text collections.

## Key Features and Components

- **Flexible Preprocessing & Tokenization (`TextConfig`)**:
  - Fine-grained character normalization: lowercase conversion, diacritic stripping, punctuation handling, emoji grouping/detection, and regex-based entity replacement (users, URLs, numbers).
  - Multi-scale tokenizers: word $n$-grams, character $q$-grams, skip-grams, paragraph/sentence splitters, and custom token generators (`AbstractTokenGenerator`).
  - Extensible token transformations (`AbstractTokenTransformation`): stemming via Snowball (through `TextSearchSnowballExt`), stopword filtering (`IgnoreStopwords`), and chained pipelines (`ChainTransformation`).
- **Vocabulary & Bag-of-Words (`Vocabulary`, `BOW`)**:
  - Fast token $\leftrightarrow$ ID mappings, document frequency tracking, and vocabulary pruning/filtering.
  - Efficient multithreaded corpus processing via `SimilaritySearch.@BATCHES`.
- **Vector Space Models (`VectorModel`)**:
  - Local weighting schemes: `TfWeighting`, `FreqWeighting`, `BinaryLocalWeighting`, `TpWeighting`.
  - Global weighting schemes: `IdfWeighting`, `BinaryGlobalWeighting`.
  - Supervised & entropy weighting: `EntropyWeighting` and `CombineWeighting` for text classification.
  - High-performance sparse vector representations (`SparseVector`, `SparseVecView`) with SIMD-accelerated and adaptive sparse dot products, cosine similarities, and centroids.
- **Search Indexes & BM25 Ranking**:
  - `BM25InvertedFile`: Fast, merge-based BM25 scoring and retrieval over posting lists.
  - `FullText` & `TextInvertedFile`: High-level full-text search indexes that wrap corpus tokenization, weighting, and inverted index search into a unified interface.
  - Direct compatibility with `SimilaritySearch.jl` metric search indexes (`SearchGraph`, `ExhaustiveSearch`, `InvertedFile`).

## Processing Pipeline

```
Raw text / Corpus
  → TextConfig (preprocessing options: diacritics, lowercase, urls, emojis, q-grams, n-grams)
  → normalize_text (character-level normalization)
  → tokenize (produces a TokenizedText or list of token strings)
  → Vocabulary (token ⇄ id table, corpus statistics, filtering)
  → bagofwords (BOW per document)
  → VectorModel (local/global weighting schemes → SparseVector)
       or
  → BM25InvertedFile / TextInvertedFile (posting lists + kNN search)
```

## Installing TextSearch

You can install `TextSearch.jl` using Julia's package manager:

```julia
] add TextSearch
```

To run the test suite:

```julia
] test TextSearch
```

## Quick Example

### 1. Vector Model and Inverted Index Search

```julia
using TextSearch, SimilaritySearch

# Sample documents
corpus = [
    "Machine learning and natural language processing in Julia",
    "High performance similarity search and vector indexing",
    "Natural language text retrieval with BM25 and inverted files",
    "Julia programming language for scientific computing and machine learning"
]

# Configure tokenization (unigrams + bigrams)
config = TextConfig(nlist=[1, 2], qlist=[])

# Build vocabulary and TF-IDF vector model
voc = Vocabulary(config, corpus)
model = VectorModel(IdfWeighting(), TfWeighting(), voc)

# Vectorize corpus into sparse vectors
X = vectorize_corpus(model, corpus)

# Index vectors with an inverted file using normalized cosine distance
db = VectorDatabase(X)
invfile = InvertedFile(Dist.NormCosine(), db)
index!(invfile)

# Search nearest documents for a query
query = "machine learning and text retrieval in Julia"
qvec = vectorize(model, query)
res = search(invfile, qvec, 3)

println("Nearest neighbor doc IDs: ", res.ids)
println("Cosine distances: ", res.dists)
```

### 2. BM25 Inverted File Search

```julia
using TextSearch, SimilaritySearch

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "brown fox jumps high over the lazy fence",
    "quick brown dogs and cats in the park",
    "the lazy dog sleeps all day"
]

# Create and populate a BM25 inverted index
invfile = BM25InvertedFile(corpus; config=TextConfig(nlist=[1]))

# Query the BM25 index (returns top-k documents ranked by BM25 score)
query = "quick brown fox"
res = search(invfile, query, 2)

for (doc_id, dist) in zip(res.ids, res.dists)
    # dist is negative score (or 1 / (1 + score)) for metric consistency
    println("Doc $doc_id: $(corpus[doc_id]) (score: $dist)")
end
```

## Documentation

Full documentation, tutorials, and API reference are available at:
- **[Latest documentation (dev)](https://sadit.github.io/TextSearch.jl/dev)**
- **[Hands-on Tutorial](https://sadit.github.io/TextSearch.jl/dev/tutorial/)**: Step-by-step guide covering vocabulary building, weighting schemes, BM25 indexing, language-aware stemming and stopwords, and integration with `WordTokenizers.jl`.

## Related Ecosystem Packages

- [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl): Approximate nearest neighbor search, graph indexes (`SearchGraph`), vector databases, and metric search algorithms.
- [`Snowball.jl`](https://github.com/JuliaText/Snowball.jl) & [`Languages.jl`](https://github.com/JuliaText/Languages.jl): Multi-language stemming and stopword dictionaries (integrated via `TextSearchSnowballExt`).
- [`WordTokenizers.jl`](https://github.com/JuliaText/WordTokenizers.jl): Advanced natural language tokenizers compatible with `TextSearch.jl`.

## Contribute

Contributions are welcome! Please open an issue or pull request on GitHub for bug reports, documentation enhancements, or new features.

---

## Release Notes

### About v1.0 series

- **SimilaritySearch.jl v1.1.0 Integration**:
  - Full compatibility with `SimilaritySearch.jl` v1.1.0 (struct-of-arrays kNN result layout, modern `InvertedFile` and `SearchGraph` APIs).
  - Adopted shared sparse-vector operations and intersection algorithms from `SimilaritySearch.jl`.
- **Full-Text Inverted Index (`FullText` & `TextInvertedFile`)**:
  - New high-level full-text search abstractions managing the lifecycle from raw text to inverted file index queries.
- **BM25 Search Optimizations**:
  - High-performance merge-based BM25 scoring over sparse vector representations.
  - Support for `SparseVecView` and `SparseVector` search queries.
- **Enhanced Tokenization & Preprocessing**:
  - Added sentence and paragraph tokenizers (`tokenize_sentences`, `tokenize_paragraphs`).
  - Generalized `AbstractTokenGenerator` and `AbstractTokenTransformation` pipelines.
  - Language-aware stemming and stopword removal via weak dependency extensions (`TextSearchSnowballExt`).
- **Parallelism & Performance**:
  - Multithreaded corpus processing, vocabulary construction, and vectorization powered by `@BATCHES`.
  - Zero-allocation vectorization buffers and SIMD-friendly adaptive sparse vector operations.
