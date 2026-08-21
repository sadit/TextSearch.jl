[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/TextSearch.jl/dev)
[![Build Status](https://github.com/sadit/TextSearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/TextSearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/TextSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/TextSearch.jl)

# TextSearch.jl

`TextSearch.jl` is a Julia library for text preprocessing, tokenization, vocabulary management, vector-space modeling (BOW, TF, TF-IDF, entropy-based weightings), BM25 ranking, and full-text inverted indexes. It is designed to work seamlessly with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl) for high-performance, multithreaded similarity search over large text collections.

## Key Features and Components

- **Flexible Preprocessing & Tokenization (`TextConfig`)**:
  - Fine-grained character normalization: lowercase conversion, diacritic stripping, punctuation handling, emoji grouping/detection, and regex-based entity replacement (users, URLs, numbers).
  - Word $n$-gram tokenization (`nlist`), paragraph/sentence splitters (`tokenize_paragraphs`, `tokenize_sentences`), and custom token generators (`AbstractTokenGenerator`).
  - Extensible token transformations (`AbstractTokenTransformation`): stemming via Snowball (through `TextSearchSnowballExt`), stopword filtering (`IgnoreStopwords`), lemma normalization (`LemmaTransformation`), and chained pipelines (`ChainTransformation`).
- **Vocabulary & Bag-of-Words (`Vocabulary`, `BOW`)**:
  - Fast token $\leftrightarrow$ ID mappings, document frequency tracking, and vocabulary pruning/filtering.
  - Efficient multithreaded corpus processing via `SimilaritySearch.@BATCHES`.
- **Vector Space Models (`VectorModel`)**:
  - Local weighting schemes: `TfWeighting`, `FreqWeighting`, `BinaryLocalWeighting`, `TpWeighting`.
  - Global weighting schemes: `IdfWeighting`, `BinaryGlobalWeighting`.
  - Supervised & entropy weighting: `EntropyWeighting` and `CombineWeighting` for text classification.
  - High-performance sparse vector representations (`SparseVector`, `SparseVecView`) with SIMD-accelerated and adaptive sparse dot products, cosine similarities, and centroids.
- **Semantic Artifacts (`LSI`, `synonyms`, `lemma_clusters`, `stopword_candidates`)**:
  - Latent semantic indexing with an exact truncated SVD, dense or ARPACK-based, chosen by corpus size.
  - Synonym networks built by (optionally approximate) all-pairs kNN over token embeddings, storing the neighbour ranking and its distances separately.
  - Lemma maps derived by grouping inflections morphologically and splitting them semantically.
  - Query-time synonym expansion (`expand_synonyms!`) for both sparse-vector and BM25 queries -- applied to queries only, never to documents.
- **Portable Profiles (`save_profile`, `load_profile`, `zip_profile`, `merge_profiles`, `refit_profile`)**:
  - A profile bundles vocabulary, weights, synonyms, lemmas and stopword candidates as plain, inspectable JSON -- no code is ever deserialized.
  - `merge_profiles` folds batched profiles of one corpus into an exact corpus-wide model; `refit_profile` adapts a generic profile to a specific dataset from a sample, adjusting statistics rather than replacing them.
- **Search Indexes & BM25 Ranking**:
  - `BM25InvertedFile`: Fast, merge-based BM25 scoring and retrieval over posting lists.
  - `FullText` & `TextInvertedFile`: High-level full-text search indexes that wrap corpus tokenization, weighting, and inverted index search into a unified interface.
  - Direct compatibility with `SimilaritySearch.jl` metric search indexes (`SearchGraph`, `ExhaustiveSearch`) and its `InvertedFiles` submodule.

## Processing Pipeline

```
Raw text / Corpus
  → TextConfig (normalization: diacritics, lowercase, urls, emojis / tokenization: n-grams
                / transformation: stopwords, lemmas, stemming)
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
# TextSearch v1.1
using TextSearch, SimilaritySearch

# Sample documents
corpus = [
    "Machine learning and natural language processing in Julia",
    "High performance similarity search and vector indexing",
    "Natural language text retrieval with BM25 and inverted files",
    "Julia programming language for scientific computing and machine learning"
]

# Configure tokenization (unigrams + bigrams)
config = TextConfig(tokenization=TokenizationConfig(nlist=[1, 2]))

# Build vocabulary and TF-IDF vector model
voc = Vocabulary(config, corpus)
model = VectorModel(IdfWeighting(), TfWeighting(), voc)

# Index the corpus: TextInvertedFile wraps vectorization and the posting lists
idx = TextInvertedFile(model; dist=Dist.NormCosine())
ctx = InvertedFileContext()
append_items!(idx, ctx, corpus)

# Search nearest documents for a query
res = search(idx, ctx, "machine learning and text retrieval in Julia", knnqueue(KnnSorted, 3))

for (id, d) in zip(res.ids, res.dists)
    println("doc $id (cosine distance $(round(d; digits=4))): ", corpus[id])
end
```

### 2. BM25 Inverted File Search

```julia
# TextSearch v1.1
using TextSearch, SimilaritySearch

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "brown fox jumps high over the lazy fence",
    "quick brown dogs and cats in the park",
    "the lazy dog sleeps all day"
]

# Create and populate a BM25 inverted index. The index is built from a Vocabulary, and
# documents are added separately -- so a corpus larger than memory can be streamed in.
voc = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1])), corpus)
invfile = BM25InvertedFile(voc)
ctx = InvertedFileContext()
append_items!(invfile, ctx, corpus)

# Query the BM25 index (returns top-k documents ranked by BM25 score)
res = search(invfile, ctx, "quick brown fox", knnqueue(KnnSorted, 2))

for (doc_id, dist) in zip(res.ids, res.dists)
    # dist is the negated BM25 score, so it is negative and MORE negative means more relevant
    println("Doc $doc_id: $(corpus[doc_id]) (distance: $(round(dist; digits=4)))")
end
```

### 3. Semantic Artifacts and Portable Profiles

Beyond indexing, a corpus can be distilled into artifacts that travel with the model:
per-token embeddings (LSI), a synonym network, and a lemma map. Together with the vocabulary
and weights they form a **profile** -- a directory of plain JSON files (or a zip of them)
that can be shipped, inspected, and adapted.

```julia
# TextSearch v1.1
using TextSearch, SimilaritySearch

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "brown foxes jump over lazy dogs every morning",
    "quick dogs and lazy cats share the park",
    "a lazy dog sleeps while the cats play",
    "foxes and dogs are both animals of the forest",
    "the forest is full of quick animals",
]

config = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
voc = Vocabulary(config, corpus)
model = VectorModel(IdfWeighting(), TfWeighting(), voc)

# Latent semantic indexing gives one vector per vocabulary token
lsi = LatentSemanticIndexing(model, corpus; maxoutdim=4, verbose=false)
wordvecs = wordvectors(lsi)

# A synonym network: neighbour tokens in rank order, with distances kept separately, since
# only the ranking takes part in query expansion
net = synonyms(lsi, 2; verbose=false)

# A lemma map: inflections are grouped by surface similarity, then split by meaning
lemmas = lemma_clusters(voc, wordvecs)

# Package everything as a portable profile
dir = mktempdir()
save_profile(dir, model;
             synonyms=net.synonyms, synonym_distances=net.distances, lemmas,
             stopword_candidates=stopword_candidates(voc, 0.9))

p = load_profile(dir)
println("profile: vocsize=$(vocsize(p.model.voc)) synonyms=$(length(p.synonyms)) lemmas=$(length(p.lemmas))")

# Adapt the profile to a different dataset, given a sample of it. Statistics are adjusted
# rather than replaced: the profile acts as a prior, the sample as evidence.
sample = ["cats sleep on the sofa", "the sofa is warm for cats", "warm cats sleep all day"]
tuned = refit_profile(p, sample; verbose=false)
println("refitted: vocsize=$(vocsize(tuned.model.voc)) trainsize=$(trainsize(tuned.model.voc))")
```

Do not judge artifact quality from a six-document corpus: LSI needs real co-occurrence
statistics before its neighbours mean anything. The mechanics are the point here.

The [`textsearch` CLI app](apps/textsearch) drives this end to end -- fitting profiles over
large corpora in batches, merging them, refitting one against a dataset sample, and probing
the result -- without writing Julia.

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

### About the v1.1 series

Changed, in how synonym networks are represented:

- **Words and distances are stored separately.** `synonyms(...)` returns `(; synonyms,
  distances)` -- a token maps to its neighbours *in rank order*, the distances live in a
  parallel structure -- and `load_profile` exposes them as two fields. Only the ranking takes
  part in query expansion (BM25 ignores query-side weights entirely, and once a network is
  merged or refitted its distances are no longer distances in any single space), so keeping
  them apart lets a consumer, or a profile on disk, carry the ranking alone. On a real profile
  that is most of the file.
- **`expand_synonyms!` weights by rank by default**, `1/rank` instead of `exp(-d)`.

Profiles written by v1.0 keep loading: they store the two interleaved, and that layout is
recognized by shape and split on load. Code that reads a network in memory needs updating:

```julia
# v1.0
net = synonyms(lsi, 8)
for (neighbour, distance) in net["dog"]; end

# v1.1
net = synonyms(lsi, 8)
for (rank, neighbour) in enumerate(net.synonyms["dog"])
    distance = net.distances["dog"][rank]   # optional; the ranking alone is usually enough
end

# and to keep expand_synonyms!' previous distance-based weighting
expand_synonyms!(vec, voc, net.synonyms; distances=net.distances)
```

New:

- **Lemma normalization in the pipeline (`LemmaTransformation`)**: a lemma is a normalization,
  so it belongs in the `TextConfig`, where it applies to documents and queries alike and the
  idf counts an inflection family together instead of splitting it across forms. Chain it
  *before* `IgnoreStopwords` -- the reverse order silently reintroduces stopwords.
- **`refit_profile`**: adapts a bootstrap profile to a dataset from a sample of it, treating
  the profile as a prior worth `kappa` documents against the sample's evidence. A word the
  base considered important but the sample never shows survives with reduced weight; one that
  mattered in neither is dropped. Layered so any program can drive it (`refit_textconfig`,
  `fold_lemmas`, `blend_vocabularies`).
- **`extend_lemmas_morphological`**: recovers lemma families for tokens a base profile never
  saw, from surface similarity alone -- no embedding is fit.
- The tokenizer's borrowed-buffer API (`tokenizerbuffer`, `borrowtokenizedtext`,
  `TokenizerBuffer`) is now exported, along with the transformation-pipeline predicates
  `has_lemma_transformation` / `with_lemma_transformation` / `without_lemma_transformation`.

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
