[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/TextSearch.jl/dev)
[![Build Status](https://github.com/sadit/TextSearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/TextSearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/TextSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/TextSearch.jl)

# TextSearch.jl

`TextSearch.jl` is a Julia library for text preprocessing, tokenization, vocabulary management, vector-space modeling (BOW, TF, TF-IDF, entropy-based weightings), BM25 ranking, and full-text inverted indexes. It is designed to work seamlessly with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl) for high-performance, multithreaded similarity search over large text collections.

## Key Features and Components

- **Flexible Preprocessing & Tokenization (`TextConfig`)**:
  - Fine-grained character normalization: lowercase conversion, diacritic stripping, punctuation handling, emoji grouping/detection, and regex-based entity replacement (users, URLs, numbers).
  - Word $n$-gram tokenization (`nlist`), paragraph/sentence splitters (`tokenize_paragraphs`, `tokenize_sentences`), and custom token generators (`AbstractTokenGenerator`).
  - Extensible token transformations (`AbstractTokenTransformation`): stopword filtering (`IgnoreStopwords`), lemma normalization (`LemmaTransformation`), and chained pipelines (`ChainTransformation`). The package has **no dependencies beyond its own** -- no weak deps, no conditional code.
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
- **Portable Profiles (`TextProfile`, `save_profile`, `load_profile`, `zip_profile`, `merge_profiles`, `refit_profile`)**:
  - A `TextProfile` bundles vocabulary, weights, synonyms, lemmas and stopwords as plain, inspectable JSON -- no code is ever deserialized. Each artifact is stored once, with a marker saying whether the profile *applies* it, and the `TextConfig` it tokenizes with is derived from those -- so what a profile applies is always what it carries.
  - Whether a profile is a bootstrap model or one tuned to a dataset is read off its recorded lineage (`isbase`/`istuned`), not declared.
  - `merge_profiles` folds batched profiles of one corpus into an exact corpus-wide model; `refit_profile` adapts a generic profile to a specific dataset from a sample, adjusting statistics rather than replacing them.
- **Search Indexes & BM25 Ranking**:
  - `BM25InvertedFile`: Fast, merge-based BM25 scoring and retrieval over posting lists.
  - `FullText` & `TextInvertedFile`: High-level full-text search indexes that wrap corpus tokenization, weighting, and inverted index search into a unified interface.
  - Direct compatibility with `SimilaritySearch.jl` metric search indexes (`SearchGraph`, `ExhaustiveSearch`) and its `InvertedFiles` submodule.

## Processing Pipeline

```
Raw text / Corpus
  → TextConfig (normalization: diacritics, lowercase, urls, emojis / tokenization: n-grams
                / transformation: stopwords, lemmas -- derived from a profile)
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

# Package everything as a portable profile. `applied` says which artifacts are in the
# pipeline as opposed to merely carried -- a base model computes the lemma map but leaves
# applying it to whoever tunes from it.
profile = TextProfile(model;
                      stopwords=Set(stopword_candidates(voc, 0.9)),
                      lemmas,
                      synonyms=net.synonyms, synonym_distances=net.distances,
                      applied=AppliedArtifacts(stopwords=true),
                      lineage=[LineageStep(:fit; trainsize=length(corpus), outdim=4)])

dir = mktempdir()
save_profile(dir, profile)

p = load_profile(dir)
println("profile: vocsize=$(vocsize(p.model.voc)) synonyms=$(length(p.synonyms)) " *
        "lemmas=$(length(p.lemmas)) base=$(isbase(p))")

# Adapt the profile to a different dataset, given a sample of it. Statistics are adjusted
# rather than replaced: the profile acts as a prior, the sample as evidence.
sample = ["cats sleep on the sofa", "the sofa is warm for cats", "warm cats sleep all day"]
tuned = refit_profile(p, sample; verbose=false)
println("refitted: vocsize=$(vocsize(tuned.model.voc)) tuned=$(istuned(tuned))")
println("lineage:  ", lineage_summary(tuned))
```

Do not judge artifact quality from a six-document corpus: LSI needs real co-occurrence
statistics before its neighbours mean anything. The mechanics are the point here.

The [`textsearch` CLI app](apps/textsearch) drives this end to end -- fitting profiles over
large corpora in batches, merging them, refitting one against a dataset sample, and probing
the result -- without writing Julia.

## Documentation

Full documentation, tutorials, and API reference are available at:
- **[Latest documentation (dev)](https://sadit.github.io/TextSearch.jl/dev)**
- **[Hands-on Tutorial](https://sadit.github.io/TextSearch.jl/dev/tutorial/)**: Step-by-step guide covering vocabulary building, weighting schemes, BM25 indexing, stopwords, dense semantic representations, and integration with `WordTokenizers.jl`.

## Related Ecosystem Packages

- [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl): Approximate nearest neighbor search, graph indexes (`SearchGraph`), vector databases, and metric search algorithms.
- [`WordTokenizers.jl`](https://github.com/JuliaText/WordTokenizers.jl): Advanced natural language tokenizers compatible with `TextSearch.jl`.

## Contribute

Contributions are welcome! Please open an issue or pull request on GitHub for bug reports, documentation enhancements, or new features.

---

## Release Notes

### About the v1.1 series

A text model is now two things with a line between them, and that line runs through most of
this release:

- **Policy** -- a `TextConfig`: normalization, tokenization. Corpus-independent, writable by
  hand.
- **Artifacts** -- a `TextProfile`: stopword set, lemma map, synonym network, vocabulary
  counters, weights. Estimated from data.

They were tangled before. `TextConfig.transformation` held corpus-derived artifacts (a
stopword `Set`, a lemma map) while the profile stored the *same* artifacts again at its own
top level, with nothing tying the copies together. Both drifted: a refitted profile once
applied a 110,393-entry lemma map while saving and reporting the 40,320-entry one, and merging
two profiles carrying identical lemma maps was rejected as "incompatible". Merging shows why
the tangle hurt -- policies must be *identical* to merge, artifacts *combine* (union, rank
fusion, plurality vote) -- two opposite operations forced through one type.

Now each artifact has one home, and the `TextConfig` a profile tokenizes with is **derived**
from its policy plus whichever artifacts it applies. What a profile applies cannot differ from
what it saves, because there is only one copy.

**Changed:**

- **`TextProfile`** replaces the anonymous NamedTuple that `load_profile`/`merge_profiles`/
  `refit_profile` passed around. Field access is unchanged (`p.model`, `p.synonyms`,
  `p.lemmas`), with two renames: `stopword_candidates` became `stopwords` (one home, plus an
  `applied` marker), and `encoder` became `lineage`.
- **`save_profile(dir, profile)`** takes a profile rather than a model plus keywords.
- **Whether a profile is a base or a tuned model is derived from its `lineage`** --
  `isbase`/`istuned` -- rather than declared. A profile with no `:refit` step is a base; one
  with a refit is tuned; a refit of a refit stays tuned with no rule for it.
- **`expand_query_synonyms` is gone from `TextConfig`.** It was a search-time decision sitting
  in the tokenizer's config, read through three levels of nesting, governing data stored
  elsewhere. Handing an index a synonym network is now itself the request to expand with it;
  a profile records the intent as `applied.synonyms`.
- **Synonym networks store words and distances separately.** `synonyms(...)` returns
  `(; synonyms, distances)` -- neighbours *in rank order*, distances parallel. Only the ranking
  participates in query expansion (BM25 ignores query-side weights, and a merged or refitted
  network's distances are no longer distances in any single space), so a consumer can carry the
  ranking alone -- most of a real network's size.
- **`expand_synonyms!` weights by rank by default**, `1/rank` instead of `exp(-d)`; pass
  `distances` for the old behaviour.
- **The profile format is `"2.0"` and v1.0 profiles are refused by name.** There is no
  conversion path: carrying two layouts is what let the copies drift. Refit or refit-from-fit
  instead.

- **Field accessors are now `get<field>`**: `gettoken`, `getoccs`, `getndocs`, `gettrainsize`,
  `getnumtokens`, `getweight`, `gettextconfig`, `getpolicy`. The bare names were shadowable --
  a local variable or keyword argument called `trainsize` or `textconfig` silently hid the
  function, and the failure surfaced far away as `objects of type X are not callable`. That
  happened twice in one sitting, so it is designed out rather than remembered. **The old names
  still work**, with a deprecation warning naming the replacement.

**Removed:**

- **Snowball stemming and the package extension**, along with `Languages`' curated stopword
  lists. TextSearch now has **no weak dependencies and no conditional code**. Lemmatization
  covers the same ground and is corpus-derived rather than rule-based, but the honest cost is
  that morphological normalization now needs a fitted vocabulary: stemming worked on the first
  document. The stemmer was also the only thing that could make a well-formed profile
  *unreadable* (`load_profile` errored unless `using Snowball, Languages` was already active),
  and the only forced serialization point in an otherwise parallel tokenizer -- `Snowball.stem`
  races on its C handle's shared buffer, so every call was under one lock.
- Character q-grams and skip-grams, which had already been gone from the tokenizer for a while
  and only survived in the README's feature list.

**New:**

- **`refit_profile`**: adapts a bootstrap profile to a dataset from a sample, treating the
  profile as a prior worth `kappa` documents against the sample's evidence. A word the base
  considered important but the sample never shows survives with reduced weight; one that
  mattered in neither is dropped. Layered so any program can drive it (`refit_textconfig`,
  `fold_lemmas`, `blend_vocabularies`), and `--avgdoclen` pins BM25's length normalization to
  the sample when the two corpora's documents are nothing alike.
- **`extend_lemmas_morphological`**: recovers lemma families for tokens a base profile never
  saw, from surface similarity alone -- no embedding is fit.
- **`LemmaTransformation`**: lemma normalization in the pipeline, so it applies to documents
  and queries alike and the idf counts an inflection family together. Chained *before*
  `IgnoreStopwords`; the reverse order silently reintroduces stopwords.
- The tokenizer's borrowed-buffer API (`tokenizerbuffer`, `borrowtokenizedtext`,
  `TokenizerBuffer`) is exported.

**Migrating a synonym-network reader:**

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
  - Corpus-derived stopword detection, applied before the vocabulary the encoder trains on.
- **Parallelism & Performance**:
  - Multithreaded corpus processing, vocabulary construction, and vectorization powered by `@BATCHES`.
  - Zero-allocation vectorization buffers and SIMD-friendly adaptive sparse vector operations.
