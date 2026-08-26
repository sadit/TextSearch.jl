[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/TextSearch.jl/dev)
[![Build Status](https://github.com/sadit/TextSearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/TextSearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/TextSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/TextSearch.jl)

# TextSearch.jl

`TextSearch.jl` is a Julia library for text preprocessing, tokenization, vocabulary management, vector-space modeling (BOW, TF, TF-IDF, entropy-based weightings), BM25 ranking, and full-text inverted indexes. It is designed to work seamlessly with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl) for high-performance, multithreaded similarity search over large text collections.

## Key Features and Components

- **Flexible Preprocessing & Tokenization (`TextConfig`)**:
  - Fine-grained character normalization: lowercase conversion, diacritic stripping, punctuation handling, emoji grouping/detection, and regex-based entity replacement (users, URLs, numbers).
  - Word $n$-gram tokenization (`nlist`), paragraph/sentence splitters (`tokenize_paragraphs`, `tokenize_sentences`), and custom token generators (`AbstractTokenGenerator`).
  - A fixed per-token pipeline (`TokenPipeline`): a lemma map applied first, then a stopword set -- plain data in one order, since the two stages do not commute and the wrong order silently readmits stopwords. New *kinds of token* extend `TokenizationConfig.generators` instead. The package has **no dependencies beyond its own** -- no weak deps, no conditional code.
- **Vocabulary & Bag-of-Words (`Vocabulary`, `BOW`)**:
  - Fast token $\leftrightarrow$ ID mappings, document frequency tracking, and vocabulary pruning/filtering.
  - Efficient multithreaded corpus processing via `SimilaritySearch.@BATCHES`.
- **Vector Space Models (`VectorModel`)**:
  - Local weighting schemes: `TfWeighting`, `FreqWeighting`, `BinaryLocalWeighting`, `TpWeighting`.
  - Global weighting schemes: `IdfWeighting`, `BinaryGlobalWeighting`.
  - Supervised & entropy weighting: `EntropyWeighting` and `CombineWeighting` for text classification.
  - High-performance sparse vector representations (`SparseVector`, `SparseVecView`) with SIMD-accelerated and adaptive sparse dot products, cosine similarities, and centroids.
- **Semantic Artifacts (`LSI`, `query_expansion`, `lemma_clusters`, `stopword_candidates`)**:
  - Latent semantic indexing with an exact truncated SVD, dense or ARPACK-based, chosen by corpus size.
  - Query expansion networks built by (optionally approximate) all-pairs kNN over token embeddings, storing the neighbour ranking and its distances separately.
  - Lemma maps derived by grouping inflections morphologically and splitting them semantically.
- **One Query Pipeline (`query_tokens`, `QueryPipeline`, `queryvector`/`querybow`/`querytokenset`)**:
  - Correction then expansion, on strings, once -- both inverted files and the CLI go through it, so a query cannot mean one thing to an index and another to a search command. The representation decides what to do with the weights it produces: weighted and normalized for a cosine index, presence-only for BM25 (whose scoring never reads the query side's frequencies), a plain set for token matching.
- **Query Correction (`derive_variants`, `resolve_query_tokens`, `QueryPolicy`)**:
  - Orthographic bridging so a profile can keep case and diacritics without becoming unsearchable: a query typed `leon` or `musica` reaches `León` and `música`. Derived from the vocabulary rather than stored, since it is a pure function of it.
  - Reportable and answerable literally: `explain` says what was searched and why, and `QueryPolicy(correction=:off)` gives back the query exactly as typed -- the "search instead for ..." escape a search that corrects by default owes the person who typed it.
- **Portable Profiles (`fit_profile`, `TextProfile`, `save_profile`, `load_profile`, `zip_profile`, `merge_profiles`, `refit_profile`)**:
  - `fit_profile(textconfig, corpus)` distills a corpus in one call, in the order the three passes have to happen in: stopwords are detected and removed *before* the vocabulary the encoder trains on, lemma families are found from the embeddings that need to exist first, and applying them rebuilds the vocabulary once more.
  - `BM25InvertedFile(profile)` / `TextInvertedFile(profile)` index with it: the idf, `avgdoclen` and tokenization are the **corpus's**, the document lengths are the **index's**.
  - A `TextProfile` bundles vocabulary, weights, query_expansion, lemmas and stopwords as plain, inspectable JSON -- no code is ever deserialized. Each artifact is stored once, with a marker saying whether the profile *applies* it, and the `TextConfig` it tokenizes with is derived from those -- so what a profile applies is always what it carries.
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
                / pipeline: lemmas then stopwords -- derived from a profile)
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
per-token embeddings (LSI), a query_expansion network, and a lemma map. Together with the vocabulary
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

# A query_expansion network: neighbour tokens in rank order, with distances kept separately, since
# only the ranking takes part in query expansion
net = query_expansion(lsi, 2; verbose=false)

# A lemma map: inflections are grouped by surface similarity, then split by meaning
lemmas = lemma_clusters(voc, wordvecs)

# All of the above in one call, in the order the passes have to happen in:
oneshot = fit_profile(config, corpus; encoder=(; outdim=4), expansion=(; k=2), verbose=false)
println("one call: vocsize=$(vocsize(oneshot.model.voc)) base=$(isbase(oneshot))")

# ...or assembled by hand, which is the same thing spelled out. `applied` says which
# artifacts are in the pipeline as opposed to merely carried -- a base model computes the
# lemma map but leaves applying it to whoever tunes from it.
profile = TextProfile(model;
                      stopwords=Set(stopword_candidates(voc, 0.9)),
                      lemmas,
                      query_expansion=net.query_expansion, query_expansion_distances=net.distances,
                      applied=AppliedArtifacts(stopwords=true),
                      lineage=[LineageStep(:fit; trainsize=length(corpus), outdim=4)])

dir = mktempdir()
save_profile(dir, profile)

p = load_profile(dir)
println("profile: vocsize=$(vocsize(p.model.voc)) query_expansion=$(length(p.query_expansion)) " *
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

### 4. Correcting and Expanding a Query

A search makes two guesses about what someone meant: that a spelling they typed is the one the
corpus uses, and that the words they typed are the only ones worth matching. Both are guesses, so
both are answerable literally -- a [`QueryPolicy`](@ref) travels with the query and says which to
make.

```julia
# TextSearch v1.1
using TextSearch, SimilaritySearch

# Keeping case and diacritics lets the corpus distinguish senses that folding destroys, at the
# cost that a query typed without them matches nothing. Correction bridges that -- at query
# time only, since applying it to documents would blur the very distinctions it exists to reach.
corpus = [
    "la música clásica del siglo XIX",
    "un festival de música popular",
    "la música es un arte",
    "el sol brilla en el cielo",
    "el sol de la mañana",
    "una nota musical: sol mayor",
    "El Sol es una estrella",
    "musica italiana del renacimiento",     # the unaccented spelling, in one document
]

config = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false, del_punc=true),
                    tokenization=TokenizationConfig(nlist=[1]))
voc = Vocabulary(config, corpus)

# The map is DERIVED from the vocabulary rather than stored: it holds only the spellings that
# cannot be computed from a folded form, so `madrid -> Madrid` is absent while accents are not.
variants = derive_variants(voc)
println("bridged:   ", sort(collect(variants)))   # sorted: Dict order is not guaranteed

# `:auto` corrects only where the evidence says the typed spelling is wrong -- here `musica`
# is in one document against `música`'s three -- and where it corrects, it replaces.
r = resolve_query_tokens(voc, ["musica"], variants, QueryPolicy(negligible_ratio=2))
println("corrected: ", r.tokens)
println("           ", only(explain(r)))

# A well-typed token is left exactly as typed, even though `Sol` exists beside `sol`
println("untouched: ", resolve_query_tokens(voc, ["sol"], variants).tokens)

# ...and `:off` answers the literal query: the "search instead for ..." escape a consumer
# that corrects by default owes the person who typed it
println("literal:   ", resolve_query_tokens(voc, ["musica"], variants,
                                            QueryPolicy(correction=:off)).tokens)
```

Output:

```
bridged:   ["clasica" => ["clásica"], "manana" => ["mañana"], "musica" => ["música"]]
corrected: ["música"]
           musica is in 1 document against música's 3, so it reads as a misspelling; searched as música (variant) instead
untouched: ["sol"]
literal:   ["musica"]
```

Correction is orthographic and exact: it reaches other spellings of the same word (case and
diacritics), every one of them a real vocabulary token. Expansion is semantic and approximate: it
reaches *different* words, from the profile's `query_expansion` network. They compose in one
direction -- correction runs first, and expansion draws from the commonest spelling of what it
produced, which is what keeps a bridge to a rare spelling from dragging that spelling's unreliable
neighbours into the query.


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
- **Artifacts** -- a `TextProfile`: stopword set, lemma map, query_expansion network, vocabulary
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
  `refit_profile` passed around. Field access is unchanged (`p.model`, `p.query_expansion`,
  `p.lemmas`), with two renames: `stopword_candidates` became `stopwords` (one home, plus an
  `applied` marker), and `encoder` became `lineage`.
- **`save_profile(dir, profile)`** takes a profile rather than a model plus keywords.
- **Whether a profile is a base or a tuned model is derived from its `lineage`** --
  `isbase`/`istuned` -- rather than declared. A profile with no `:refit` step is a base; one
  with a refit is tuned; a refit of a refit stays tuned with no rule for it.
- **`expand_query_synonyms` is gone from `TextConfig`.** It was a search-time decision sitting
  in the tokenizer's config, read through three levels of nesting, governing data stored
  elsewhere. Handing an index a query_expansion network is now itself the request to expand with it;
  a profile records the intent as `applied.query_expansion`.
- **Query expansion networks store words and distances separately.** `query_expansion(...)` returns
  `(; query_expansion, distances)` -- neighbours *in rank order*, distances parallel. Only the ranking
  participates in query expansion (BM25 ignores query-side weights, and a merged or refitted
  network's distances are no longer distances in any single space), so a consumer can carry the
  ranking alone -- most of a real network's size.
- **`expand_query!` weights by rank by default**, `1/rank` instead of `exp(-d)`; pass
  `distances` for the old behaviour.

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
- **`TokenPipeline`**: replaces the `AbstractTokenTransformation` hierarchy with a struct of two
  data fields, a lemma map and a stopword set, applied in that fixed order. Lemmas apply to
  documents and queries alike, so the idf counts an inflection family together; the order is
  load-bearing, since filtering first lets `"las"` past a set holding `"la"` and only then
  rewrites it. Measured on 120,000 paragraphs, the concrete field types also made applying
  lemmas free: 14.65s -> 10.44s, against 10.28s for the filter alone.
- **Query correction** (`derive_variants`, `resolve_query_tokens`, `QueryPolicy`, `explain`):
  orthographic bridging so a profile can keep case and diacritics without becoming
  unsearchable. Preserving them separates senses that folding destroys -- on Spanish Wikipedia
  `granada` unfolded reaches the heraldic charge as well as the city, likewise `cuba` the
  barrel and `leon` the animal -- at the cost that a query typed `leon` matches nothing. The
  map holds only the spellings that cannot be *computed* from a folded form, so `madrid ->
  Madrid` is generated at query time and only accent restoration is derived, and it is derived
  from the vocabulary rather than stored, being a pure function of it.

  `QueryPolicy` decides what to do with it. `:auto` corrects only where the evidence says the
  typed spelling is wrong -- absent, or negligible beside a commoner spelling of the same word
  -- and where it corrects it *replaces*; `:off` answers the query exactly as typed, which is
  the "search instead for ..." escape a search that corrects by default owes the person who
  typed it; `:always` bridges without evidence. `explain` renders what happened as a
  comparison, since an absolute count is not a reason: *"musica is in 1020 documents against
  música's 199211, so it reads as a misspelling"*.
- **`fit_profile`**: a corpus becomes a [`TextProfile`](#) in one call, with the three passes in
  the order they have to be in -- stopwords detected and removed *before* the vocabulary the
  encoder trains on, lemma families found from embeddings that must exist first, and a rebuild
  under the lemma map when it is applied. This used to live only in the CLI, 110 lines of it, so
  every other caller reinvented the order; the README and the tutorial had each reinvented it
  differently. Keywords are grouped as the concerns they belong to (`stopwords`, `encoder`,
  `expansion`, `lemmas`) and map 1:1 onto the app's config file.
- **One query pipeline** (`query_tokens`, `QueryPipeline`, `queryvector`/`querybow`/
  `querytokenset`). There were two and neither could do what the other did: the vector-level path
  weighted what it added but iterated token *ids*, so a misspelling absent from the vocabulary
  could never be corrected; the CLI's path corrected but added terms unweighted. The unified one
  runs on strings and emits weights, and both inverted files go through it -- so a query cannot
  mean one thing to an index and another to `textsearch search`. `expand_query!` is no longer
  called anywhere inside the package.
- **`BM25InvertedFile(profile)` / `TextInvertedFile(profile)`**: index with a fitted profile,
  which is what having profiles is for. The idf, `avgdoclen` and tokenization are the corpus's,
  the document lengths are the index's. Expansion follows the profile's `applied` marker
  (`expansion=true` overrides, which is what a base profile needs); correction follows the
  `QueryPolicy`, since it depends on nothing but the vocabulary.
- **Shorter ways to say the usual thing**, with the long ones intact: `TextConfig(lc=false)`
  forwards flat keywords to the sub-configs (`nlist=[1]` was written in fifty places while
  already being the default), and `VectorModel(voc)` is TF-IDF. The config types also define
  `==` by value, which they did not -- Julia compared their fields with `===` and several are
  heap objects, so two configs built from the same settings came out unequal.
- The tokenizer's borrowed-buffer API (`tokenizerbuffer`, `borrowtokenizedtext`,
  `TokenizerBuffer`) is exported.

**Worth knowing:** a fit is *not* bit-reproducible. Neighbour ties are now ordered
deterministically, but the factorization under the embeddings runs threaded, so its reductions sum
in a nondeterministic order and values differ around the seventh significant digit -- enough to
flip a near-tie across the top-k boundary. Two fits of one corpus are equivalent, not equal, so a
published profile has to be verified structurally (counters, stopwords, vocabulary, network keys,
distances to a tolerance) rather than by checksum.

**Migrating a query_expansion-network reader:**

```julia
# v1.0
net = query_expansion(lsi, 8)
for (neighbour, distance) in net["dog"]; end

# v1.1
net = query_expansion(lsi, 8)
for (rank, neighbour) in enumerate(net.query_expansion["dog"])
    distance = net.distances["dog"][rank]   # optional; the ranking alone is usually enough
end

# and to keep expand_query!' previous distance-based weighting
expand_query!(vec, voc, net.query_expansion; distances=net.distances)
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
