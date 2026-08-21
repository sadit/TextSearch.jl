# textsearch

A command-line application for fitting, searching, and managing
[TextSearch.jl](https://github.com/sadit/TextSearch.jl) profiles -- precomputed vocabulary,
weights, synonym networks, clustering-derived lemmas, and stopword candidates for a text
corpus, packaged as a single `.zip` you can install, share, and query.

- **[Install](#install)**
- **[Manual](#manual)** -- one section per subcommand, with every option
- **[Tutorial](#tutorial)** -- a worked example from a raw corpus to a search hit

## Requirements

**Julia ≥ 1.12.**

## Install

`textsearch` is a [Julia app](https://pkgdocs.julialang.org/dev/apps/) (Pkg's app support is
currently experimental). Install it once, from the repo root:

```sh
julia -e 'using Pkg; pkg"app develop apps/textsearch"'
```

This registers a `textsearch` executable under `~/.julia/bin/`, managed by Pkg. Make sure
`~/.julia/bin` is on your `PATH`, then run it directly:

```sh
textsearch <subcommand> [options]
```

`app develop` points the installed app at this checked-out source -- edits to
`apps/textsearch` or the parent `TextSearch` package take effect without reinstalling.

If you'd rather not install anything, run it as a plain script (this is what every example
below does, so you can copy-paste them without installing first):

```sh
julia --project=apps/textsearch apps/textsearch/src/main.jl <subcommand> [options]
```

## Manual

### `fit` -- build a profile from a corpus

```
textsearch fit [--config CONFIG]
```

Fits a profile from a corpus. Instead of a wall of flags, `fit` opens a TOML config file in
your `$EDITOR` (falling back to `vi` if `$EDITOR` is unset), visudo-style: it writes a
commented template, blocks until you save and exit, then reads it back and runs. Pass
`--config path/to/fit.toml` to skip the editor entirely and read a config file directly --
useful for scripting, and it lets you version-control a config for reuse.

The config has seven tables:

```toml
[input]
format = "jsonl"        # "plaintext" | "csv" | "jsonl" | "json" | "parquet"
path = "corpus.jsonl"
text_key = "text"        # column/JSON-key holding the document text; ignored for "plaintext"

[output]
dir = "./profiles"
prefix = "corpus"
batch_size = 10000       # max docs per output profile .zip; 0 = unbounded (one profile)

[normalization]           # see TextSearch.NormalizationConfig for what each flag does
del_diac = true
del_dup = false
del_punc = false
group_num = true
group_url = true
group_usr = false
group_emo = false
lc = true

[tokenization]
nlist = [1]               # [1] = unigrams; e.g. [1, 2] adds word bigrams too
mark_token_type = true

[stopwords]
enabled = true
doc_freq_threshold = 0.5  # tokens in more than this fraction of documents are candidates

[encoder]
kind = "lsi"              # "lsi" | "external"
outdim = 128               # kind="lsi": target LSI dimension
scaling = "none"           # kind="lsi": "none" | "inv_singular_values" | "singular_values"
external_path = ""         # kind="external": path to a JSON {token: [vector]} mapping

[synonyms]
k = 8                      # neighbors per token in the synonym network

[lemmas]
algorithm = "fft"          # "fft" | "dnet" | "randsel" | "multirandsel"
num_clusters = 0            # 0 = auto (sqrt(vocabulary size))
selector = "shortest"       # "shortest" | "most_frequent" | "shortest_then_most_frequent"
apply = true                # bake the map into the profile's TextConfig (see below)
```

Notes:

- **Formats.** `plaintext` is split into paragraph-level documents. `csv`/`jsonl`/`json`
  pull `text_key` out of each row/object; `parquet` does the same via any column readable
  through Tables.jl. Corpus and query records don't need to be pre-tokenized.
- **Batching.** A large corpus produces several `.zip` files (`prefix-0001.zip`,
  `prefix-0002.zip`, ...), each an *independent* profile: its own vocabulary, weights,
  synonyms, and lemmas, computed only from that batch's documents -- nothing is shared or
  averaged across batches. Combining several profiles back into one is `merge`'s job (see
  below); `fit` never does that itself.
- **Where lemmas are applied.** With `lemmas.apply = true` (the default) the lemma map is
  baked into the profile's `TextConfig` as a `LemmaTransformation`, chained *before*
  `IgnoreStopwords`. A lemma is a normalization, so it belongs on both sides: once it is in
  the `TextConfig`, `vectorize`/`bagofwords`/the inverted files/`search` all apply it to
  documents and queries alike, and the idf counts a whole inflection family together instead
  of splitting it across its forms. This needs a *third* tokenization pass, because the map
  is derived from embeddings over the vocabulary it rewrites and so cannot be known any
  earlier; the synonym network is rewritten onto lemmas at the same time, since its entries
  would otherwise name tokens the vocabulary no longer has and be dropped in silence. LSI is
  deliberately not recomputed -- the embeddings' job was to find the families. Set
  `apply = false` to keep the map as a reviewable artifact only, the same detected-versus-
  applied distinction stopword candidates have.
- **Stopwords and encoder ordering.** When `stopwords.enabled = true`, candidates are
  detected from a first, unfiltered tokenization pass, then wired into the *real*
  vocabulary's `TextConfig` before that vocabulary is built -- so stopword tokens never
  enter the vocabulary the encoder (tf-idf weights, and LSI if used) trains on. This means
  the corpus is tokenized twice when stopwords are enabled; that's expected, not a
  performance bug to chase.
- **Encoder.** `"lsi"` fits an LSI projection internally (see `LSI.LatentSemanticIndexing`
  in the library). `"external"` instead reads a precomputed `token -> vector` JSON mapping
  from `external_path`; either way the resulting per-token vectors feed the same
  synonym-network and lemma-clustering steps.
- On a small/toy corpus (a handful of short documents, as in the tutorial below), don't
  expect polished synonyms or lemmas -- LSI needs real co-occurrence statistics to separate
  meaningful clusters from noise. The mechanics are still worth trying at that scale; just
  don't judge output quality from it.

### `search` -- grep-like search over a collection

```
textsearch search <profile> <query> --collection PATH [--format FORMAT]
                   [--text-key KEY] [-t THRESHOLD]
                   [--no-lemmas] [--no-synonyms] [--synonyms-k K] [--chunk N]
```

`profile` is an installed nickname or a path to a profile `.zip`/directory. Prints every
record of `--collection` whose text shares at least `--threshold` tokens with `query`, as
one JSONL line per hit:

```sh
textsearch search mynick "red car" --collection reviews.jsonl -t 2
```

`-t`/`--threshold` (default `1`) is the minimum token-set intersection size -- `1` matches
if *any* query token is shared (union), raising it toward the query's own token count
gives progressively stricter, AND-like matching (the same `t`-threshold idea
`SimilaritySearch.InvertedFiles` uses). There is no ranking or scoring; matches print in
corpus-encounter order, exactly like `grep`.

**The whole pipe runs, and each artifact applies where it belongs.** Normalization,
tokenization, stopwords and lemmas come from the profile's `TextConfig`, so they apply
identically to the query and to every document -- that is what a normalization means.
Synonym expansion applies to the **query only**: it widens what the query reaches, and
applying it to documents would make everything match everything. Each synonym is itself run
through the same `TextConfig`, so one stored in an inflected form arrives lemmatized and
meets document tokens on the same footing.

This makes `search` the way to exercise a profile's artifacts end to end, so each can be
switched off to see what it contributes: `--no-lemmas` removes the lemma step from the
tokenization pipeline (on both sides), `--no-synonyms` skips expansion, and `--synonyms-k`
caps how many synonyms each query token may contribute (`0`, the default, uses every one the
profile stored). The effective query token set, what the synonyms added, and whether the
profile actually carries a lemma map are all reported on **stderr**, leaving stdout pure
JSONL for piping.

Matching runs on all available threads (the installed shim passes `--threads=auto`), over
buffers of `--chunk` records at a time. Output does not depend on either: each task writes
only its own slot in a preallocated vector and does no I/O, and printing happens
single-threaded afterwards in index order, so there is exactly one writer and hits always
come out in corpus order. `--chunk` bounds memory, not results. The speedup is roughly
2.6x end to end rather than the thread count, because a large share of the time is GC: the
tokenizer allocates a fresh `String` per token, which dominates matching and is not
something parallelism or buffer reuse can recover.

**This is NOT fast to start** -- loading the profile and opening the collection has real
cost, unlike a real `grep`. Prefer it over `grep` only when you need corpus-consistent
tokenization/normalization (accents, casing, number/URL grouping, stopword removal, ...),
not raw byte-level matching. Collections stream rather than load fully into memory: JSONL
and CSV are read one line/row at a time, Parquet at row-group granularity; a plain JSON
array cannot stream (JSON3 needs to see the closing `]` before trusting any element) and
is read whole -- prefer JSONL for large collections.

### Profile management -- `list` / `info` / `install` / `uninstall`

Installed profiles live under `~/.textsearch/profiles/` (override the whole base directory
with the `TEXTSEARCH_HOME` environment variable).

```
textsearch install <path.zip> [nickname] [--force]   # copy a profile zip in, under a nickname
textsearch list                                       # nicknames of everything installed
textsearch info <nickname>                            # corpus stats, TextConfig, file path
textsearch uninstall <nickname>                       # print the file's path -- does NOT delete it
```

- `install` derives the nickname from the zip's filename if you don't give one explicitly;
  `--force` overwrites an existing nickname (without it, a name collision is an error).
- `info` prints `trainsize`/`vocsize`/`numtokens`/`avgdoclen`, how many synonym/lemma/
  stopword-candidate entries were saved, the encoder used, the full `TextConfig`
  (normalization + tokenization + transformation), and the file's absolute path.
- `uninstall` is deliberately non-destructive: it only looks up and prints the installed
  file's path, it never deletes anything. `textsearch` never silently removes a profile you
  may have spent real compute producing -- remove it yourself once you're sure, with the
  `rm` command it hands you.

### `merge` -- fold batched profiles into one

```
textsearch merge <profiles...> --out OUT.zip [--doc-freq-threshold F] [--synonyms-k N]
```

This is what makes `fit`'s batching usable: batching a large corpus produces one
*independent* profile per batch, and `merge` folds them back into a single corpus-wide
profile. Each input may be an installed nickname, a path to a profile `.zip`/directory, or
a **directory containing** profile `.zip`s -- the usual case, since `fit` writes a whole
batch of them into one output directory:

```sh
textsearch fit --config wiki-es.toml            # -> profiles/wiki20231101-es/*.zip
textsearch merge profiles/wiki20231101-es --out wiki20231101-es.zip
textsearch install wiki20231101-es.zip wiki-es
```

**What is exact, and what isn't** -- worth understanding before trusting a merged profile:

- **Vocabulary counts and weights are exact.** `occs`/`ndocs`/`trainsize`/`numtokens` are
  additive across disjoint batches, and the weighting scheme is *recomputed* from the
  merged counters -- so a merged profile's IDF is the true corpus-wide IDF, identical to
  what a single unbatched fit over the whole corpus would produce (the test suite asserts
  exactly this equality). This is the main reason to merge rather than to just pick one
  batch.
- **Synonyms are a rank-consensus fusion, not a recomputation.** Every input fit its own
  encoder, so its neighbor *distances* live in its own embedding space and aren't
  numerically comparable across inputs. What does transfer is the *ranking*, so the lists
  are combined with Reciprocal Rank Fusion: a neighbor several independently-fit batches
  all rank highly wins over one a single batch happened to like. The number kept beside
  each neighbor is the mean of the distances the contributing batches reported -- no longer
  a distance in any one space. `--synonyms-k 0` (default) keeps as many neighbors per token
  as the richest input had.
- **Lemmas are a plurality vote** over the inputs' clusterings. Independent votes can
  disagree in ways one clustering never does (`a => b` here, `b => a` there), so winning
  edges are followed to a fixed point and any cycle is resolved in favor of its most
  frequent member -- a merged lemma map is always acyclic and terminating.
- **Stopword candidates** are recomputed from the merged counters, then unioned with the
  inputs' recorded ones: a token every batch already removed is absent from the merged
  vocabulary and can't be re-derived, but it is still a stopword and stays recorded.

Inputs must share their normalization and tokenization settings and their weighting scheme;
transformations may differ *only* in their stopword set (the normal case when each batch
detected its own), which merges to the union. `EntropyWeighting` profiles cannot be merged,
since recomputing supervised weights would need the labeled corpus.

### `refit` -- adapt a bootstrap profile to a dataset

```
textsearch refit <base-profile> --sample PATH --out OUT.zip
                 [--format FMT] [--text-key KEY]
                 [--kappa N | --base-weight W] [--no-lemmas]
                 [--keep-rate T] [--keep-floor N] [--drop-distances] [--chunk N]
```

A profile fit from a large generic corpus is a **bootstrap** model: reasonable statistics
for a language, not a model for anyone's dataset. `refit` adapts one to a specific dataset
given a sample of it, and writes a new **self-contained** profile -- nothing in the output
refers back to the base, and it is typically smaller and more accurate for that dataset than
the generic profile it came from.

```sh
textsearch refit wiki-es --sample my-reviews.jsonl --out reviews-es.zip
textsearch install reviews-es.zip reviews-es
```

Statistics are **adjusted, not replaced**. The base acts as a prior worth `--kappa`
documents against the sample's evidence:

```
ndocs(t)  = ndocs_sample(t) + round(kappa * ndocs_base(t) / trainsize_base)
trainsize = trainsize_sample + kappa
```

`--kappa 0` (the default) uses the sample's own document count, weighting the two sides
equally; halve it for 1/3 base, double it for 2/3. `--base-weight 0.75` says the same thing
as a fraction. Expressing the base's authority *in documents* is what makes the output
sample-sized rather than base-sized, and what makes the knob mean something concrete.

Two consequences fall out of that arithmetic, and they are the point of the whole command:

- A word the base considers important but the sample never shows **keeps only its
  kappa-weighted share**, so it survives with reduced importance. Nothing special is done
  for it; lowering weight is just what the interpolation does.
- A word that mattered in neither is **dropped**: `--keep-rate`/`--keep-floor` decide, and
  anything whose blended count rounds below one document falls out regardless. `--keep-floor`
  is an absolute document count, so a single-document typo in a huge base corpus cannot clear
  a small rate threshold.

**Why counters and not weights.** BM25 never reads a model's precomputed weight vector -- it
derives its own IDF from `ndocs`/`trainsize` and normalizes by `avgdoclen`. Blending weights
alone would tune the tf-idf path and leave BM25 with the base corpus' numbers. Blending the
counters tunes both, and the weight vector is recomputed from them.

Both counters (`ndocs` and `occs`) are scaled by the same per-base-document denominator, which
is what keeps the output a *possible* corpus: scaling occurrences by the base's share of total
tokens instead looks equally reasonable but makes the two round against different denominators,
leaving carried tokens present in documents yet never occurring.

One consequence needs a decision: `avgdoclen` -- what BM25 divides document lengths by -- comes
out as a weighted mean of the two corpora's average lengths rather than the sample's. That is
honest, since the pseudo-documents the prior contributes are base documents, but it pulls
length normalization toward the base, and the pull is large when the two corpora are nothing
alike: Wikipedia-es against 400 product reviews lands at 141 tokens/document at the default
kappa and 56 at `--base-weight 0.2`, against the sample's own 9.2. `--avgdoclen sample` pins it
to the sample's instead (or pass a number); use it when the profile will index documents shaped
like the sample, which is the usual reason to refit. Only `numtokens` moves -- the counts the
weights come from are untouched -- because that field's single consumer is `avgdoclen` itself.

**Lemmas.** By default the base's lemma map is *applied* here -- chained into the new
profile's `TextConfig` -- and the base's own counters are folded through the same map so both
sides stay comparable. That folding is exact in `occs` and an over-estimate in `ndocs` (a
document containing two forms of one family counts twice), which is why every `ndocs` is
capped at `trainsize`; the counts are reported on stderr. Pass `--no-lemmas` to leave the map
carried but unapplied. This is why `fit` defaults to `apply = false`: whether to lemmatize
belongs to the model being tuned, not to the generic base.

A map inherited from the base says nothing about words the base never saw, so those stay
unmerged with their document frequency split across forms. `--extend-lemmas` recovers them
from **surface similarity alone** -- morphology is what actually groups an inflection family;
`fit` uses embeddings only to *split* one whose members mean different things -- so no
embedding is fit and the cost is one extra pass over the sample. Tune it with
`--morphology`/`--morphology-threshold`/`--qgram`/`--min-common-prefix`/`--lemma-selector`.
The grouping runs over the base and sample vocabularies merged, so a new form can elect the
established one (`"audifonos"` -> the base's `"audifono"`), and only the new tokens get
entries: the base's own clustering decisions are never overruled. Against Wikipedia-es it
recovered an accent pair (`devolvi` -> `devolvió`) and a diminutive (`baratito` -> `barato`),
and reviving one lemma pulled its whole base family back into the profile. The tradeoff is
that nothing can veto a grouping on meaning, so look-alike words with unrelated senses will
merge where a full `fit` would have kept them apart.

**What is not recomputed.** No embedding is fit here. Synonyms and lemmas come from the base,
with synonym entries pointing at pruned tokens removed. That is exactly what makes a refit
cheap next to a fit, and the point of bootstrapping. `--drop-distances` omits the synonym
distances from the output for the smallest possible profile, since only the ranking is used
on the normal query path.

The **applied** stopword set stays the base's -- it has to, since the base's counts were
collected under it and swapping it mid-blend would compare two incomparable vocabularies.
New candidates are recomputed from the blended counters and recorded for review.

The sample streams in batches of `--chunk` documents, so it can be far larger than memory.

**From a program**, rather than the CLI: `refit` is an operation of TextSearch itself, and
the library API is layered so any program can tune a base model when its sample arrives ---
`refit_textconfig(base; apply_lemmas)` gives the config the sample **must** be tokenized
under, `refit_profile(base, sample_voc)` takes a `Vocabulary` built however you like
(streamed, or grown over time with `update_voc!`), and `refit_profile(base, sample_docs)` is
the convenience form. `fold_lemmas` and `blend_vocabularies` are the pieces underneath.

## Tutorial

A complete walkthrough, from a raw corpus to a search hit, using a tiny 7-document
Spanish corpus about a garden. Every command below is copy-pasteable and was actually run
to write this section -- the output shown is real, not illustrative.

### 1. Write a corpus

```sh
cat > corpus.jsonl << 'EOF'
{"text": "la casa roja tiene un jardin grande"}
{"text": "la casa verde esta cerca del rio"}
{"text": "el jardin de la casa azul es pequeno"}
{"text": "una manzana roja y una pera verde"}
{"text": "la pera verde esta muy rica hoy"}
{"text": "la manzana verde esta rica tambien"}
{"text": "una hoja verde cayo en el jardin"}
EOF
```

### 2. Fit a profile

Write a config (this is exactly what `textsearch fit`'s `$EDITOR` template looks like,
with `stopwords.enabled` turned on so we can see it do something on such a small corpus):

```sh
cat > fit.toml << 'EOF'
[input]
format = "jsonl"
path = "corpus.jsonl"
text_key = "text"

[output]
dir = "./profiles"
prefix = "jardin"
batch_size = 10000

[normalization]
del_diac = true
del_dup = false
del_punc = false
group_num = true
group_url = true
group_usr = false
group_emo = false
lc = true

[tokenization]
nlist = [1]
mark_token_type = true

[stopwords]
enabled = true
doc_freq_threshold = 0.5

[encoder]
kind = "lsi"
outdim = 8
scaling = "none"
external_path = ""

[synonyms]
k = 3

[lemmas]
algorithm = "fft"
num_clusters = 0
selector = "shortest"
EOF

textsearch fit --config fit.toml
```

```
saved profile 1/1 (7 docs, vocsize=26) -> ./profiles/jardin-0001.zip
```

One batch was enough for 7 documents, so exactly one profile came out. `doc_freq_threshold
= 0.5` flagged "la" (in nearly every document) and "verde" (in 4/7) as stopword
candidates -- confirmed below.

### 3. Install it under a nickname

```sh
textsearch install ./profiles/jardin-0001.zip jardin
```

```
installed 'jardin' -> ~/.textsearch/profiles/jardin.zip
```

### 4. List and inspect it

```sh
textsearch list
```
```
jardin
```

```sh
textsearch info jardin
```
```
nickname:  jardin
path:      ~/.textsearch/profiles/jardin.zip
trainsize: 7
vocsize:   26
numtokens: 38
avgdoclen: 5.428571428571429
synonyms:  26 tokens
lemmas:    20 remapped tokens
stopword_candidates: 2 tokens
encoder:   lsi (scaling=none, source_path=, outdim=8)

TextConfig:
  ...
  transformation: IgnoreStopwords(Set(["verde", "la"]))
  ...
```

`stopwords.enabled = true` did exactly what it says: "la" and "verde" were detected as
candidates *and* wired into the profile's `IgnoreStopwords` transformation before the
vocabulary was built, so neither ever entered vectorization or LSI. The high `lemmas`
count (20 of 26 tokens remapped) is the small-corpus caveat from the Manual above in
action -- with only 7 short documents, LSI doesn't have enough signal to separate
meaningful token clusters, so clustering merges much more aggressively than it would on a
real corpus. The mechanics are correctly demonstrated either way.

### 5. Search it

```sh
textsearch search jardin "manzana pera" --collection corpus.jsonl --format jsonl
```
```
{"text":"una manzana roja y una pera verde"}
{"text":"la pera verde esta muy rica hoy"}
{"text":"la manzana verde esta rica tambien"}
```

Default `-t 1`: any document containing "manzana" *or* "pera" matches. Raise the
threshold to require both:

```sh
textsearch search jardin "manzana pera" --collection corpus.jsonl --format jsonl -t 2
```
```
{"text":"una manzana roja y una pera verde"}
```

Only the one document containing *both* words survives.

### 6. Uninstall (i.e., find out where it lives)

```sh
textsearch uninstall jardin
```
```
'jardin' is installed at:
~/.textsearch/profiles/jardin.zip
textsearch does not delete profile files automatically -- remove it yourself if you're sure, e.g.:
  rm '~/.textsearch/profiles/jardin.zip'
```

The file is still there -- `uninstall` only ever prints the path, per the Manual above.
