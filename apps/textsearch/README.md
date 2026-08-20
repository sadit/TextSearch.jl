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
```

`profile` is an installed nickname or a path to a profile `.zip`/directory. Prints every
record of `--collection` whose text shares at least `--threshold` tokens (after the
profile's normalization/tokenization) with `query`, as one JSONL line per hit:

```sh
textsearch search mynick "red car" --collection reviews.jsonl -t 2
```

`-t`/`--threshold` (default `1`) is the minimum token-set intersection size -- `1` matches
if *any* query token is shared (union), raising it toward the query's own token count
gives progressively stricter, AND-like matching (the same `t`-threshold idea
`SimilaritySearch.InvertedFiles` uses). There is no ranking or scoring; matches print in
corpus-encounter order, as found, exactly like `grep`.

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
