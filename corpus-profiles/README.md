# corpus-profiles

Recipes for building **distributable TextSearch.jl profiles** -- per-language/per-domain
bundles of vocabulary, weights, synonym network, lemmas, and stopword candidates, as
produced by [`textsearch fit`](../apps/textsearch/README.md) and consumed by
`TextSearch.load_profile`.

## What lives here, and what doesn't

**Tracked in git:** only the *recipes* -- the scripts that fetch/prepare a corpus and the
`fit` config TOMLs that turn it into a profile.

**Not tracked in git:** every byte of data. Downloaded corpora and the profile `.zip`s we
fit from them are large binary artifacts; they are published as **GitHub release
attachments**, and only once we've verified a profile is actually correct. Nothing is
attached to a release straight out of the generator.

`.gitignore` here enforces that as an *allowlist*: everything is ignored by default and
only `*.jl`/`*.sh`/`*.toml`/`README.md` are re-included. A new corpus format or output
layout therefore can't slip into a commit just because nobody remembered to add its
extension to an exclusion list. If you add a genuinely new kind of tracked recipe file,
extend the allowlist in `.gitignore` rather than force-adding the file.

Because the recipes are versioned and the data isn't, any published profile should be
reproducible from this directory alone.

## Layout

```
corpus-profiles/
├── lib/
│   ├── common.sh              shared: paths, textsearch CLI lookup, fit-config rendering
│   └── parquet_to_jsonl.jl    generic streaming parquet -> JSONL ("text" key)
├── corpora/
│   └── wikipedia.sh           driver: HuggingFace wikimedia/wikipedia -> profiles
├── raw/        (ignored)      downloaded corpus shards
├── work/       (ignored)      intermediate JSONL + rendered fit configs
└── profiles/   (ignored)      fitted profile .zip files
```

Adding another corpus (Common Crawl, a news dump, ...) means adding one driver under
`corpora/` that sources `lib/common.sh`, does its own *fetch* and *prepare* (raw → JSONL
with a `text` field), then calls the shared `ts_render_fit_config` + `ts_fit`. Everything
downstream of "a JSONL file with a text field" is already corpus-agnostic, and
`lib/parquet_to_jsonl.jl` is reusable for any parquet-based corpus.

## Wikipedia

```sh
corpora/wikipedia.sh --lang es                    # newest snapshot, all articles
corpora/wikipedia.sh --lang es --limit 25000       # one batch's worth (smoke test)
corpora/wikipedia.sh --lang de --snapshot 20231101
corpora/wikipedia.sh --lang es --steps fetch       # download only
corpora/wikipedia.sh --help                        # all options
```

Source: the [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
dataset, one parquet config per (snapshot, language) with `id`/`url`/`title`/`text`
columns, one row per article, wiki markup already stripped. Snapshots are auto-discovered
from the HuggingFace API, so `--snapshot` is only needed to pin an older dump. Downloads
are plain `curl` (resumable, size-verified) -- no `huggingface_hub`/`datasets` install
required. Output profile name is `wiki<SNAPSHOT>-<LANG>`, e.g. `wiki20231101-es`.

As of this writing the only published snapshot is **20231101** (323 languages); Spanish is
**1,841,155 articles** across 13 parquet shards, 3.49 GB.

## Cost: read this before launching a full language

`fit`'s heaviest step is the **synonym network**, an all-pairs kNN over the vocabulary.
Done exactly it is O(vocabulary²), which makes vocabulary size -- not corpus size -- the
thing that decides whether a run finishes today or next month.

By default (`[synonyms] approx = "auto"`) vocabularies past a few thousand tokens use an
autotuned approximate `SearchGraph` instead: construction tuned to `MinRecall(0.97)`,
search parameters to `MinRecall(0.9)`. Measured against the exact answer on a real 37,388-
token vocabulary (64 threads), that is **4× faster at essentially no cost where it
matters**:

| rank | recall of the exact neighbors |
|---|---|
| nearest neighbor (rank 1) | **0.969** |
| top 3 | 0.949 |
| top 8 | 0.855 |

i.e. the closest synonyms -- the ones a synonym network is actually for -- are almost all
recovered, and only the far tail of the list degrades. Raising `construction_recall` to
0.999 does not improve this (it saturates), so 0.97 is the default. Since exact search
grows quadratically while the graph does not, the speedup widens with vocabulary size.

Even so, vocabulary size still drives everything. Measured:

| corpus slice | articles | vocabulary | notes |
|---|---|---|---|
| `ab` (Abkhazian), whole | 1,009 | 37,381 | whole pipeline ≈ 1 min |
| `es`, 25,000 articles | 25,000 | 611,797 | one `--batch-size 25000` batch |

Two things fall out of that Spanish measurement:

**1. Most of the vocabulary is noise.** Of those 611,797 tokens, **53% occur in exactly
one article** (typos, IDs, foreign words, stray markup) and only 9.6% occur in 20+. Those
tokens cannot have a meaningful embedding, so they contribute nothing but quadratic cost
and junk synonyms. Hence `--min-ndocs` (config: `[vocabulary] min_ndocs`), which prunes
them *before* the encoder runs:

| `min_ndocs` | Spanish vocabulary (25k articles) | share kept | relative synonym cost |
|---|---|---|---|
| 1 (no pruning) | 611,742 | 100% | 1× |
| 2 | 285,903 | 46.7% | ~0.22× |
| 3 | 204,858 | 33.5% | ~0.11× |
| 5 *(driver default)* | 143,066 | 23.4% | ~0.055× |
| 10 | 90,815 | 14.8% | ~0.022× |
| 20 | 58,767 | 9.6% | ~0.009× |

The default is `--min-ndocs 5`. Raise it for a faster/cleaner profile, drop it to `1` only
if you specifically want the full long tail and are prepared to wait.

**2. Batching a whole language produces many profiles; `merge` folds them back.** Spanish
at `--batch-size 25000` is 74 batches, i.e. 74 *independent* profiles, each with its own
vocabulary and IDF over its own 25k articles. `textsearch merge` combines them into the
single `wiki20231101-es` profile you actually want, and does so **exactly** for the parts
that matter most -- the merged vocabulary counts and IDF weights are identical to what one
unbatched fit over the whole corpus would produce (synonyms are fused by rank consensus and
lemmas by plurality vote; see the [app README](../apps/textsearch/README.md#merge----fold-batched-profiles-into-one)):

```sh
corpora/wikipedia.sh --lang es
textsearch merge profiles/wiki20231101-es --out wiki20231101-es.zip
```

The other half of the cost is the **LSI factorization**, and it is driven by batch size
rather than vocabulary. The exact dense path (`[encoder] factorization = "full"`) builds an
`n×n` Gram matrix and takes a *complete* eigendecomposition, computing every eigenpair to
keep `outdim` of them -- `O(n^3)` time and an `n^2` allocation. ARPACK's Lanczos iteration
(`"lanczos"`) is equally exact but never forms that matrix, and `"auto"` switches to it past
a few thousand documents per batch. Measured on Spanish Wikipedia slices, factorization only,
same prebuilt matrix, 64 threads:

| documents in batch | `full` | `lanczos` |
|---|---|---|
| 2,000 | **4.6s** | 14.8s |
| 4,000 | 13.2s | **9.5s** |
| 8,000 | 48.8s | **11.5s** |

`full` grows steeply while `lanczos` stays roughly flat (its cost follows the number of
nonzeros, not `n^3`), which is why the crossover sits low and why `lanczos` is what makes
large batches viable at all.

**Still, mind the total cost before launching a full language.** Before these two changes,
one 25,000-article Spanish batch at `--min-ndocs 5` (≈143k tokens) ran over 50 minutes
without finishing with exact synonym search, and over 30 minutes stuck in the dense
factorization. Practical options:

- **Prune harder.** `--min-ndocs 20` cuts the vocabulary to ~59k, roughly 6× less synonym
  work than `min_ndocs=5` and ~110× less than no pruning, at the cost of the rare-token
  tail. This is the first knob to reach for.
- **Sample instead of covering everything.** `--limit 200000 --batch-size 25000` gives 8
  batches to merge rather than 74 -- usually plenty for stable corpus-wide vocabulary and
  IDF statistics.
- **Cover everything anyway**, accepting a multi-day run, then merge once at the end.

Stopword detection, for the record, works well at this scale without any tuning: on the
Spanish slice it flagged 55 candidates, headed by `de . , en la el y del a un` -- real
function words and punctuation, exactly what `[stopwords] doc_freq_threshold = 0.5` is
supposed to catch. The synonym network is likewise good out of the box: `rio` ->
`confluencia, afluente, desembocadura, fluvial`, `futbol` -> `copa, supercopa, balompie,
clubes`.

## Lemmas need morphology, not just embeddings

The `[lemmas]` step deserves a note, because the obvious version of it does not work. LSI
embeddings encode *distributional* similarity, so a token's neighbours are its topical
relatives, not its inflected forms -- `guerra` sits next to `belico` and `aliados`, which is
what the synonym network is for. Electing one representative per semantic cluster therefore
produced, measured on Spanish Wikipedia, mappings like `casas -> dia`, `ciudades -> 市`,
`mujeres -> %`: 99.7% of the vocabulary remapped with only 0.3% of pairs even sharing a
prefix. More clusters did not help -- shrinking them to an average of 1.5 tokens still only
reached 2%.

So each semantic cluster is split again by *surface* similarity before a lemma is elected.
Measured on 2,000 Spanish articles (39,407 tokens), share of mappings where one token is a
prefix of the other -- a conservative proxy, since it cannot see correct gender/number pairs
like `abrupto`/`abrupta`:

| morphology | threshold | vocabulary remapped | prefix-related | time |
|---|---|---|---|---|
| `none` | -- | 99.5% | 0.5% | 3.1s |
| `jaccard` | 0.3 | 7.1% | **70.7%** | 0.7s |
| `jaccard` | 0.4 | 10.3% | 50.0% | 0.7s |
| `levenshtein` | 0.2 | 6.7% | 56.3% | 3.1s |
| `levenshtein` | 0.5 | 74.2% | 1.1% | 3.0s |

Jaccard over character bigrams wins on both quality and speed, so it is the default at
`0.3`. `min_common_prefix = 3` then removes the failure mode that position-blind n-gram
similarity has left: `abioticos`/`bioticos` and `abandonadas`/`donadas` share nearly every
bigram while being different words. With all of it in place the output looks like a
lemmatizer -- `aberturas -> abertura`, `acabaran -> acabar`, `absolutas -> absoluto`,
`algebraico -> algebraica` -- over ~6.6% of the vocabulary.

One caveat worth knowing: the elected lemma is the shortest (or most frequent) member of its
family, not a linguistically-derived base form, so it can point "backwards"
(`aborigen -> aborigenes`). For search normalization what matters is that every variant
collapses onto the *same* key, which it does.

## Workflow

1. **Write a recipe** -- a driver under `corpora/` (see `wikipedia.sh`) that obtains the
   corpus and converts it to JSONL, plus the fit parameters it should use.
2. **Generate** the profile(s): `corpora/<corpus>.sh --lang ...`. Start with `--limit` to
   smoke-test the whole path cheaply before committing real compute.
3. **Verify** before publishing anything. At minimum: `textsearch info` on the result (does
   `vocsize`/`trainsize` look sane? are the stopword candidates actually stopwords for that
   language? do the synonyms and lemmas look linguistically plausible rather than like
   clustering noise?), plus a few `textsearch search` queries against a held-out slice.
4. **Publish** the verified `.zip`s as attachments on a GitHub release, so users can
   `textsearch install <downloaded>.zip <nickname>` without refitting anything.
