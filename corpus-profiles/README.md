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

`fit`'s dominant cost is the **synonym network**, an exact all-pairs kNN over the
vocabulary -- O(vocabulary²). That makes vocabulary size, not corpus size, the thing that
decides whether a run finishes today or next month. Measured on this repo (64 threads):

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

Even so, **mind the total cost before launching a full language.** One 25,000-article
Spanish batch at `--min-ndocs 5` (≈143k tokens) ran for **over 50 minutes** without
finishing on 64 threads, so 74 of them is measured in days, not hours. Practical options:

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
supposed to catch.

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
