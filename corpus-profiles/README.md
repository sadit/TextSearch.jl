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
- **Sample instead of covering everything.** `--limit 200000` is usually plenty for stable
  corpus-wide vocabulary and IDF statistics.

## `--parts` is about memory and snapshots, not speed

Once the factorization stopped being cubic in batch size, batch size stopped mattering for
time. Measured on the same 100,000 Spanish articles, varying only how they were split:

| parts | batch size | total time | peak RSS | total .zip |
|---|---|---|---|---|
| 10 | 10,000 | 580s | 3.1 GB | 153 MB |
| 4 | 30,000 | 575s | 3.7 GB | 102 MB |
| 1 | 100,000 | **572s** | 5.1 GB | **59 MB** |

Within 1.4% -- so the driver takes `--parts N` (default 16) and derives the batch size from
the document count, because what the split actually controls is:

- **peak memory**, which grows with the part size, and
- **snapshots**: each part is written as soon as it is fitted, so an interrupted run keeps
  the parts it finished. `--resume` then skips those instead of refitting them (it still
  reads their documents, so later parts keep the same boundaries).

Fewer, larger parts are otherwise better: less duplicated vocabulary on disk (the 10-part
split stores overlapping vocabularies ten times over), better per-part statistics, and fewer
merges -- which matters because `merge` is exact only for counts and weights, while synonyms
and lemmas go through rank fusion and voting.

Extrapolating the measured throughput to all **1,841,155** Spanish articles: roughly **3
hours**, near enough independent of the split. That is an upper bound -- vocabulary per part
falls monotonically through the corpus (93,886 down to 54,944 across the ten parts above),
since the early articles are the long ones.

Stopword detection works out of the box: on the Spanish slice it flagged 55 candidates,
headed by `de . , en la el y del a un` -- real function words and punctuation, exactly what
`[stopwords] doc_freq_threshold` is supposed to catch. The synonym network is likewise good
without tuning: `rio` -> `confluencia, afluente, desembocadura, fluvial`, `futbol` -> `copa,
supercopa, balompie, clubes`.

## The stopword threshold is not scale-invariant

`doc_freq_threshold` is a document-frequency *ratio* relative to the batch, so the same number
means different things at different batch sizes. A bigger batch is more diverse, and every
token's ratio falls. Measured on Portuguese:

| | first 10k articles | 122,831 articles |
|---|---|---|
| `area` | 0.520 | 0.353 |
| `populacao` | 0.511 | 0.225 |
| `habitantes` | 0.454 | 0.299 |
| `historia` | 0.443 | 0.292 |
| `portuguesa` | 0.462 | 0.109 |

That is the whole trap: on a 10k probe those five sit in or beside the (0.4, 0.5] band, so the
probe argues for 0.5 to protect them, and at the size the fit actually runs not one of them is
close to the cutoff. **Calibrate at the production batch size**, which for a vocabulary-only
pass costs a couple of minutes against hours for the full fit -- stopword detection reads
document frequencies off the unpruned vocabulary, so `Vocabulary` alone sees exactly the input
the fit will see.

Read the (0.4, 0.5] band, since that band *is* the difference between the two candidate
values. At the production batch size:

| lang | band (0.4, 0.5] | threshold |
|---|---|---|
| pt | `ao das mais a sao ligacoes externas seu ou pela` (10) | 0.4 |
| en | `first one this has after are were be two his its new or` (13) | 0.4 |
| es | `pero ser son anos durante sin donde e historia otros ha ano cuando esta vease gran ya le uno asi nombre` (21) | 0.5 |

Spanish keeps 0.5 because four of its twenty-one are content words -- `historia`, `ano`,
`anos`, `nombre` -- and the error is asymmetric. Removing a token deletes it from the base
vocabulary, where no later `refit` can bring it back; keeping one costs almost nothing, because
idf already drives a high-document-frequency token's weight toward zero (measured on Arabic,
the most frequent token carried 0.64% of the maximum weight). The reason to remove function
words at all is cost, not relevance: they dominate `numtokens`/`avgdoclen`, which is what BM25
normalizes by, they spend the all-pairs synonym kNN budget, and they crowd the leading LSI
dimensions.

These values are the per-language defaults in `corpora/wikipedia.sh`;
`--doc-freq-threshold` overrides them. Languages outside the measured set default to 0.5, the
end that removes less, and that default should not be trusted as a measurement -- Arabic
flagged 19 candidates at 0.5 where Spanish flagged 46 on the same 10k, because its function
words are proclitics glued inside other tokens rather than separate tokens at all.

## `--min-chars` quietly drops a large slice of the corpus

The prepare step skips articles shorter than `--min-chars` (default 200). Verified against the
parquet row counts, that is not a rounding error:

| lang | parquet rows | JSONL records | dropped |
|---|---|---|---|
| es | 1,841,155 | 1,778,209 | 62,946 (3.4%) |
| en | 6,407,814 | 6,083,989 | 323,825 (5.1%) |
| pt | 1,112,246 | 982,641 | 129,605 (11.7%) |

Portuguese loses the most because pt.wikipedia carries an enormous number of very short
freguesia and municipality stubs. All 60 shards were present and byte-identical to what the
HuggingFace API reports in every case, so nothing is lost in the download or truncated in the
conversion -- this is entirely the filter.

Unlike `del_diac` or `min_ndocs`, this default was never measured or argued for. It matters
because it drops the *shortest* documents, which biases `avgdoclen` upward, and `avgdoclen` is
what BM25 divides every document length by: a profile fitted this way tells BM25 the average
document is longer than it is in the corpus being indexed. It also shifts stopword detection,
since a stub is lexically almost all function words, so their document frequency rises when
stubs are included. Pass `--min-chars 0` to keep everything.

## Paragraphs as documents make the stopword threshold stop mattering

`--split-paragraphs` emits one document per paragraph, prefixed with the article title, instead
of one per article. On 10,000 Spanish articles that is 272,466 documents, and it changes what
document frequency measures. Article frequency cannot tell a real stopword from a Wikipedia
artifact; paragraph frequency separates them by a factor of 55:

| token | df per article | df per paragraph | drop |
|---|---|---|---|
| `de` | 0.998 | 0.934 | /1.1 |
| `la` | 0.964 | 0.795 | /1.2 |
| `que` | 0.829 | 0.570 | /1.5 |
| `enlaces` | 0.826 | 0.017 | **/49** |
| `externos` | 0.822 | 0.016 | **/53** |
| `referencias` | 0.711 | 0.018 | **/39** |
| `vease` | 0.512 | 0.019 | **/27** |
| `anio` | 0.580 | 0.069 | /8.5 |
| `nombre` | 0.421 | 0.038 | /11 |

A function word is in nearly every paragraph; a section heading is in one paragraph of its
article, and a domain term in a handful. So the three populations land in disjoint bands, and a
fit under otherwise identical settings detects **11** stopwords -- `0 a de del el en la los que
se y`, function words and nothing else -- against **46** at article level that included
`enlaces externos referencias vease anio parte dos`. It also removes 31% of all token
occurrences rather than 43%, i.e. it keeps more content.

That is why the per-language thresholds above do not apply in this mode, and why the driver uses
0.1 instead: at paragraph level 0.5 flags 11 tokens, 0.3 flags 19 and 0.1 flags 31, and the
first 30 by frequency are all function words while the highest content word sits at 0.069. The
band is wide enough that the exact value stops being a tuning problem.

Synonyms improve too, since co-occurring in an 87-token paragraph is a much tighter semantic
window than co-occurring in a 2,300-token article, which pulls in terms that share a document
but not a meaning:

| token | per article | per paragraph |
|---|---|---|
| `planeta` | `larense protoplanetas haumea verrier` | `marte saturno neptuno urano jupiter` |
| `montana` | `downhill andinismo quebrantahuesos alimoche` | `orografia cumbres altitud macizos` |
| `iglesia` | `catolica fieles eclesiologia kirche` | `ortodoxa anglicana apostolica luterana` |
| `novela` | `policiaca spade ignatius minkoff` | `picaresca cuento narrativa autobiografica` |

(`quebrantahuesos` and `alimoche` are birds that appear in mountain articles; `larense` is a
Venezuelan demonym.) Not everything improves -- `futbol` loses the accented `futbol`, `medicina`
fills with inflections, `ciudad` picks up toponyms -- and lemmas come out essentially unchanged,
which is correct: `order=:morphology_first` groups by surface similarity, which does not depend
on how the corpus is cut into documents.

Costs: the fit is only ~33% slower for 27x the documents, since its cost is dominated by
vocabulary size (106,436 against 94,330) rather than document count, and the profile grows from
22 MB to 24 MB. What does grow is the index built from such a profile: 27x the documents.
`min_ndocs` also changes meaning, from "in 5 articles" to "in 5 paragraphs", which is a weaker
filter -- 106,436 tokens survive instead of 94,330.

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
similarity leaves: `abioticos`/`bioticos` and `abandonadas`/`donadas` share nearly every
bigram while being different words, and no choice of `qgram` separates them (the pairs score
*identically* to correct ones like `aberturas`/`abertura` at every `q`, because both are "one
character different at one end").

Two more choices only showed their teeth at full scale, and both are worth knowing before
tuning this:

**Grouping is leader-based, not single-linkage.** Transitive linking chains -- `A~B` and
`B~C` merge even when `A` and `C` are unrelated -- and on 143k Spanish tokens the chains
swallowed whole prefix neighbourhoods: a 292-member "family" spanning `concentra`...`cons`,
with `cara` merged into `caracas` and `caracalla`. 324 families exceeded 20 members, holding
17% of all mappings. Attaching members to a *seed* instead bounds each group by one radius
around its lemma: the largest family drops from 292 to 26 and only 4 exceed 20 members.

**The selector seeds the grouping, so it decides more than ties.** With `shortest`, a short
misspelling wins the seed and fragments the family around it -- the typo `guera` seeded a
group that absorbed `guerra` and left `guerras` stranded. `most_frequent` (the default) seeds
on the form the corpus actually uses, which recovered `guerras -> guerra`,
`jugadores -> jugador` and `concentraciones -> concentracion` in the same run.

With all of it in place, at 25,000 Spanish articles: ~32% of the vocabulary remapped, largest
family 26, and output like `himnos -> himno`, `murallas -> muralla`, `compresores -> compresor`,
`examinaron -> examinar`, `biologicas -> biologica`.

One caveat worth knowing: the elected lemma is the most frequent member of its family, not a
linguistically-derived base form, so it can point "backwards". For search normalization what
matters is that every variant collapses onto the *same* key, which it does.

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
