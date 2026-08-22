# 2026-08-21/22 — paragraph units, `numtokens`, and what query expansion is for

Branch: `paragraph-documents`. Started as "refit the es/pt/en Wikipedia profiles", became a
correctness pass over BM25's length normalization and a long investigation into why the
query-expansion network contains what it contains.

---

## 1. Bugs found, with their measured impact

### `numtokens` counted distinct tokens, so `avgdoclen` was not a length

`Vocabulary` accumulated `batch_numtokens += length(bow)`, and `bow` is filled as a **set**
(`bow[id] = 1` in `_locked_tokenize_and_push`) because its only job is to say which tokens a
document contained, for `ndocs`. So `numtokens` was the sum of per-document *distinct* counts and
`avgdoclen = numtokens / trainsize` was the mean number of distinct tokens — not the "average
document length in tokens" both docstrings promised.

BM25 mixes the two notions in one formula: an index measures each document's length as its total
occurrences (`bm25_register_postings!` sums frequencies) and `BM25Scorer(voc)` divides by
`avgdoclen(voc)`.

    den = tf + k1(1-b) + doclen * k1*b/avgdoclen

Measured on 300 Spanish Wikipedia articles: `avgdoclen` read **863** against indexed lengths
averaging **2992**, so `doclen / avgdoclen` was **3.47** for an average document where it must be
1.0. The length normalization behaved as if **b = 2.6**, outside BM25's valid [0, 1], heavily
over-penalizing long documents. Every profile built before this carried it.

The first fix attempt, `sum(values(bow))`, was a **no-op** — since bow holds only 1s, summing its
values is exactly `length(bow)`. The new test caught it. The real count comes from where the
tokens are walked, so `_locked_tokenize_and_push` now returns how many it pushed.

No test covered `numtokens` or `avgdoclen`, which is why it survived: tokenization was never
wrong, only the accounting over it. Two tests now pin it, the important one being
`sum(idx.doclens) == getnumtokens(voc)` and `mean(idx.doclens) == avgdoclen(voc)`, which ties the
two notions of length together so they cannot drift apart again.

### `merge_profiles` kept stopwords with a fraction of their counts

`fit` applies stopwords by tokenizing under `IgnoreStopwords`, so a flagged token never enters that
batch's vocabulary. Merging unioned the sets but never touched the vocabulary, so a token flagged
in some batches and not others arrived with the counts of only the batches that kept it. On
Portuguese Wikipedia that was **18 of 35** merged stopwords, with `como` at df **0.049** against a
true corpus df above 0.5 — weight 4.35 where 1.33 is right, i.e. **3.3x** overweighted: a function
word promoted to a rare discriminative term while the profile's own artifact called it a stopword.

Dropping such a token deletes content words — 35 of English's 89 were flagged by at most 2 of 48
batches, `american`, `united`, `states`, `family`, `history` among them, so "united states history"
would have lost every term. Keeping the partial counts is the inflated-idf bug itself. So the
missing counts are **imputed**: a batch that removed a token recorded no number but did record that
its document frequency there exceeded that batch's threshold, making `threshold * trainsize` a real
lower bound and the tightest available. Only batches whose vocabulary genuinely lacks the token are
imputed for — a profile may *list* a stopword it never applied, and its counts are then exact, so
imputing on top would double-count. `fit` now records its `doc_freq_threshold` in the lineage.

Result on the real profiles: pt 35 -> 17 stopwords, es 39 -> 16, en 89 -> 19, all function words,
with `historia` at 2.020 and `nombre` at 2.299 rather than deleted.

### `merge_profiles` discarded the inputs' lineage

`istuned` reads nothing else, so merging refitted profiles produced a profile reporting itself as a
base. Each distinct input stage is now kept once with the number of inputs that contributed it.

### `--min-chars 200` silently dropped 3–12% of every corpus

Verified against parquet row counts: 62,946 Spanish articles (3.4%), 323,825 English (5.1%),
129,605 Portuguese (11.7%) — pt.wikipedia is full of very short freguesia stubs. All 60 shards were
present and byte-identical to the HuggingFace API, so the loss was entirely the filter. Unlike
`del_diac` or `min_ndocs` this default was never measured or argued for. Its bias is upward on
`avgdoclen`, since it drops the *shortest* documents; measured in tokens the discarded records
average 22 tokens and the bias is **+17.7% for pt**, +1.9% en, +0.8% es on the first shard.

---

## 2. Decisions taken

### `language` goes in `TextConfig` (policy), inert for now

es and pt Wikipedia profiles have byte-identical normalization and tokenization, so `merge` folded
them into one 129,940-token vocabulary of neither language without complaint. `language` sits in
policy because the language *selects* the policy — it is the same kind of fact as `del_diac`, and
hand-authorable without a corpus. A *detected* distribution would be a corpus-derived artifact and
would belong in the lineage. It changes nothing about tokenization; the one behaviour it buys is
`merge` refusing a declared mismatch, with `:unknown` permissive.

### The stopword threshold must be calibrated at the scale it is applied

`doc_freq_threshold` is a document-frequency **ratio relative to the batch**, and that ratio is not
scale-invariant. On Portuguese, `area` sits at 0.520 over the first 10k articles and 0.353 over the
122,831 a real part holds; `populacao` 0.511 -> 0.225, `portuguesa` 0.462 -> 0.109. A 10k probe
therefore argues for protecting words that are nowhere near the cutoff at production size, and the
per-language conclusions inverted when re-measured: es 0.5, pt 0.4, en 0.4, the mirror image of the
probe's answer.

Read the (0.4, 0.5] band, since that band *is* the difference between the two candidate values.
Spanish keeps 0.5 because four of its twenty-one are content (`historia`, `ano`, `anos`, `nombre`)
and the error is asymmetric: removing a token deletes it from the base vocabulary where no later
`refit` can recover it, while keeping one costs almost nothing since idf already drives a
high-document-frequency token's weight toward zero.

### Paragraphs as documents

`--split-paragraphs` emits one document per paragraph, prefixed with the article title. On 10,000
Spanish articles that is 272,466 documents (36 blocks per article, median 18, p90 90).

The point is that it changes what document frequency measures. Article frequency cannot separate a
real stopword from a Wikipedia artifact; paragraph frequency separates them by a factor of 55:

| token | df/article | df/paragraph | drop |
|---|---|---|---|
| `de` | 0.998 | 0.934 | /1.1 |
| `la` | 0.964 | 0.795 | /1.2 |
| `que` | 0.829 | 0.570 | /1.5 |
| `enlaces` | 0.826 | 0.017 | **/49** |
| `externos` | 0.822 | 0.016 | **/53** |
| `referencias` | 0.711 | 0.018 | **/39** |
| `vease` | 0.512 | 0.019 | **/27** |
| `ano` | 0.580 | 0.069 | /8.5 |
| `nombre` | 0.421 | 0.038 | /11 |

A function word is in nearly every paragraph; a section heading is in one paragraph of its article;
a domain term in a handful. A fit under otherwise identical settings detects **11** stopwords
(`0 a de del el en la los que se y`, function words and nothing else) against **46** at article
level that included `enlaces externos referencias vease ano parte dos`, and removes 31% of all
token occurrences rather than 43%.

**The practical payoff is that the threshold stops being a tuning problem.** At paragraph level 0.5
flags 11, 0.3 flags 19, 0.1 flags 31, and the first 30 by frequency are all function words while
the highest content word sits at 0.069. Paragraph mode therefore uses 0.1, and the per-language
article-level values do not apply to it.

Query expansion improves too, since co-occurring in an 87-token paragraph is a much tighter
semantic window than in a 2,300-token article:

| token | per article | per paragraph |
|---|---|---|
| `planeta` | `larense protoplanetas haumea verrier` | `marte saturno neptuno urano jupiter` |
| `montana` | `downhill andinismo quebrantahuesos alimoche` | `orografia cumbres altitud macizos` |
| `iglesia` | `catolica fieles eclesiologia kirche` | `ortodoxa anglicana apostolica luterana` |
| `novela` | `policiaca spade ignatius minkoff` | `picaresca cuento narrativa autobiografica` |

(`quebrantahuesos` and `alimoche` are birds that appear in mountain articles; `larense` is a
Venezuelan demonym.) Not everything improves: `futbol` loses the accented `futbol`, `medicina`
fills with inflections, `ciudad` picks up toponyms.

Lemmas come out **essentially identical**, which is correct by design and worth confirming:
`order=:morphology_first` groups by surface similarity over the whole vocabulary and lets
embeddings only split a family, and surface similarity does not depend on document granularity.

Headings are folded **forward** into the paragraph they introduce rather than dropped. Wikipedia
already glues a heading to its section's first paragraph when prose follows, but a heading over a
list or table lands as its own block — 9.9% of all blocks are a single word. Folding cut the
discards from 66,363 to 1,438.

Costs: the fit is only ~33% slower for 27x the documents, since its cost is dominated by vocabulary
size (106,436 against 94,330) rather than document count. What grows is the index built from such a
profile. `min_ndocs` also changes meaning, from "in 5 articles" to "in 5 paragraphs" — accepted as
close enough for profile construction.

### The artifact is `query_expansion`, not `synonyms`

Renamed across library, app, on-disk format, tests, docs and corpus scripts. The name was not
cosmetic: calling it a synonym network invited judging its entries as substitutable words, and a
long stretch of this session went into filtering it on that basis before the objective was
restated. The artifact exists to **enrich a query**, and topically related terms are what serve
that.

The rename also exposed documentation that had been false since the profile refactor:
`src/bm25/invfile.jl` still told readers that expansion requires
`voc.textconfig.expand_query_synonyms = true`, a field that no longer exists.

### `_PROFILE_FORMAT_VERSION` stays at `1.0`

There is no release and no profile outside this repository, so the version is not yet a
compatibility mechanism and bumping it on every layout change buys nothing. `load_profile` still
refuses a mismatch, so the field is ready from the first release onward.

This corrected a test that had **silently stopped testing**: "an older format version is refused"
faked an old version by writing `"1.0"`, which became the current version — the profile would be
valid and nothing would be refused. It now writes `"0.9"`.

### `k` stays at 8

Measured to 32, neighbours 9–32 are real content (`medicina` gains `urgencias obstetricia
psiquiatria cardiologia odontologia`, `planeta` gains `fobos ceres deimos caronte sedna`), so the
inflection crowding at k=8 was a truncation artifact and not an embedding one. It costs 4x the
expansion storage — on the pt profile, 21 MB of a 49 MB zip becomes ~85 MB — and the profiles are
meant to ship as release attachments, so this is a distribution decision rather than a quality one.

---

## 3. Dead ends, all measured

The failing population is a token whose expansion entries are incoherent: `comun` returning
`aguilucho pintojo arrendajo cerceta` (bird names, from "nombre comun" in species articles),
`bojan` returning `cahill musica carnatica`. The healthy comparisons are `innodb` -> `mysql oracle
sqlite mariadb interbase` and `planeta` -> `marte saturno neptuno urano jupiter`.

1. **Absolute distance ceiling.** No global scale exists: `innodb`, whose list is perfect, has the
   farthest first neighbour of every token examined (0.501) — farther than `boca` (0.387), `comun`
   (0.426) and `seccion` (0.465), whose lists are junk.
2. **Knee / second difference of the sorted distance curve.** There is no knee: `planeta` runs
   0.037 -> 0.179 over 24 neighbours in smooth increments, and its largest curvature falls at
   position 9, in the middle of the correct list. In `innodb` the real good/junk boundary sits
   between `interbase` (0.572) and `sidereo` (0.579), a gap of 0.007 — indistinguishable from the
   0.005 gaps inside the good part.
3. **Local radius from reverse votes** (`bichromatic_metricjoin`, kept as `prune=:localradius`).
   Correctly empties the polysemous fillers but also empties `innodb`, `cistoscopia` and
   `aluminosilicato`, because the estimate needs the token to be in someone else's top-k: a rare
   term close to `mysql` gets no voters since `mysql` has closer neighbours. It hits precisely the
   rare-but-coherent technical vocabulary the network is most useful for.
4. **Reciprocity, document-frequency floor, LSI norm.** Reciprocity measures frequency, not
   coherence: `forma` and `tiene` (junk) score 1.00 while `innodb` and `destacamento` (good) score
   0.00. Document frequency does not separate at the bottom either — at 5–8 documents both
   `innodb`/`ruderal` (good) and `bojan`/`declararan` (bad) live together. The raw LSI norm over
   sqrt(ndocs) separates by ~2x in the mid and high bands (`guardia` 0.00215 vs `boca` 0.00099) and
   not at all in the rare band (good 0.00040–0.00085 against bad 0.00049–0.00056), i.e. it works
   where it is not needed.
5. **Typing by syntactic class**, induced from the function word preceding each token — no tagger,
   one bigram pass. Works on the target case (`comun`/`popular` 0.86 against `comun`/`aguilucho`
   0.03) but does **not** implement "nouns with nouns": the profile separates common from proper
   nouns, since `planeta` follows a determiner (0.96) and `marte` a preposition (0.91) as Spanish
   proper nouns take no article, putting `planeta`/`marte` at 0.058 — *below* the 0.14 that must be
   cut to remove `comun`/`aguilucho`. Coarsening the anchors into groups trades one failure for
   another: `tiene`/`hizo` improves 0.53 -> 0.92 and `nuevo`/`ciudad` breaks 0.01 -> 0.97. Coverage
   bounds it further: 40,132 of 106,436 vocabulary tokens have 10+ anchor observations.
6. **Asymmetric: incompatible profiles AND a much rarer neighbour.** This one survives the joint
   distribution — at (cosine < 0.1, ndocs ratio > 50) it cuts 1,182 pairs, 0.34% of profiled pairs,
   and the margin is a factor of 34 between the worst good pair (`planeta`/`marte`, ratio 3.5) and
   the best junk pair (`comun`/`mirlo`, 277). It dies on the backfill measurement: only 813 of
   106,447 tokens lose any neighbour, and reading their full lists shows it removing **good** pairs
   — `se` -> `encuentra encuentran trata denomina observa llama`, `sus` -> `respectivos propios
   respectivas propias`, `fue` -> `exonerado rechazada trasladado absuelto`. A set of past
   participles is exactly right for a passive auxiliary. The freed slots then take `melamina`,
   `bechstein`, `aimee`, `teropodo`, `xanadu`.
7. **PPMI over a full context window**, to measure substitutability instead of topical relatedness.
   Fails: `comun`/`popular` 0.050 against `comun`/`aguilucho` 0.037 — no separation — and
   `rio`/`jardim` 0.214, the highest of all. The anchor-profile version (5) worked **because** it
   was restricted to 38 function words, all frequent; opening the window lets content contexts back
   in and the topical signal returns. PMI also over-rewards rare pairs by construction: measured on
   this corpus, `comun`/`aguilucho` co-occur **twice** in 23M positions and score PMI 13.38, above
   `muy` (10.72), `nombre` (10.44), `mas` (10.23) and `es` (10.02), which are the contexts that
   actually characterize `comun`. PMI puts the noise above the signal.
8. **Second-order similarity** (overlap of the two tokens' own neighbour sets) is *inverted* for the
   target case: `comun`/`aguilucho` 0.524 against `comun`/`popular` 0.000. Obvious in retrospect —
   it derives from the same network, so it amplifies its topical structure rather than correcting it.

**The common cause of 1–4** is that in 256 dimensions under cosine the distances concentrate: the
ranking carries information, the magnitudes and their differences do not. **The common cause of
7–8** is that any measure derived from document co-occurrence, or from a full window, carries the
topical signal. What actually separates `innodb` from `bojan` is whether the token's few documents
are about the same thing — context coherence, which no statistic already on hand encodes.

---

## 4. The reformulation that dissolved the problem

Stated by the user after the eight attempts above: **the objective is query enrichment, not
synonyms.** Highly popular words do not need expansion; mid-frequency words maybe; the Zipf tail
does, and it is the majority of the vocabulary. And a tail word should not be connected to a very
popular one — only to something mid, or to other tail terms.

Measured on the paragraph-level network (851,488 pairs; head = df > 0.01, 594 tokens; mid 5,953;
tail 99,889):

| source | -> head | -> mid | -> tail | total |
|---|---|---|---|---|
| **head** | 522 | 1,483 | 2,747 | **4,752 (0.6%)** |
| **mid** | 2,267 | 18,063 | 27,294 | 47,624 (5.6%) |
| **tail** | **27,733 (3.3%)** | 126,445 (14.8%) | 644,934 (75.7%) | 799,112 |

Two things fall out.

**Everything the eight heuristics were fighting over is 0.6% of the network.** `comun`, `forma`,
`tiene`, `se`, `fue`, `por`, `los` — 4,752 pairs. All of them have a head word as the *source*, and
under the reformulation those pairs should not exist at all: not because the association is wrong,
but because a head word does not need expanding. The entire junk population disappears by
construction, using only the `df` already in the vocabulary.

**The harmful case is 6x larger and was never examined**: 27,733 tail -> head pairs. `snezhko`
(weight 15.60) -> `por` (df 0.431); `ssr` -> `union`; `otomo` -> `pelicula`. Non-Latin-script tokens
at maximum idf pointing at common Spanish words. Since `expand_query!` appends
`weight * weight_fn(rank)` — the weight of the token that produced the expansion — such a pair
injects a term with no discriminating power **at high weight**, which is worse than anything the
filters were chasing.

And several of those pairs are *semantically correct*: `pente` -> `griego`, `theou` -> `dios`,
`koine` -> `lengua`. They are right and useless, which is the sharpest statement of why "good
synonym" and "useful expansion" are different criteria.

Sweeping the head cut: at 0.05 the head is 57 tokens — the function words plus `anos ano parte forma
ciudad ser ha donde` — and removing them as sources costs 456 pairs (0.1%). Dropping to 0.02 starts
taking `guerra sistema espana mundo`, and 0.01 takes `musica universidad familia imperio`, which do
benefit from expansion. For the **target** side the criterion is relative rather than absolute, and
one ratio knob covers it: `planeta` -> `marte` moves toward something rarer (0.29), `innodb` ->
`mysql` roughly level, `snezhko` -> `por` jumps 20,000x. Known cost of the ratio rule: it also cuts
the legitimate rare -> common direction, such as a misspelling pointing at the correct word.

**Implemented** as `head_df` and `max_target_ratio` on `query_expansion`, exposed through the fit
config and the wikipedia driver. It is a filter of *purpose*, not of quality, which is why it works
where eight quality filters failed.

The defaults encode the lesson this session learned twice. `max_target_ratio` carries a real
default (50) because it is a ratio of two frequencies within one corpus and therefore
scale-invariant -- it means the same thing for articles and for paragraphs. `head_df` carries
**no** default, because it is a document frequency, i.e. a ratio relative to whatever a document
*is*: 0.05 means "one paragraph in twenty" for a paragraph profile and something else entirely for
an article profile. Whoever knows the unit sets it; the driver sets 0.05 for paragraph runs and
leaves it off for article runs, where it was never measured.

Two incidental notes from implementing it. `getndocs` was not imported into `module LSI`, which
Julia reports only at run time from inside the closure -- the same submodule trap hit earlier with
BM25/LSI/RI. And the first version of the test picked a head token present in *every* document,
which the pre-existing NaN guard already empties: a token with near-uniform document frequency gets
an all-zero LSI embedding, so it has no neighbours and is nobody's neighbour. The test was
exercising that guard rather than the new filter while passing.

---

## 5. On the alternatives (fastText, GloVe, word2vec)

Qualitative, in the terms that matter here.

**Cost.** A single truncated SVD (ARPACK) against many SGD epochs, and deterministic — two runs
produce the same file, which matters for an artifact meant to be published.

**Precision for substitutability.** word2vec or fastText with a small window would beat
document-level LSI, but *because of the matrix, not the algorithm*: skip-gram with negative sampling
implicitly factorizes shifted PMI, and count-based PPMI-SVD matches it with tuned hyperparameters.
The gain is in moving from document co-occurrence to window co-occurrence, which is reachable with
the machinery already here.

**Training data.** SVD methods need less, since they use exact global counts rather than sampled
updates. word2vec and GloVe typically want hundreds of millions of tokens; a 122k-document batch is
nowhere near that.

**The decisive architectural point.** These profiles **merge** and **refit**. A neural embedding
does neither: `merge_profiles` fuses networks by rank consensus precisely because each batch lives
in its own space, and with word2vec that worsens — random initialization, non-deterministic.
Counts are additive; trained weights are not. That is the whole design of `refit_profile` and of the
profile format.

**What fastText alone offers**: subword information, which would address the rare tail where
nothing else worked — a rare word inheriting from its morphological relatives, and incidentally
subsuming part of what the lemma map does. It is the one solid reason to consider a neural method
here, and it would still cost storing subword vectors and losing mergeability.

---

## 6. Method lessons worth carrying forward

- **An aggregate metric cannot distinguish repair from damage when the change reorders rankings.**
  The RRF normalization rewrote 64.9% of the network and its promoted candidates had mean presence
  1.78 inputs against 7.07 for demoted ones — exactly the intended correction — while `cidade` lost
  `cidades` and `municipio` lost `municipios`. Read the lists.
- **A pair judged in isolation is convincing and a full list reverses the verdict.** `fue` ->
  `exonerado` reads as noise alone and is correct as one of eight past participles. This error was
  made twice in one session.
- **Calibrate a parameter at the scale where it is applied.** A ratio relative to the batch means
  something different at 10k and at 122k documents, and everything derived from the small
  measurement had to be redone.
- **Do not edit a shell script while it is executing.** bash reads incrementally, so rewriting
  `wikipedia.sh` under a running driver made its read offset land mid-token; the run had already
  finished its work, but its exit code was garbage.
- **A test can silently stop testing.** Both the format-version test (faking "old" with a version
  that became current) and the stopword fixture (listing a stopword without applying it, so the
  input's counts stayed exact and the code path under test never ran) passed while checking
  nothing.
