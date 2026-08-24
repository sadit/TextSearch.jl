# 2026-08-23 — French, Italian and Basque, and what a non-Indo-European language does to the method

Three more corpora, built the same way as pt/es/en: paragraph units, `lc=false`, `del_diac=false`,
stopwords at 0.1, `head_df=0.05`, 800,000 paragraphs per part. Probed on 10,000 articles each
first, then run whole. Basque was the interesting one and the reason to write this down.

## What came out

| | paragraphs | vocsize | avgdoclen | stopwords | variants | fit | merge |
|---|---|---|---|---|---|---|---|
| eu | 1,748,911 | 319,606 | 31.4 | 47 | 17,135 | 9 min / 3 parts | 43s, 2.8 GB |
| it | 11,362,306 | 654,111 | 42.9 | 157 | 52,315 | 64 min / 15 | 3 min, 9.1 GB |
| fr | 21,419,641 | 868,175 | 36.2 | 113 | 131,866 | 100 min / 27 | 4m54s, 13.6 GB |

The merge memory model held to the largest case yet: 0.44 GB per part plus ~1.7 GB predicted 13 GB
for 27 parts, and it took 13.6.

## The threshold held from probe to full corpus, which is the thing that keeps being in doubt

`doc_freq_threshold` is a *ratio*, and calibrating it at one scale and applying it at another has
gone wrong twice in this project. So the number worth watching was what happens between a
10,000-article probe and the whole corpus:

| | probe | full | corpus grew by |
|---|---|---|---|
| eu | 51 | 47 | 15× |
| it | 146 | 157 | 55× |
| fr | 97 | 113 | 61× |

Barely moved. At paragraph level the band between function words and content is wide enough that
the exact value stops mattering — now confirmed in five languages, one of them not Indo-European.
No per-language tuning was needed for any of the three.

## French gains the most from keeping accents, by a wide margin

**131,866 non-derivable variants** — more than English with more than twice the paragraphs. And
they are its most frequent words, not a tail:

```
ete → été(31864)      etre → être(26548)     meme → même(26441)
apres → après(23015)  francais → français(20935)   ou → où(20687)
```

The end-to-end demonstration, on the finished profile:

```
$ textsearch search fr "eglise medievale" --collection paragraphs.jsonl -t 2
query: 2 token(s) -> médiévale Église église
  ~ eglise is in 78 documents against église's 263871, so it reads as a misspelling;
    searched as église (variant), Église (variant) instead
  ~ medievale is in 154 documents against médiévale's 26678, ... instead
  + 16 query_expansion(s) -> Antiquité Moyen abbatiale clocher conventuelle dédiée
    médiéval médiévales médiévaux paroissiale presbytère prieurale sacristie siècles Âge
```

Typing French without accents is what people actually do, and `eglise` **existed** in the
vocabulary with 78 documents — so without the ratio rule this query would have returned those 78.
That is the rule earning its place on a language it was not designed against.

## Italian: the apostrophe split works in its favour

157 stopwords, the most of any language here, because splitting on the apostrophe hands the
detector the elided prepositions and the threshold takes all of them: `dell della dall alla nell
nella all`. Its variants are the final-syllable accents — `citta → città`, `puo → può`,
`cosi → così`, `gia → già`, `pero → però`.

## The apostrophe also has one real defect

```
l'homme     -> ["homme"]           l is a stopword, filtered      ✓
qu'il       -> []                  both stopwords                 ✓
aujourd'hui -> ["aujourd", "hui"]                                 ✗
c'è (it)    -> ["c"]                                              ✗
```

Most splits are right: the clitic becomes a one-letter token and frequency removes it. But
`aujourd'hui` is **one word** destroyed into two meaningless fragments, in 6,393 and 7,400
paragraphs of the probe. Left alone deliberately — the fragments co-occur, so they behave like a
bigram of each other — but it is the cost of splitting on apostrophes with no exceptions, and a
language-specific rule is the fix if it ever matters.

## Basque, measured rather than assumed

Comparing vocabularies across corpora of different sizes is the scale error this project keeps
making, so this is the **same 100,000 paragraphs** of each, same config:

| | types | occurrences | types/occurrence | avgdoclen |
|---|---|---|---|---|
| es | 280,969 | 9,183,610 | 0.0306 | 91.8 |
| fr | 237,219 | 8,184,303 | 0.0290 | 81.8 |
| it | 267,289 | 8,419,477 | 0.0317 | 84.2 |
| **eu** | **335,679** | **5,724,232** | **0.0586** | **57.2** |

**1.9× the type/token ratio from 30% less text.** That is agglutination, quantified: case and
number live inside the word, so every noun appears as a family of forms. It shows up three more
ways — the shortest paragraphs (31.4 tokens in the finished profile against 36–45), the fewest
stopwords (47, since function morphemes are suffixes rather than separate words, the same reason
Arabic flagged 19 against Spanish's 46 at article level), and this:

```
gerra  → piztu gerraren Gerra Aliatuen Hotza Armada gerran
musika → Musika musikaren Guano punk musikak heavy ska
```

**The expansion network returns inflections, not concepts.** `gerraren`, `gerran`, `Gerra` are
case forms of the query word.

Tested whether applying lemmas fixes it: **no**. The map collapses 27,353 tokens and takes the
vocabulary from 81,211 to 53,858 — a third — but `gerraren` and `musikaren` survive and the lists
barely change. Morphological clustering with `min_common_prefix=3` and a semantic split is not
catching Basque case suffixes.

And on reflection that is not a defect to fight. In an agglutinative language, a query `gerra`
reaching `gerraren` and `gerran` is exactly the recall you need, because the corpus writes the
word in whatever case the sentence requires. The expansion artifact is doing the lemmatizer's job.
For Basque it therefore matters *more* than elsewhere, not less — which is the opposite of what I
expected going in, and the reason to keep it carried in the profile.

## Two atomicity holes, found by an operational accident

The French fit was killed twice, at the same stage, with SIGKILL and no Julia error. Not memory —
no cgroup limit, 239 GB available. Someone else's workflow on this shared machine runs
`killall -9 julia` (visible in `ps`, alongside a 13-day-old Jupyter kernel from another user), and
it does not distinguish whose Julia it kills.

That is an operating condition rather than a bug, but chasing it exposed two places where
**existence cannot tell finished from interrupted** — the third and fourth instances of that shape
in this project:

- `fit --resume` skips a part whose `.zip` exists. Killed during `zip_profile`, it would leave a
  truncated part that the next run skips and the merge then reads. Now written to `.partial` and
  renamed on success.
- `--limit` changed what a conversion contained without changing its name, so a 10,000-article
  probe and a full run shared a JSONL and a profile directory. A limited run is now
  `…-paragraphs-first10000`.

The retry loop in the runner then made the external kills cost a part instead of a run.

## Forecasting, for once, was fine

Estimated paragraphs from the probes (applying the ~3× correction the Spanish precedent gave for
"the first 10,000 articles are the longest"): eu 1.4M against 1.75M actual, it 8.7M against 11.4M,
fr 24M against 21.4M. All within 25% — the first estimates in this project that did not need
redoing.
