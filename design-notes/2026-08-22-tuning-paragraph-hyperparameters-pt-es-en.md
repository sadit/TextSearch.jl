# 2026-08-22 — tuning the paragraph-level hyperparameters for pt, es and en

Companion to `2026-08-22-paragraph-units-and-query-expansion.md`, which established paragraphs as
the document unit. That note's paragraph-level numbers were **all Spanish**; this one measures the
three corpora and records the settings to build them with.

Measured on 10,000 articles of each language (first shard of `wikimedia/wikipedia` `20231101.*`),
split into paragraphs with headings folded forward and `--min-tokens 4`.

## The settings

| | value | scale-invariant? |
|---|---|---|
| document unit | paragraph, title-prefixed, headings folded forward | — |
| `min_tokens` | 4 | no (tokens, not a ratio) |
| `doc_freq_threshold` (stopwords) | **0.1** | no — this is a per-unit value |
| `head_df` (no expansion above) | **0.05** | no — same reason |
| `max_target_ratio` | **50** | **yes** |
| `min_ndocs` | 5 | no |
| `outdim` | 256 | yes |
| `k` | 8 | yes |
| `del_diac` / `del_punc` | false / true | yes |

The same values serve all three languages, which is the point: at article level the per-corpus
thresholds had to differ (es 0.5, pt 0.4, en 0.4) and inverted once when recalibrated at the right
batch size.

## What the measurement confirmed

| | paragraphs | per article | vocabulary | avgdoclen | flagged @0.1 | lowest of that set |
|---|---|---|---|---|---|---|
| es | 272,466 | 27.2 | 413,375 | 86.8 | 31 | `este` 0.122 |
| pt | 187,761 | 18.8 | 290,074 | 80.0 | 37 | `seu` 0.132 |
| en | 189,967 | 19.0 | 283,676 | 80.5 | 40 | `have` 0.123 |

**0.1 flags nothing but function words, in all three.** The band immediately below the cut is what
proves it, since that is what a lower threshold would take next:

- pt (0.10, 0.20], n=10: `sua à não ou também entre seu pela pelo ser`
- en (0.10, 0.20], n=20: `which are also be or this his were he has one not but have first had its
  their other after`
- es (0.10, 0.20], n=8: `no fue entre lo también sus son este`

So the threshold genuinely stops being a per-language tuning problem under paragraph units. That
was the substantive claim of the paragraph change and it survives contact with the other two
languages.

## What it corrected

**The factor of 55 between real stopwords and Wikipedia artifacts is Spanish, not general.**

| | artifacts | comparable content |
|---|---|---|
| es | `referencias` 0.018, `enlaces` 0.017, `externos` 0.016 | `musica` 0.020, `nombre` 0.038 |
| pt | `ligacoes` 0.014, `externas` 0.012, `referencias` 0.003 | `musica` 0.014, `nome` 0.038 |
| **en** | `references` 0.036, `see` 0.035, `external` 0.031, `links` 0.030 | `name` 0.033, `music` 0.025 |

In Spanish the artifacts fall below nearly all content. In English they land at 0.030–0.036,
**interleaved** with `name` (0.033) and `music` (0.025) — the bands touch, and the separation is a
factor of about 21 rather than 55.

The cause is in the first table: English averages 19 paragraphs per article against Spanish's 27,
so there is less to dilute a section heading into. Nothing breaks, because the artifacts reach
stopword status in no language and the point of paragraph units was that they stop *dominating*
rather than that they be eliminated. But the number was reported as a property of the method and it
is a property of Spanish Wikipedia.

Generalizing: **the dilution a paragraph split buys is proportional to paragraphs per article**, so
a corpus of short documents gets less of it. Worth checking that ratio before assuming the
separation holds on a new corpus.

## A structural constraint on `head_df`

`head_df` has to sit **below** `doc_freq_threshold` to do anything at all, because tokens above the
stopword threshold are no longer in the vocabulary by the time the expansion network is built. With
stopwords at 0.1 and `head_df` at 0.05, the working band is the strip in between:

- es: 68 tokens above 0.05, 31 removed as stopwords → **37** denied a list
- pt: 82 above, 37 removed → **45**
- en: 89 above, 40 removed → **49**

Consistent across the three, and it is the right population: the function words that survived the
stopword cut plus the likes of `anos ano parte forma ciudad ser ha donde` (es), `era mas grande
anos até foram tem parte` (pt), `been two they who new all most when such more` (en). None of those
needs enriching.

The cost, stated plainly: at 0.05 the band also contains `historia`/`história`/`history` (0.049–
0.069 across the three) and, in pt, `cidade século país freguesia estado norte`. Those are content
words that lose their expansion list. The judgment is that a term in more than one paragraph in
twenty is popular enough not to need help — defensible, and the knob is there to disagree with.

## Still unmeasured

- **The article-level `head_df`.** Article runs leave it at 0, since a document frequency means
  something different there and nobody measured it.
- **Whether these values hold at full corpus scale.** They were measured on 10k-article slices;
  the previous scale lesson was that a ratio relative to the batch is not scale-invariant, and a
  paragraph corpus of 48M documents is not a 272k one. The stopword threshold in particular is
  applied per batch at fit time, so the batch is what matters rather than the corpus, but the batch
  will be much larger than 272k paragraphs.
- **Fit cost at paragraph scale.** Full es is roughly 48M paragraph documents, pt 21M, en 116M.
  Fit cost is dominated by vocabulary rather than document count, so the earlier ~4 hour
  article-level estimate is not transferable; measure one shard before committing.
