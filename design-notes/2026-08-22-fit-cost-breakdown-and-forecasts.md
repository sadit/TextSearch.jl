# 2026-08-22 — where a fit's time goes, and what the three corpora will cost

Measured before committing the compute, on Spanish Wikipedia. Companion to
`2026-08-22-tuning-paragraph-hyperparameters-pt-es-en.md`, which fixed the hyperparameters.

## Corpus size: the first shard is an outlier, and every earlier estimate was wrong

Paragraph counts per shard, es (141,628 articles each, `--split-paragraphs --min-tokens 4`):

| shard | paragraphs | per article |
|---|---|---|
| 1st | 2,482,491 | 17.5 |
| 4th | 1,040,688 | 7.35 |
| 7th | 950,004 | 6.71 |
| 13th | 953,629 | 6.73 |

Wikipedia orders articles longest-first, so the first shard has more than twice the paragraphs of
any other and the count flattens by the fourth. **Every paragraph-level measurement in this
project was taken on the first 10,000 articles, which gave 27 paragraphs per article** -- four
times the corpus-wide figure. That inflated the size estimate three times over:

    48M paragraphs (first estimate, from 27/article)
 -> 31M (after measuring shard 1 at 17.5/article)
 -> ~14.5M (after measuring shards 4, 7 and 13 at ~6.7-7.4/article)

Projected, using es's corpus-wide 8.15 paragraphs per article scaled by each language's
first-10k ratio (pt 18.8, en 19.0 against es 27):

| | articles | paragraphs (est.) | JSONL |
|---|---|---|---|
| pt | 982,641 | ~5.6M | ~3.4 GB |
| es | 1,778,209 | ~14.5M | ~8.8 GB |
| en | 6,083,989 | ~34.9M | ~21 GB |
| | | **~55M** | **~33 GB** |

This is the third time in the project that measuring a convenient slice and extrapolating gave
the wrong answer -- the others being the stopword threshold calibrated at 10k and applied at
122k, and the factor-55 artifact separation measured on Spanish and reported as general.

## Where the time goes

Nine stages of `_fit_one_batch`, timed on the same 10,000 Spanish articles as articles and as
paragraphs, so the split's effect on the *distribution* is visible and not just on the total:

| stage | articles (10k) | paragraphs (272k) |
|---|---|---|
| 1. vocabulary, detection pass | 35.6s (39%) | 36.5s (30%) |
| 2. `stopword_candidates` | 0.0s | 0.0s |
| 3. vocabulary, 2nd pass + prune | 21.1s (23%) | 23.4s (19%) |
| 4. weights | 0.1s | 0.0s |
| **5. LSI (truncated SVD)** | **20.5s (23%)** | **55.4s (45%)** |
| 6. wordvectors | 0.6s | 0.1s |
| 7. query-expansion kNN | 10.4s (11%) | 6.3s |
| 8. lemma_clusters | 1.3s | 1.2s |
| 9. save_profile | 1.2s | 0.4s |
| **total** | **90.9s** | **123.2s** |

**Tokenization is linear in text, not in document count.** The two passes cost the same under both
units (35.6 -> 36.5 and 21.1 -> 23.4) because it is the same text; splitting into paragraphs only
regroups it. The only stage that grows with document count is the LSI, 20.5s -> 55.4s for 27x the
documents.

So a fit is roughly linear in corpus size, and batch size only optimizes the vocabulary-dependent
half. Batch scaling, measured end-to-end through the app:

| batch | time | s/100k docs | peak RSS | vocabulary (raw -> pruned) | profile |
|---|---|---|---|---|---|
| 200k | 2:22 | 71 | 2.7 GB | 352,839 -> 90,429 | 19 MB |
| 400k | 3:47 | 57 | 3.4 GB | 500,679 -> 129,493 | 27 MB |
| 800k | 6:39 | 50 | 5.4 GB | 730,986 -> 187,952 | 39 MB |

Doubling the batch costs ~1.7x, not 2x, because the vocabulary grows sublinearly (Heaps) and the
vocabulary-dependent work with it. Bigger batches are better, bounded by the linear half and by
memory.

## Three ways to avoid the detection pass, and which wins

Stage 1 exists only to produce the list of tokens above the threshold, and it is the largest
single stage. Three alternatives were measured.

**(a) Filter pass 1's vocabulary instead of re-tokenizing.** Removing a token from the tokenizer
does not change any other token's counts, so filtering `voc0` should equal a second pass with
`IgnoreStopwords`. Measured on 272,466 paragraphs, it is **bit-exact**: vocsize 106,416 both ways,
numtokens 13,991,721 both ways, zero tokens missing and zero with differing counts -- in 0.1s
against 22.5s. **Only for unigram configurations**: with `nlist=[1,2]` the two diverge badly
(vocsize 408,132 vs 543,498, with 145,762 tokens of one absent from the other), because dropping a
stopword rewrites the n-gram stream.

**(b) Detect on a sample.** With paragraphs at threshold 0.1, samples of 20%, 10% and 5% all
recover the set **exactly** (31 tokens), 5% in 1.9s against 37.3s. With articles at 0.5 even 20%
differs (`anos` appears), and 2% loses `hasta pero`. The reason is the same property that makes the
threshold easy to set under paragraph units: the band around the cut is empty there (highest
content word 0.069 against a cut of 0.1) so sampling noise cannot cross it, while at article level
21 tokens crowd (0.4, 0.5].

**(c) The first batch detects, later batches reuse.** Measured across five 400k-paragraph batches
-- three disjoint windows of shard 1 and two of shard 7:

| batch | own set | not covered by the 1st | removed extra |
|---|---|---|---|
| s1 [0,400k) | 31 | — | — |
| s1 [800k,1.2M) | 30 | **0** | 1 (`son`) |
| s1 [1.6M,2M) | 30 | **0** | 1 (`son`) |
| s7 [0,400k) | 27 | **0** | 4 (`entre este lo son`) |
| s7 [400k,800k) | 26 | **0** | 5 (`entre este lo son sus`) |

The first batch's set is a superset of all of them, and the union of the five equals it exactly.
Reuse loses nothing and removes at most five extra function words.

The prediction going in was the opposite -- shard 7 has 6.7 paragraphs per article against shard
1's 17.5, so less dilution, so more tokens should cross the cut. It flags **fewer** (26-27 against
30-31), because its short articles have short paragraphs and a short paragraph simply does not
contain `entre` or `sus`. Fewer tokens per paragraph lowers the document frequency of mid-frequency
function words, and that effect dominates.

Which gives the structural reason the style works, and its limit: Wikipedia orders articles
longest-first and longer paragraphs hold more function words, so the first batch maximizes their
document frequency **by construction**. On a corpus in arbitrary order, or one whose first batch is
topically narrow, it would not hold; the safe fallback there is detection on a sample of the whole
corpus rather than of the first batch.

**Comparison**, per part:

| | cost of stages 1+3 | exact | fixes the merge |
|---|---|---|---|
| today: two full passes | 61.9s | yes | no |
| (a) filter `voc0` | 39.5s | yes (unigrams only) | no |
| (b) 5% sample | 24.4s | in practice | no |
| **(c) reuse the first batch** | **22.5s** for parts 2+ | **yes** | **yes** |

(c) wins on every axis, and the last column is the one that matters most: if every part removes the
*same* set, no token is removed in some parts and kept in others, so no counts are partial, so the
merge needs no imputation at all. That is the entire bug class fixed in `_impute_removed_stopwords`,
gone by construction. The imputation must stay for profiles that mix different sets, but a corpus
built this way will not produce them.

Note (a) and (b) are mutually exclusive: filtering needs the full counts a sampled pass does not
have.

## Time forecasts

Per-part cost at 800k paragraphs is 399s measured end-to-end; with reuse, parts after the first
skip the detection pass, which is ~30% of the stage work, giving ~287s.

| | paragraphs | parts @800k | fit |
|---|---|---|---|
| pt | ~5.6M | 7 | **0.6 h** |
| es | ~14.5M | 18 | **1.5 h** |
| en | ~34.9M | 44 | **3.5 h** |
| | | 69 | **~5.6 h** |

Plus ~50 min of parquet-to-JSONL conversion across the 60 shards, and ~15 min of merges. **About 7
hours total**, against the 16.4 hours estimated before the corpus size was measured properly.

Resources: ~33 GB of intermediate JSONL, ~2.7 GB of part profiles, 5.4 GB peak RSS per fit. Disk
is not a constraint (2.8 TB free).

Unverified in these forecasts: the en and pt paragraph counts are scaled from es rather than
measured shard by shard, and the ~30% detection-pass share is taken from the 272k-paragraph
breakdown rather than measured at 800k.
