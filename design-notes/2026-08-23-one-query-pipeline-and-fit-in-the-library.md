# 2026-08-23 — one query pipeline, and a library you can actually fit a profile with

The ask was shortcuts: find patterns written the long way and compact them, keeping the long path.
Measuring first turned up the boilerplate, but it also turned up something the shortcuts were
sitting on top of — the profiles we had just spent a day building were **not usable from the
library**. So this note is mostly about that, and the compaction is the tail end.

## What was actually repeated

| pattern | occurrences | note |
|---|---|---|
| `verbose=false` | 177 | the default is `true`; left alone, see below |
| `nlist=[1]` | 50 | **and it is already the default** |
| `VectorModel(IdfWeighting(), TfWeighting(), voc)` | 31 of 43 | the other 12 use four other combinations |
| `NormalizationConfig(...)` spelled out | 13 files | always inside `TextConfig(normalization=…, tokenization=…)` |
| `_fit_one_batch` | 110 lines, 14 library calls | no library equivalent existed |
| `_query_tokens` | 31 lines | nor for this |

`verbose=true` stays: it is right for a default, and turning it off in tests is not the same
problem as turning it off in production. Recorded because 177 occurrences look like a smell and
are not one.

## The thing the shortcuts were hiding

`TextInvertedFile(textconfig, corpus)` already existed as a one-call constructor, which I cited as
precedent for adding more. It is the **cosine** index; BM25 — the one that matters for these
profiles, since `avgdoclen` and `doclen` are BM25 concepts — had no such thing. Following that up
turned into the real finding.

Tried a pre-trained profile (pt, 6,665,754 paragraphs) against `BM25InvertedFile` over 20,000
paragraphs. Half of it was right and worth confirming: the scorer's `trainsize` and `avgdoclen`
are the **corpus's**, and the idf reads document frequencies from the profile's vocabulary, while
`doclens` are the **indexed** documents'. That split is the whole point of shipping counters.

The other half was broken:

```
região      hits=3  Guiné-Bissau / dividida em oito regiões        ✓
regiao      hits=3  Lista de aves do Brasil / Trogoniformes        ✗
município   hits=3  Condeixa-a-Velha, antiga freguesia             ✓
municipio   hits=3  Fascismo / Giovanni Gentile                    ✗
```

`regiao` did not return nothing — it returned *garbage*. It is in the vocabulary (33 documents),
so `expand_query!` fired and added the neighbours of a misspelling. The index path **did not
correct and did expand**, which is exactly the failure mode fixed in the CLI the day before.

And nothing tested it: the four test files touching `load_profile` never built an index, and the
files building indexes never loaded a profile.

## Why there were two query pipelines, and why unifying is not a choice between them

Looking for how to merge them explained why they had diverged. Each could do something the other
could not:

- **Vector-level** (`expand_query!`, called by both inverted files) weighted what it added, by rank
  (`1/rank`) or distance (`exp(-d)`), and summed the contributions of a neighbour reachable from
  several query tokens. But it iterated a query vector's nonzeros — already token ids — so a typed
  spelling absent from the vocabulary never reached it. Correction was impossible **by
  construction**.
- **String-level** (the CLI's own) corrected first and expanded only from the commonest spelling
  of each corrected group, but added terms unweighted, because it fed a `Set` for grep-like
  matching.

So the unified pipeline runs on strings, where correction is possible, and emits weights, so
nothing is lost:

```
text -> tokenize -> correct -> expand -> QueryTerm(token, source, factor, reason)
```

and each consumer materializes what it needs, keeping its own semantics: `queryvector` weights and
normalizes, `querybow` is presence-only, `querytokenset` is the plain set.

Three details each cost a debugging round:

- **`querybow` being presence-only is not a shortcut.** `bm25score` never reads the query side's
  frequencies, only which ids are present, and `BOW`'s counts are `Int32`. The dict overload of
  `expand_query!` used `one(V)` for exactly this reason. A weight there would be carried through a
  whole search and then discarded.
- **A neighbour inherits the *source term's* weight**, times the factor — not its own idf. That is
  what `expand_query!` did (`nzval[i] * w`), and it is a real design decision: a neighbour of a
  rare, high-idf query word enters heavier than a neighbour of a common one. `QueryTerm` therefore
  carries a `source` and a `factor` rather than an absolute weight.
- **Expansion contributions must not be deduplicated in the pipeline.** `expand_query!` merged
  duplicate ids by *summing* them. Deduplicating moved weights by up to 0.149 on a two-token query
  while leaving the id set identical — the kind of difference that passes an eyeball check.
  Deduplication belongs in materialization, where a set collapses them anyway.

Verified against the old path on three queries × two weightings: identical indices and weights to
1e-6.

## The asymmetry that fell out

`BM25InvertedFile(p::TextProfile)` and `TextInvertedFile(p::TextProfile)` now exist, and they gate
the two query-side artifacts differently:

- **Expansion follows the profile.** The network is handed over only when
  `applied.query_expansion` says the profile endorses it; `expansion=true` takes it anyway, which
  is what a *base* profile needs — and all three corpus profiles are base.
- **Correction follows the policy.** It depends on nothing but the vocabulary, which every profile
  has. The variant map is derived once at construction, since 0.6s per index is fine and per query
  is absurd.

Coherent rather than arbitrary: one is an artifact the profile may carry without meaning it, the
other is a property of the query.

## `fit_profile`, and what comparing it exposed

The 110 lines moved into the library with grouped keywords (`stopwords`, `encoder`, `expansion`,
`lemmas`) that map 1:1 onto the config file, so the app really is only turning TOML into
arguments. `cli_fit.jl` 360 → 224 lines.

Verified by fitting one corpus both ways and comparing every field — trainsize, vocsize,
numtokens, stopwords, lemmas, network, distances, applied, materialized `TextConfig`, lineage
params: **ten for ten identical**, with and without applied lemmas. Getting there exposed two
things that were not mine.

**Config equality was by identity.** `NormalizationConfig`/`TokenizationConfig`/`TextConfig` had
no `==`, so Julia compared fields with `===` and several are heap objects (`nlist`, the emoji
table, the compiled regexes): two configs built from the same settings compared unequal.
`merge_profiles` carried a private field-by-field comparison because of it, `refit_profile` called
into that, and a test *documented the trap* (`@test tc.tokenization != tc2.tokenization`). The
types now define `==` by value; the merge keeps only the one thing it is genuinely stricter about
(it declines configs carrying custom generators, which cannot be shown equivalent).

**A fit is not reproducible, and sorting cannot make it so.** This one matters beyond tidiness.

`allknn` returns neighbours by increasing distance but says nothing about the order among equals,
and it varied: `esta` came out `["rica","manzana","pera"]` one run and `["rica","pera","manzana"]`
the next, with identical distances. Sorting ties by `(distance, token)` fixes that, and is worth
having.

What it does not fix: the factorization runs threaded, so its reductions sum in a nondeterministic
order and the embeddings differ around the **seventh significant digit** — 0.05991161 against
0.05991143 for the same pair. That is enough to flip a near-tie across the top-k boundary, so
`rica`'s third neighbour changed between runs even after the sort.

So **two fits of one corpus are equivalent, not equal, and a published profile cannot be verified
by checksum.** Verification has to be structural: counters, stopword sets, vocabulary, network
keys, distances to a tolerance. Worth settling before the three profiles go up as release
attachments. The tests pin what is stable and pin the tie ordering hermetically, on synthetic
vectors where the ties are exact.

## The compaction, finally

- **Flat keywords on `TextConfig`**: `TextConfig(lc=false, del_diac=false)`. Saying a setting both
  ways is an error, not precedence. The copy constructor takes them too — and rebuilds a
  sub-config only when something is actually overridden, which a test asserting
  `getpolicy(p).tokenization === gettextconfig(p).tokenization` caught me breaking.
- **`VectorModel(voc)`** for TF-IDF.
- **The app stopped carrying its own query pipeline.** `_query_tokens` became `_query_report`,
  which calls the library and keeps only what the stderr summary needs. Beyond line count:
  `textsearch search` can no longer drift from what an index would do with the same query.

## What a caller writes now

```julia
p   = fit_profile(TextConfig(lc=false), corpus; stopwords=(; doc_freq_threshold=0.1))
idx = BM25InvertedFile(p; expansion=true)
append_items!(idx, ctx, docs)
search(idx, ctx, "musica de leon", knnqueue(KnnSorted, 10))
```

Against reproducing 110 lines and still getting the query side wrong.
