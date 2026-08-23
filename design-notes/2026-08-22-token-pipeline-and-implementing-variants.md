# 2026-08-22 — from a transformation hierarchy to a fixed pipeline, and implementing the variants

Continues [the casing/diacritics exploration](2026-08-22-casing-diacritics-and-query-side-variants.md),
which established *that* a base profile should keep case and diacritics and *that* something has to
bridge a folded query back to them. This note is about building it, and about the refactor that had
to happen first.

## The false start: a query-side `TextConfig`

The first design followed from a question worth asking — nothing in the tokenizer, the vectorizer,
BM25 or `TextInvertedFile` can tell whether it is fitting a profile, indexing, or answering a query,
and those stages should not all behave alike. The plan that came out of it: profile and index behave
identically, queries differ, so derive a second `TextConfig` for queries (`getquerytextconfig`) with
a `variants` stage added to the pipeline.

Both were built and both were deleted within the hour. **A pipeline of plain data cannot decide
whether to bridge**, because that decision needs to know whether the typed token is in the
vocabulary, and a `Dict` lookup stage does not have a vocabulary. Bridging is therefore not a
tokenization stage at all; it happens above tokenization, in `resolve_query_tokens`, which has the
`Vocabulary` in hand.

With variants out of the pipeline there is **no tokenization-time difference between documents and
queries left**, so the two-config derivation had nothing to derive and went too. One config serves
fitting, indexing and searching.

What the detour bought, and it is not nothing: it forced stating why fit and index must agree
*exactly*. That is not a preference to be configured — the vocabulary, the document frequencies and
the token ids all come from fit, so an index that tokenizes differently is producing ids that mean
something else. It is a correctness requirement, and it is now written down where the config is
built instead of being an unstated assumption that a future "configurable per stage" idea could
break.

## Why the generic transformation mechanism lost to a struct with two fields

`TextConfig.transformation::AbstractTokenTransformation` — an abstract type, five concrete
subtypes, a `transform` hook double-dispatching on transformation-and-generator, and
`ChainTransformation` to compose them — became `TextConfig.pipeline::TokenPipeline`, a struct of a
lemma `Dict` and a stopword `Set` applied in a fixed order.

The mechanism was right when TextSearch was more open. By the end it held exactly two real stages,
and **both are data rather than behaviour**: a set to filter by, a map to rewrite through. The one
stage that genuinely needed behaviour was the Snowball stemmer, removed earlier this cycle. Three
specific costs, all hit in this session:

**Order fails silently.** The stages do not commute. With the filter first, `"las"` is not in a set
containing `"la"`, survives it, and is only then rewritten — so a stopword enters the vocabulary
through the back door. This was documented *backwards* once and only measurement caught it. There
is now one function that can express the order and nowhere else.

**A missing dispatch method rejected identical inputs.** `merge_profiles` compared transformations
through one method per type, and the absent `LemmaTransformation` method made it refuse two
profiles carrying *the same* lemma map as "incompatible". A mechanism with a per-type comparison
obligation will eventually be missing one; `expected.pipeline == got.pipeline` cannot be.

**It was not free.** `ChainTransformation`'s field was typed
`AbstractVector{<:AbstractTokenTransformation}` — not concrete — so every step of every token paid
a dynamic dispatch. On 120,000 Spanish Wikipedia paragraphs (71.0M characters):

| pipeline | before | after |
|---|---|---|
| identity | 9.72s / 2.66 GB | 9.82–9.97s / 2.65 GB (noise) |
| stopwords only | 10.12s / 2.61 GB | 10.28s / 2.60 GB (+1.6%) |
| lemmas + stopwords | 14.65s / 3.62 GB | 10.44–10.50s / 2.60 GB (**−29%, −1.02 GB**) |

Applying lemmas went from costing 45% extra to costing nothing (10.44s against 10.28s for the
filter alone). That matters beyond the seconds: the `apply=false` default for lemmas in base
profiles now rests on a design argument only, not also on a budget one. And at full-corpus scale a
gigabyte of garbage per 71M characters is tens of gigabytes the collector no longer moves.

**Where extensibility actually lives.** A new *kind of token* — character q-grams, collocations,
keeping `H2O` whole, splitting `getUserName` into three — is an `AbstractTokenGenerator` in
`TokenizationConfig.generators`, which is the documented extension point and the right one: a
generator sees the word stream and may emit one token or several, while the pipeline is strictly
per-token. What has no home any more is an *algorithmic per-token rewrite*; adding one means adding
a named field with a documented position. That is cheaper than what it replaced (a new type, a
`transform_unigram` method, a comparison method — the one that was forgotten — and a decision about
where it chains).

## The variants map: only store what cannot be computed

The measurement that shaped the artifact. Over 272,466 Spanish paragraphs (10,000 articles),
folding the vocabulary gives 68,693 folded-form → token pairs. Of those:

- **46,200 (67%) are derivable** — `madrid → Madrid`, `general → General`, `GENERAL` — reachable
  from the folded form by `uppercasefirst`/`uppercase` with no corpus knowledge. Generating them at
  query time instead of storing them cuts the map 70%.
- **Accent restoration is not derivable and must be stored.** From `practico` there is no telling
  whether the corpus writes `práctico` or `practicó`; both are real words and only the corpus knows
  which it has.
- A **`min_ndocs` floor** removes the rest. A token in a handful of documents is not a plausible
  query target. Floor 20 takes the map from 62,825 keys to **8,153 — 87% smaller**.

And the two filters compose in the right direction rather than fighting: the derivable fraction
*falls* from 67% of pairs at a floor of 5 documents to 53% at 100, because rare tokens are
disproportionately proper nouns whose only variation is a capital. Raising the floor throws away
mostly the entries that did not need storing.

## Two policies, because reach and precision are the caller's call

- **`:strict`** — a token present in the vocabulary is used as typed and nothing is added. Someone
  who wrote `Sol` or `práctico` said something specific and it exists; writing carefully should not
  be penalized. Only a miss triggers folding.
- **`:aggressive`** — skip the presence check. This is the *only* way to reach an accented
  alternative of a token that is itself a vocabulary entry: typed `practico` stops at step 1 under
  `:strict` and never sees `practicó`.

That `:strict` limitation is pinned in a test, so it stays a decision instead of becoming a
surprise.

## It is spelling correction, so it is reportable

What this does is spelling correction — a deterministic kind covering case and diacritics but not
transpositions — and a search that silently substitutes what the user asked for owes them a way to
see it. `resolve_query_tokens` returns a `QueryResolution` recording, per typed token, whether it
was found and what was added with a **reason**: `:derived` for a computed spelling, `:variant` for a
stored one. `explain` renders it and distinguishes a correction ("leon not found, searched as …")
from an enrichment ("sol also searched as …").

The reason being a symbol is the extension point: edit-distance correction (`guerar → guerra`,
which SimilaritySearch could index the vocabulary for) reports as `:edit` through the same channel
without changing the signature. Deliberately not built — the core comes first.

## Merge and refit

`merge_profiles` **unions** variant maps rather than fusing them by rank consensus. They are derived
from a vocabulary, not estimated from a corpus, so there is nothing to vote on — the same argument
that made variants a separate artifact instead of entries in `query_expansion`. `refit_profile`
carries the base's map unchanged, since a refitted vocabulary is a subset and no entry can become
newly wrong.

## Open, and worth measuring before the three profiles are rebuilt

Measured on the wired end-to-end run: the same 272,466 Spanish paragraphs, `lc=false`,
`del_diac=false`, `doc_freq_threshold=0.1`, `head_df=0.05`, `max_target_ratio=50`,
`[variants] min_ndocs=20`. Fit took ~4 minutes and produced **8,153 keys — the number predicted
from the offline measurement, unchanged**, even though this vocabulary also passed through stopword
removal, which the prediction had not.

### Confirmed healthy

- **The invariant holds**: 0 of the 9,605 stored pairs is a derivable form. The map is exactly the
  non-computable part.
- **Fan-out is tight**: 6,725 keys (82%) reach one spelling, 1,409 reach two, and only 19 reach
  three or more. There is no structural explosion.
- **7,635 of 8,153 keys (94%) are not vocabulary tokens themselves.** The artifact is almost
  entirely about queries that would otherwise return nothing — which is the argument for it.
- **`:strict` is inert exactly where it should be.** `granada`, `sol`, `cuba`, `guerra` have no
  stored variants, so the sense separation that motivated `lc=false` survives untouched.

### `:strict` traps the user on a token that exists but is noise

`min_ndocs=5` on the vocabulary means unaccented misspellings and foreign-language fragments *are*
vocabulary tokens, so `:strict`'s premise — "it exists, therefore they meant it" — fails:

| typed | its docs | best variant | ratio |
|---|---|---|---|
| `ingles` | 7 | `inglés` 5,188 | 741× |
| `dia` | 10 | `día` 7,093 | 709× |
| `region` | 10 | `región` 6,360 | 636× |
| `musica` | 9 | `música` 4,404 | 489× |
| `rio` | 12 | `río` 4,265 | 355× |
| `ano` | 65 | `año` 18,285 | 281× |

Of the 518 keys that are vocabulary tokens, **236 have a variant at least 2× more frequent, 111 at
10×, 36 at 50×**. End to end on a 20,000-paragraph slice, `textsearch search musica` returns **0
paragraphs under `:strict` and 314 under `:aggressive`**.

The damage is doubled by expansion: the rare typed form contributes *its own* neighbours, and those
are junk with high idf. `musica`(9 docs, Italian-language paragraphs) gives
`libreto Puccini Verdi Vivaldi Semiramide Rossini`; `rio`(12) gives `llorar tiró enteró Nazgûl`
(the verb *reír*). A real search for `musica de leon` returns Antonio Vivaldi and a chess article.

The fix that follows: when the typed token is present but its best variant is R× more frequent,
treat it as a misspelling and **replace** rather than augment — reporting it as a correction, which
`explain` can already distinguish. Note the sign: `query_expansion`'s `max_target_ratio` forbids
connecting toward a much more popular word, while here a large popularity gap is *evidence*. Both
are right, because expansion is semantic and correction is orthographic.

### Derived forms bypass the floor that stored forms respect

`resolve_query_tokens` admits a derived candidate on vocabulary presence alone, with no document
count, while stored variants were filtered at `min_ndocs=20` when the map was built. So the query
side is more permissive than the artifact:

- `SOL` (5 docs) → `digitalizada máx chip flash aleatorio SDRAM`
- `Ano` (5 docs) → `Poo Bioko Ilha Mbini Corisco Elobey` (Annobón)
- `Musica` (9 docs)

Only 850 of 16,304 possible derived forms exist in the vocabulary at all, so applying the same floor
is cheap and removes precisely these.

### Expansion over the bridged set: no explosion, but real sense mixing

The feared blow-up did not happen — worst case is `esta` at 6 bridged forms + 32 neighbours = 38
terms, typical is 9–24. But attribution per source token shows the mixing plainly:

- `rio` (aggressive) = *reír* + Rio de Janeiro botany in Portuguese + rivers
- `sol` = the musical note + the astronomical Sun + SDRAM (from `SOL`, 5 docs)
- `ano` = the anatomical term + Annobón + `año`'s zodiac neighbours

**The noise tracks how rare the bridged form is, not whether it was bridged.** `Sol`(1,659), a
*derived* form, gives the best list of the group (`afelio perihelio eclipses eclíptica astro`).
So the answer to "expand the bridged set or only what was typed?" is neither: bridge broadly for
**matching** (each form is exact and cheap) and expand only the dominant form of the group.
Expanding only what was typed fails precisely when the typed form is the rare one — the `musica`
case above.

One honest counterexample: `esta`(20,395) / `Esta`(7,878) / `está`(16,200) / `Está`(1,421). `Esta`
is frequent but a *positional artifact*, and its neighbours are circuit-diagram noise
(`conmutación Circuito tautología C=`), while the much rarer `Está` gives the useful
`situada ubicado rodeado`. A purely frequency-based rule picks wrong here. Under `:strict` the case
never arises, since `esta` is present; it is a cost of `:aggressive`.

### The stopword/casing leak is real, and larger than expected

This was listed as a suspicion; it is now measured. 34 stopwords were detected at threshold 0.1.
The threshold caught four capitalized forms (`El`, `En`, `La`, `Los` — the commonest
paragraph-initial words) and let **52 twins through**:

| survivor | docs | df | its lowercase twin |
|---|---|---|---|
| `Las` | 22,040 | 0.081 | `las` removed |
| `A` | 19,701 | 0.072 | `a` removed |
| `Se` | 17,315 | 0.064 | `se` removed |
| `Por` | 12,891 | 0.047 | `por` removed |
| `De` | 9,800 | 0.036 | `de` removed |
| `Es` | 9,574 | 0.035 | `es` removed |
| `También` | 8,965 | 0.033 | `también` removed |

So the same function word is filtered in one casing and indexed as content in the other. It also
corrupts the variants map: **`tambien` bridges to `También` and nothing else**, because `también`
was removed — a query `tambien` now reaches only paragraph-initial occurrences.

The cause is combining `lc=false` with frequency-based detection: the threshold measures a fraction
of a *spelling*, not of a word. The fix is to detect over **case-folded** forms and remove every
casing of a detected stopword. Fold case only, not diacritics — folding diacritics would merge
`té`/`te`, `más`/`mas`, `sí`/`si` and delete content words, which is what `del_diac=false` exists to
prevent. Summing df across casings shifts the calibration slightly; for function words the margin is
enormous (`de` 0.55 + `De` 0.036), so only tokens sitting on the threshold could reclassify.

### Measured and *not* a problem

Wikitext markup in the vocabulary looks alarming when reading the rare tail — `align=left`,
`bgcolor=`, `#FFFFCC`, `rowspan=0`, `class=wikitable` — and those tokens cluster into their own
expansion lists. But there are **76 of them in 119,411 tokens (0.06%), 3,928 document occurrences
in total**. Not worth a converter change; recorded so it does not get re-flagged.

### Fixed, and re-measured on the same 272,466 paragraphs

**Stopword detection is per spelling; removal is per word.** The leak is closed by extending each
flagged spelling to every other casing the vocabulary holds, not by pooling document frequencies
before comparing to the threshold — the pooled value is not observable from these counters (a
document holding both `de` and `De` is counted twice, and the sum can exceed 1) and it would move a
threshold that was calibrated per spelling. Detected stopwords went **34 → 90**, surviving twins
**52 → 0**, vocabulary 119,411 → 119,366. The sweep also picks up mixed casings that only exist as
noise (`CoMo`, `coMo`, `DEl`, `eL`, `parA`). `también` and `También` now go together, so a query
`tambien` contributes nothing — the same as under `lc=true`, and correct.

**One ratio rule replaced both of the variant fixes.** They turned out to be the same measurement:
each typed token defines a *group* of candidate spellings, and a spelling holding less than
`1/negligible_ratio` of the group's commonest is negligible. Default 50.

- It decides whether `:strict` bridges a token that *is* present: `musica`(9) beside `música`(4,404)
  is corrected, `granada`(43) beside `Granada`(1,438) is not.
- It prunes negligible *bridged* spellings, which is the gap where the query side was looser than
  the artifact: `sol` no longer reaches `SOL`(5), `ano` no longer reaches `Ano`(5).
- **The typed form is never dropped.** Keeping it costs a handful of false positives; discarding a
  person's own word to search something else instead is not a trade to make unasked.

**Expansion comes from one spelling per typed token** — the group's dominant — via
`expansion_sources`. This is what actually removed the junk:

| query | before | after |
|---|---|---|
| `musica` | `libreto Puccini Verdi Semiramide Rossini` | `folclórica musicales carnática musical bands` |
| `rio` | `llorar tiró Nazgûl` + `Jardim Botânico espécies` | `afluente confluencia cauce desemboca` |
| `mexico` | `Aztec morphology Assoc clinical Vol living` | `mexicanos Tenochtitlan Tijuana Mexicana` |
| `sol` (aggr.) | `… digitalizada máx chip flash SDRAM` | `afelio perihelio eclipses eclíptica` |

End to end, `search musica` went from **0 paragraphs to 314**, and `musica de leon` from 39 search
terms of mixed provenance to 24 coherent ones.

**Where the threshold sits.** Over the 517 map keys that are themselves vocabulary tokens, the
dominant/typed ratio distributes as 266 in [1,2), 87 in [2,5), 50 in [5,10), 39 in [10,20), 23 in
[20,35), 15 in [35,50), then 6, 5, 11 and 15 in [50,75), [75,100), [100,200) and above. There is no
clean gap at 50 — the decay is smooth — but the region around it is *sparse*, so the choice is not
delicate: 37 of 517 keys are corrected at 50, and moving to 40 or 60 moves a handful. Of the
sense-separation cases that motivated `lc=false`, five of six stay untouched (`sol` 1.4×, `iglesia`
1.8×, `palma` 3.3×, `granada` 33.4×, plus `guerra` with no variants) and one crosses: `cuba`(18) vs
`Cuba`(2,178) at 121×, which reads as a typo rather than a sense split. Its 18 barrel paragraphs
still match; only their neighbours are no longer pulled in.

**Two interactions worth recording.** `head_df=0.05` already excludes very frequent words as
expansion sources, so `ano → año`(df 0.067) and `esta`(0.075) bridge and then expand to nothing —
the two filters compose as intended rather than fighting. And the ratio rule has an inverse sign to
`max_target_ratio` in `query_expansion`: there a large popularity gap disqualifies a neighbour,
here it is the evidence of a misspelling. Both are right, because expansion is semantic and
correction is orthographic.

### Reversed the same session: correcting should replace, because the mode is expressible

The "never drop what the user typed" rule above was a hedge, and it was a hedge for a reason that
turned out to be removable: with no way to express *how* a query should be treated, dropping a
person's word gave them no way to get it back, so the safe move was to always union. Model it on
what commercial search does instead — answer the most probable reading, and always offer the same
query answered literally — and the hedge is unnecessary: correcting can actually correct, because
"search instead for X" exists.

So the shape is a mark on the query rather than a policy buried in a call:

```julia
QueryPolicy(; correction=:auto, expansion=true, expansion_k=0, negligible_ratio=50)
```

- `:auto` corrects only where the evidence says the typed spelling is wrong — absent, or
  negligible beside a commoner spelling of the same word — and where it corrects it **replaces**.
- `:off` is the literal answer. A consumer that corrects by default owes it.
- `:always` bridges without evidence, so it *adds* rather than replaces: nothing said the typed
  form was wrong.

That "drop exactly when the evidence says wrong" rule is what makes `:auto` and `:always` one
mechanism rather than two, and it made `rare` collapse into `kept` on `ResolvedToken` — the
outcome, not the reason, is what a consumer needs in order to phrase three different sentences
(not there at all / there but too rare / enriched and left standing).

Two things follow that were not obvious before framing it this way:

- **Expansion is the same kind of thing.** It is the other guess a search makes about intent, so
  it belongs on the same object and is answerable literally in the same way. `--no-query_expansion`
  already existed; what changed is that it stopped being an unrelated flag.
- **`negligible_ratio` found its home.** It is neither a fit-time knob (the same profile serves
  every reading) nor a bare library default — it is part of how *this query* should be treated, and
  it rides along with the mode. That answers a question left open an hour earlier about where to
  put it.

Measured on the CLI, `musica de leon` over the paragraph profile:

| `--correction` | searched | expansion |
|---|---|---|
| `off` | `leon musica` | `Puccini Semiramide Verdi Vivaldi libreto` |
| `auto` | `Leon León Léon Música león música` | `leonesa castellanoleonés folclórica musicales …` |
| `always` | same as auto here | same |

The `off` row is the honest one to keep in mind: asked for the literal query, the profile answers
with the literal query's neighbours — Italian opera, from nine paragraphs — and that is correct
behaviour, not a bug to filter away.

### Still open

- Everything under *Still unmeasured* in the previous note stands: whether `lc=false` degrades the
  rare tail, and whether the casing signal holds on later shards (these 10,000 articles are the
  longest and best-written in the corpus).
- The rare tail of `query_expansion` is **mixed, not junk**, which reconfirms the earlier session's
  finding that no cheap statistic separates a good list from a bad one down there: 60% of the
  119,411 keys have fewer than 20 documents, and among them `Messner → Mallory ochomiles Everest
  Tenzing sherpas` and `ABM → antibalísticos SALT START Gromyko` sit beside pure co-occurrence
  noise. So an absolute floor on expansion sources is not supportable; the relative rule above is.
