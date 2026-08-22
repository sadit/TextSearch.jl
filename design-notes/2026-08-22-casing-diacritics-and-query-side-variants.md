# 2026-08-22 — case, diacritics, and where orthographic variants belong

Design exploration, not implemented. The question: should a base profile fold case and diacritics
at fit time, or preserve them and let the consumer fold at query time?

Current profiles use `lc=true, del_diac=false`. The proposal on the table is `lc=false,
del_diac=false` for a base profile, with the folding derived later: read the vocabulary, normalize
each token, and where the folded form is absent add a correction entry so a query typed in the
folded form still reaches the corpus form.

## Measured on 272,466 Spanish Wikipedia paragraphs (10,000 articles)

### Cost: much smaller than argued

| | `lc=true` | `lc=false` |
|---|---|---|
| vocabulary (min_ndocs=5) | 106,447 | 119,445 |

**×1.12, not ×2.** The objection raised against `lc=false` was that Spanish capitalizes at sentence
start so nearly every frequent word would gain a capitalized twin. That is wrong: sentence-initial
position is dominated by a bounded set of function words, and the long tail of content words that
happen to start a sentence is rare enough per token that `min_ndocs=5` prunes it.

Of 54,213 tokens carrying an uppercase letter, only **16,463 have a lowercase twin** in the
vocabulary. The rest are proper nouns that duplicate nothing. The frequent twins are exactly the
predicted noise -- `El/el` (0.277/0.680), `La/la`, `En/en`, `Los`, `Las`, `A`, `Se`, `Por`, `De` --
which the stopword threshold or `head_df` catches anyway.

Where the uppercase form *dominates* its twin, it is genuine proper-noun usage: `Estados`
(0.0413/0.0076), `Unidos` (0.0381/0.0017), `San` (0.0381/0.0029), `Historia`, `Nacional`, `Estado`.

### Benefit: it recovers senses that folding destroys entirely

This is the class eight statistical filters could not touch (see
`2026-08-22-paragraph-units-and-query-expansion.md`), and casing separates it for free, from a
well-written corpus, with no corpus-derived model at all:

| token | `lc=true` (folded) | `lc=false` lowercase | `lc=false` Uppercase |
|---|---|---|---|
| `granada` | `baza granadino nazari guadix` | **`gorro gules azur bordura heraldica`** | `Baza nazari Sacromonte` |
| `lima` | `peru peruana piura pisco` | **`limon mazamorra crema cascara jugo`** | `Peru Piura Ica` |
| `cuba` | `cubanos habana cubana` | **`osmosis bebida evapora hirviendo`** | `cubanos Matanzas Camaguey` |
| `palma` | `aridane tenerife gomera` | **`cana trigo cacao coco arroz`** | `Gomera Aridane Tenerife` |
| `leon` | `leonesa ribadeneyra sancha` | **`serpiente leones laurel timbrado`** | `leonesa Ribadeneyra Febres` |
| `concepcion` | `penco chiguayante talcahuano` | **`pensamiento vision materialista filosofica`** | `Penco Chiguayante Talcahuano` |
| `sol` | `afelio eclipses perihelio` | **`mediodia brilla diurno cielo aurora`** | `afelio perihelio orbita` |
| `iglesia` | `ortodoxa anglicana catolica` | **`parroquial catedral convento capilla`** | `catolica ortodoxa apostolica` |

In **every** pair, folding returns only the proper-noun sense and loses the common-noun sense
completely: the heraldic pomegranate, the barrel, the philosophical concept, the animal in its
heraldic use (`timbrado`, `laurel`), the church building as distinct from the institution.

### The premise "trust the LSI to link the twins" does not hold -- and that is the good news

Of the 16,376 lowercase forms with a list, the uppercase twin appears in only **13.6%**, is in the
top three in **10.0%**, and is rank 1 in **6.3%**. Reverse direction: 19.4%.

What matters is *which* cases fall on each side.

Where the twin **is** reachable, it is function words -- `ademas:1`, `aunque:1`, `tras:1`,
`estos:1`, `estas:1`, `hay:1`, `si:8` -- where the capital is only sentence-initial and the two
forms mean the same thing.

Where it is **not**, the lowercase form has a coherent list of its own distinct sense:

| lowercase | its list | while the uppercase is |
|---|---|---|
| `unidos` | `covalente covalentes atomos enlazados enlaces` | Estados **Unidos** |
| `nacional` | `Tumucumaque Yellowstone Monfrague Huascaran` | (parque) **Nacional** |
| `historia` | `Fujur interminable Atreyu historica` | **Historia** as a title |
| `guerra` | `Buque Clausewitz Pentecontecia civil` | **Guerra** as a name |
| `republica` | `presidencialista semipresidencialista parlamentaria` | **Republica** |
| `estado` | `Nadu Illinois Tamil Zacatecas` | **Estado** |

So **the 13.6% where the twin is reachable are the cases where casing carries no meaning, and the
86% where it is not are the cases where it does.** The LSI is behaving correctly: it refuses to
link forms that differ in sense. `leon` should not have `Leon` as a synonym any more than the
animal is a synonym for the city.

## What follows for the design

1. **`lc=false` for a base profile is well supported.** ×1.12 vocabulary for a sense separation
   that no statistical method achieved, on a corpus whose casing is reliable. It also matches the
   policy already applied to lemmas and stopwords: the base carries, the consumer decides.

2. **The derived folding is therefore necessary, not optional.** It cannot lean on the expansion
   network, because that network correctly does *not* bridge forms whose senses differ. Whatever
   reaches `leon` from a query typed `leon` has to be an explicit orthographic bridge.

3. **It should be its own artifact, not entries in `query_expansion`.** An orthographic identity is
   deterministic, derivable from Unicode without a corpus, and exact; the expansion network is
   statistical, corpus-derived and approximate. Merging them by the same rule is wrong in a
   concrete way: `merge_profiles` fuses the expansion network by rank consensus (RRF), while
   orthographic variants should merge by plain union, since there is nothing to vote on.

4. **It must fan out one-to-many.** A query `leon` -- unaccented and lowercase -- should reach
   `leon`, `leon`(accented) and `Leon`, because the user's intent is ambiguous. Ranking among them
   is a separate question; reachability comes first.

5. **It has to be applied at query tokenization, not on the query vector.** Measured constraint:
   `cli_search`'s `_query_tokens` looks the network up **by string** before any vocabulary lookup,
   so a variant keyed by a form absent from the corpus works there. But the library's
   `expand_query!` -- both the `SparseVector` and the `AbstractDict` overloads -- iterates the
   query vector's nonzeros, which are already vocabulary ids. A query token absent from the
   vocabulary never reached the vector, so nothing can expand it. A variant mechanism that only
   works in the CLI path would be a trap.

6. **Refit: make folding terminal.** Refit blends the base's counters with a sample tokenized under
   the base's `TextConfig`, so it all happens in the unfolded space and refit needs no changes.
   What must be prevented is refitting an already-folded profile, which would mix the two spaces.
   The lineage can express that: record folding as a step and have `refit_profile` refuse a profile
   carrying it, the way `load_profile` refuses a foreign format version. One terminal form,
   explicit, no special cases.

## Still unmeasured

- **Whether `lc=false` degrades the rare tail.** The evidence above is from frequent tokens, where
  each split form keeps enough documents for a coherent list. Splitting a token with 8 documents is
  a different matter, and the session established that no cheap statistic distinguishes a good list
  from a bad one down there.
- **Whether the casing signal holds on later shards.** These 10,000 articles are the longest and
  best-written in the corpus; shard 7 averages 6.7 paragraphs per article against shard 1's 17.5.
  A good result here is a ceiling, not an average.
- **The same question for `del_diac`.** Diacritics are already preserved (`del_diac=false`), and the
  measured payoff is comparable -- `rio` (df 0.0913) gives `afluente confluencia desembocar` while
  `rio`(unaccented, 0.0057) gives `sul janeiro sao`, and `ano` versus `ano` separates the year from
  the anatomical term. The variant mechanism has to cover this direction too, and it is the more
  common query error: users omit accents far more often than they miscase.
