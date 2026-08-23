# 2026-08-23 — building the pt and es paragraph profiles, and what the real corpus showed

First run of the whole pipeline at corpus scale: paragraph units, `lc=false`, `del_diac=false`,
stopwords at 0.1, `head_df=0.05`, 800,000 paragraphs per part. Everything measured here is on the
real dumps, not on a 10k-article probe — which matters, because three of the findings below are
things a probe cannot show.

## What it produced

| | articles | paragraphs | par./art. | parts | convert | fit | profiles |
|---|---|---|---|---|---|---|---|
| pt | 982,641 | 6,665,754 | 6.78 | 9 | 3.0 min | 31.5 min | 362 MB |
| es | 1,778,209 | 14,577,732 | 8.20 | 19 | 6.6 min | 69 min | 813 MB |

`avgdoclen` came out **47.8** tokens for pt's first part and **48.6** for es's, against roughly
2,300 for an article. That is the quantity BM25 normalizes every document length by, and it is the
whole point of the paragraph decision.

Merging pt's 9 parts: **1m39s, 5.8 GB peak RSS**. That was the open risk — 19 and 67 parts were
unmeasured — and it is now bounded: the machine has 264 GB, so even 67 parts extrapolate to ~43 GB.

Verifications that ran against the profiles rather than the logs: all 28 parts load, their
`trainsize` values sum to exactly the conversion's record counts, and the stopword sets are
byte-identical across every part of each language (119 in pt, 98 in es). That last one is what
makes the merge exact — no part removed a token another counted, so there is nothing to impute.

## My paragraphs-per-article estimate was 55% high, twice, for one reason

Sampling the article JSONL at six byte offsets gave 10.37 par./art. for pt and 12.87 for es. The
real numbers are 6.78 and 8.20 — factors of 0.654 and 0.637, suspiciously consistent. The cause:
the sampler counted paragraphs for every article it saw, while the converter drops whole articles
under `--min-chars 200` before splitting them. pt Wikipedia is full of very short freguesia stubs.

Worth recording because the *shape* of this mistake is now the session's most repeated one: a
measurement that omits a filter the real pipeline applies. Previously it was `doc_freq_threshold`
measured at the wrong corpus size (twice), and paragraph counts taken from the first 10k articles
when the dump is ordered longest-first. Same class every time.

With the factor known, en projects to **~53.7M paragraphs, ~67 parts, ~4.5h** including conversion.

## The casing sweep earns its keep at scale

Stopwords came out 119 in pt and 98 in es, all function words, with **zero surviving casing twins**.
What the sweep pulled in beyond the obvious is the interesting part:

```
es:  CoMo  ParA  coMo  parA  dE  deL  eL  eN  laS  loS  LAs  DEl  LOs
pt:  CoM  PAra  DAs  DoS  doS  dA  dAs  dE  dO  dOs  foI  nA  nO  oS  UMa
```

Typos and OCR noise in the dumps. Under per-spelling detection each would have been indexed as a
content word with a high idf. pt also swept `À É à é`, which are a preposition and a verb.

## Interrupted work that looks finished is the dangerous kind

Two failures found by running this for real, both silent, both now impossible:

- **A partial conversion was reused.** A pt run stopped after 2,851,359 of 6,665,754 paragraphs;
  the next run reported "reusing existing wiki20231101-pt-paragraphs.jsonl (1.5G)" and started
  fitting on it. `prepare`'s reuse check can only see that a file is non-empty, not that its
  conversion finished. Fixed by writing to `$JSONL.partial` and renaming only on success.
- **A paragraph run would have landed on an article run's profiles.** `PROFILE_NAME` carried no
  `-paragraphs` suffix, so only the JSONL was separated. With `--resume`, a paragraph fit over a
  directory holding the previous day's 16 article-level parts would have skipped every part and
  reported success.

The general rule these share: a guard that tests *existence* cannot distinguish finished from
abandoned, and every step here reuses by existence.

## Variants: measured wrong, then deleted as an artifact

The parts each stored a variant map (11k–14k entries) and the merge unioned them, giving 21,646
keys. Deriving from the merged vocabulary instead gives **30,968** — the union is a strict subset
missing 30%. The mechanism is arithmetic: each map was built with a per-part floor of 20 documents,
so a spelling appearing ~15 times in each of 9 parts (132 corpus-wide) never cleared the floor
anywhere. `tropecar -> tropeçar`, `adocante -> adoçante`, `arrendatario -> arrendatário`,
`futeis -> fúteis` are all of that shape.

That killed the artifact rather than the merge rule: a variant map is a pure function of the
vocabulary the profile already carries, so storing one is a second copy that can go stale. Deriving
costs 0.24s over 479,245 tokens. The floor went with it — it existed to bound the stored size, and
transiently a floor of 1 costs 0.62s and 10.3 MB for 61,925 keys against 0.27s and 4.1 MB for
30,968, twice the coverage of exactly the tail a person mistypes. Note a vocabulary pruned at fit
time already imposes its own floor: at `min_ndocs=5` there, floors of 1, 2 and 5 give identical
maps.

It also removed a bug rather than fixing it. `merge_profiles` built `AppliedArtifacts` without
naming `variants`, which silently defaulted to false, so every merged profile carried a complete
map with correction switched off. With no field, that class of mistake has nowhere to live.

## Open: `head_df` does not survive a merge

**Not fixed — recorded deliberately.** Of the 11 tokens with `df > 0.05` in pt's merged
vocabulary, **7 still carry an expansion list**. The cut is applied per part at fit time, so a
token that fell below 0.05 in *some* part got a list there and the fusion kept it. Same
scale-invariance problem as `doc_freq_threshold`, in a new dimension: not corpus size this time,
but the difference between a part and the merge.

The junk and the useful lists separate cleanly, and not at 0.05:

```
0.0625  foram        -> destruídas incendiados expulsos destelhadas     noise
0.0570  até          -> perdurou pincelam Repetir Alaxarafe             noise
0.0559  seus         -> cabritos taráxaco angélica febra chupa          noise
0.0542  ele          -> privá Sontee drogou Yūichi salvara              noise
0.0518  onde         -> anotado Kronecker radicou Laplaciano            noise
0.0484  cidade       -> Jinja Aldan Surabaia Bongor Ağsu                noise
0.0466  primeiro     -> Fastnet amp Narendra Ingvar Bainimarama         noise
0.0447  sendo        -> substituída TLN acabou Inulina yondan           noise
────────────────────────────────────────────────────────────────────────────────
0.0353  Estados      -> Unidos Céticos Dançarinas Condecorados          half
0.0328  Rio          -> Janeiro Carnavalescos Afluentes Desfile         useful
0.0222  Universidade -> universidade Reitores Harvard Stanford          useful
0.0199  junho        -> maio julho setembro agosto                      useful
0.0190  música       -> folclórica erudita eletrônica eletroacústica    useful
0.0169  sistema      -> nervoso sistemas operativo límbico              useful
0.0153  jogo         -> eletrônico jogabilidade multiplayer             useful
```

So there are two separable changes whenever this is picked up: apply the cut **at merge time on the
merged counters** (the same argument that just deleted the stored variant map — what counts is the
combined vocabulary, not what each part believed), and lower the value to around **0.035** for a
corpus this size. The practical damage of leaving it is bounded: these are words nobody searches
for alone, so the junk only surfaces when a short query happens to contain one. That is why it was
left.

## Also here

`textsearch info` now takes a path to a `.zip`/directory instead of only an installed nickname. The
driver's closing message tells you to verify a fresh profile before publishing it, and verification
should not require installing it first. It also reports variants at all, which it never did.
