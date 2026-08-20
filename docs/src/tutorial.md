```@meta
CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```

# Tutorial

This tutorial walks through building, querying, persisting, and customizing text search
indexes with `TextSearch.jl` and [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl).
Every code block on this page is a real, executed Julia session (not hand-typed
transcripts), so the output you see is always in sync with the current code.

A few sections use packages beyond `TextSearch`/`SimilaritySearch`:

```julia
] add JLD2 WordTokenizers
```

- [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl) — saving/loading indexes to disk.
- [`WordTokenizers.jl`](https://github.com/JuliaText/WordTokenizers.jl) — an
  alternative, general-purpose English tokenizer, used both to *feed* TextSearch's
  own pipeline (sentence splitting) and to *replace* it entirely.

!!! note "A naming collision to know about"
    Both `TextSearch` and `WordTokenizers` export a function named `tokenize`. If you
    `using` both, calling `tokenize` unqualified is ambiguous — Julia will tell you so.
    Qualify it (`TextSearch.tokenize(...)` / `WordTokenizers.tokenize(...)`) whenever
    both packages are loaded together, as in the examples below.

## A small corpus from Project Gutenberg

As a running example we use Edgar Allan Poe's short story *The Cask of Amontillado*
(1846), split into its 54 paragraphs — public domain, small enough to read in one
sitting, and long enough to make search results meaningful. The text comes from
[Project Gutenberg](https://www.gutenberg.org/ebooks/1063); each paragraph below is one
"document".

```@setup gutenberg
using TextSearch, SimilaritySearch

# a quiet InvertedFileContext: SimilaritySearch's default logger prints a
# timestamped progress line to stderr on every `append_items!`, which is noisy
# in a tutorial; LogList([]) with no sub-loggers silences it.
quietctx() = InvertedFileContext(logger=SimilaritySearch.LogList(SimilaritySearch.AbstractLog[]))

CASK_OF_AMONTILLADO = [
    "The thousand injuries of Fortunato I had borne as I best could, but when he ventured upon insult, I vowed revenge. You, who so well know the nature of my soul, will not suppose, however, that I gave utterance to a threat. _At length_ I would be avenged; this was a point definitely settled--but the very definitiveness with which it was resolved, precluded the idea of risk. I must not only punish, but punish with impunity. A wrong is unredressed when retribution overtakes its redresser. It is equally unredressed when the avenger fails to make himself felt as such to him who has done the wrong.",
    "It must be understood that neither by word nor deed had I given Fortunato cause to doubt my good will. I continued, as was my wont, to smile in his face, and he did not perceive that my smile _now_ was at the thought of his immolation.",
    "He had a weak point--this Fortunato--although in other regards he was a man to be respected and even feared. He prided himself on his connoisseurship in wine. Few Italians have the true virtuoso spirit. For the most part their enthusiasm is adopted to suit the time and opportunity--to practise imposture upon the British and Austrian _millionaires_. In painting and gemmary, Fortunato, like his countrymen, was a quack--but in the matter of old wines he was sincere. In this respect I did not differ from him materially: I was skillful in the Italian vintages myself, and bought largely whenever I could.",
    "It was about dusk, one evening during the supreme madness of the carnival season, that I encountered my friend. He accosted me with excessive warmth, for he had been drinking much. The man wore motley. He had on a tight-fitting parti-striped dress, and his head was surmounted by the conical cap and bells. I was so pleased to see him, that I thought I should never have done wringing his hand.",
    "I said to him--\"My dear Fortunato, you are luckily met. How remarkably well you are looking to-day! But I have received a pipe of what passes for Amontillado, and I have my doubts.\"",
    "\"How?\" said he. \"Amontillado? A pipe? Impossible! And in the middle of the carnival!\"",
    "\"I have my doubts,\" I replied; \"and I was silly enough to pay the full Amontillado price without consulting you in the matter. You were not to be found, and I was fearful of losing a bargain.\"",
    "\"As you are engaged, I am on my way to Luchesi. If any one has a critical turn, it is he. He will tell me--\"",
    "\"Luchesi cannot tell Amontillado from Sherry.\"",
    "\"And yet some fools will have it that his taste is a match for your own.\"",
    "\"My friend, no; I will not impose upon your good nature. I perceive you have an engagement. Luchesi--\"",
    "\"My friend, no. It is not the engagement, but the severe cold with which I perceive you are afflicted. The vaults are insufferably damp. They are encrusted with nitre.\"",
    "\"Let us go, nevertheless. The cold is merely nothing. Amontillado! You have been imposed upon. And as for Luchesi, he cannot distinguish Sherry from Amontillado.\"",
    "Thus speaking, Fortunato possessed himself of my arm. Putting on a mask of black silk, and drawing a _roquelaire_ closely about my person, I suffered him to hurry me to my palazzo.",
    "There were no attendants at home; they had absconded to make merry in honour of the time. I had told them that I should not return until the morning, and had given them explicit orders not to stir from the house. These orders were sufficient, I well knew, to insure their immediate disappearance, one and all, as soon as my back was turned.",
    "I took from their sconces two flambeaux, and giving one to Fortunato, bowed him through several suites of rooms to the archway that led into the vaults. I passed down a long and winding staircase, requesting him to be cautious as he followed. We came at length to the foot of the descent, and stood together on the damp ground of the catacombs of the Montresors.",
    "The gait of my friend was unsteady, and the bells upon his cap jingled as he strode.",
    "\"It is farther on,\" said I; \"but observe the white web-work which gleams from these cavern walls.\"",
    "He turned towards me, and looked into my eyes with two filmy orbs that distilled the rheum of intoxication.",
    "\"Nitre,\" I replied. \"How long have you had that cough?\"",
    "\"Ugh! ugh! ugh!--ugh! ugh! ugh!--ugh! ugh! ugh!--ugh! ugh! ugh!--ugh! ugh! ugh!\"",
    "My poor friend found it impossible to reply for many minutes.",
    "\"Come,\" I said, with decision, \"we will go back; your health is precious. You are rich, respected, admired, beloved; you are happy, as once I was. You are a man to be missed. For me it is no matter. We will go back; you will be ill, and I cannot be responsible. Besides, there is Luchesi--\"",
    "\"Enough,\" he said; \"the cough is a mere nothing; it will not kill me. I shall not die of a cough.\"",
    "\"True--true,\" I replied; \"and, indeed, I had no intention of alarming you unnecessarily--but you should use all proper caution. A draught of this Medoc will defend us from the damps.\"",
    "Here I knocked off the neck of a bottle which I drew from a long row of its fellows that lay upon the mould.",
    "\"Drink,\" I said, presenting him the wine.",
    "He raised it to his lips with a leer. He paused and nodded to me familiarly, while his bells jingled.",
    "\"I drink,\" he said, \"to the buried that repose around us.\"",
    "\"These vaults,\" he said, \"are extensive.\"",
    "\"The Montresors,\" I replied, \"were a great and numerous family.\"",
    "\"A huge human foot d'or, in a field azure; the foot crushes a serpent rampant whose fangs are imbedded in the heel.\"",
    "The wine sparkled in his eyes and the bells jingled. My own fancy grew warm with the Medoc. We had passed through walls of piled bones, with casks and puncheons intermingling, into the inmost recesses of catacombs. I paused again, and this time I made bold to seize Fortunato by an arm above the elbow.",
    "\"The nitre!\" I said; \"see, it increases. It hangs like moss upon the vaults. We are below the river's bed. The drops of moisture trickle among the bones. Come, we will go back ere it is too late. Your cough--\"",
    "\"It is nothing,\" he said; \"let us go on. But first, another draught of the Medoc.\"",
    "I broke and reached him a flagon of De Grave. He emptied it at a breath. His eyes flashed with a fierce light. He laughed and threw the bottle upwards with a gesticulation I did not understand.",
    "I looked at him in surprise. He repeated the movement--a grotesque one.",
    "\"It is this,\" I answered, producing a trowel from beneath the folds of my _roquelaire_.",
    "\"You jest,\" he exclaimed, recoiling a few paces. \"But let us proceed to the Amontillado.\"",
    "\"Be it so,\" I said, replacing the tool beneath the cloak and again offering him my arm. He leaned upon it heavily. We continued our route in search of the Amontillado. We passed through a range of low arches, descended, passed on, and descending again, arrived at a deep crypt, in which the foulness of the air caused our flambeaux rather to glow than flame.",
    "At the most remote end of the crypt there appeared another less spacious. Its walls had been lined with human remains, piled to the vault overhead, in the fashion of the great catacombs of Paris. Three sides of this interior crypt were still ornamented in this manner. From the fourth side the bones had been thrown down, and lay promiscuously upon the earth, forming at one point a mound of some size. Within the wall thus exposed by the displacing of the bones, we perceived a still interior recess, in depth about four feet in width three, in height six or seven. It seemed to have been constructed for no especial use within itself, but formed merely the interval between two of the colossal supports of the roof of the catacombs, and was backed by one of their circumscribing walls of solid granite.",
    "It was in vain that Fortunato, uplifting his dull torch, endeavoured to pry into the depth of the recess. Its termination the feeble light did not enable us to see.",
    "\"Proceed,\" I said; \"herein is the Amontillado. As for Luchesi--\"",
    "\"He is an ignoramus,\" interrupted my friend, as he stepped unsteadily forward, while I followed immediately at his heels. In an instant he had reached the extremity of the niche, and finding his progress arrested by the rock, stood stupidly bewildered. A moment more and I had fettered him to the granite. In its surface were two iron staples, distant from each other about two feet, horizontally. From one of these depended a short chain, from the other a padlock. Throwing the links about his waist, it was but the work of a few seconds to secure it. He was too much astounded to resist. Withdrawing the key I stepped back from the recess.",
    "\"Pass your hand,\" I said, \"over the wall; you cannot help feeling the nitre. Indeed, it is _very_ damp. Once more let me _implore_ you to return. No? Then I must positively leave you. But I must first render you all the little attentions in my power.\"",
    "\"The Amontillado!\" ejaculated my friend, not yet recovered from his astonishment.",
    "As I said these words I busied myself among the pile of bones of which I have before spoken. Throwing them aside, I soon uncovered a quantity of building stone and mortar. With these materials and with the aid of my trowel, I began vigorously to wall up the entrance of the niche.",
    "I had scarcely laid the first tier of the masonry when I discovered that the intoxication of Fortunato had in a great measure worn off. The earliest indication I had of this was a low moaning cry from the depth of the recess. It was _not_ the cry of a drunken man. There was then a long and obstinate silence. I laid the second tier, and the third, and the fourth; and then I heard the furious vibrations of the chain. The noise lasted for several minutes, during which, that I might hearken to it with the more satisfaction, I ceased my labours and sat down upon the bones. When at last the clanking subsided, I resumed the trowel, and finished without interruption the fifth, the sixth, and the seventh tier. The wall was now nearly upon a level with my breast. I again paused, and holding the flambeaux over the mason-work, threw a few feeble rays upon the figure within.",
    "A succession of loud and shrill screams, bursting suddenly from the throat of the chained form, seemed to thrust me violently back. For a brief moment I hesitated--I trembled. Unsheathing my rapier, I began to grope with it about the recess; but the thought of an instant reassured me. I placed my hand upon the solid fabric of the catacombs, and felt satisfied. I reapproached the wall; I replied to the yells of him who clamoured. I re-echoed--I aided--I surpassed them in volume and in strength. I did this, and the clamourer grew still.",
    "It was now midnight, and my task was drawing to a close. I had completed the eighth, the ninth, and the tenth tier. I had finished a portion of the last and the eleventh; there remained but a single stone to be fitted and plastered in. I struggled with its weight; I placed it partially in its destined position. But now there came from out the niche a low laugh that erected the hairs upon my head. It was succeeded by a sad voice, which I had difficulty in recognizing as that of the noble Fortunato. The voice said--",
    "\"Ha! ha! ha!--he! he! he!--a very good joke indeed--an excellent jest. We shall have many a rich laugh about it at the palazzo--he! he! he!--over our wine--he! he! he!\"",
    "\"He! he! he!--he! he! he!--yes, the Amontillado. But is it not getting late? Will not they be awaiting us at the palazzo, the Lady Fortunato and the rest? Let us be gone.\"",
    "But to these words I hearkened in vain for a reply. I grew impatient. I called aloud--",
    "No answer still. I thrust a torch through the remaining aperture and let it fall within. There came forth in reply only a jingling of the bells. My heart grew sick on account of the dampness of the catacombs. I hastened to make an end of my labour. I forced the last stone into its position; I plastered it up. Against the new masonry I re-erected the old rampart of bones. For the half of a century no mortal has disturbed them. _In pace requiescat!_",
]
```

```@example gutenberg
length(CASK_OF_AMONTILLADO), CASK_OF_AMONTILLADO[1]
```

### Building a vocabulary and a vector model

[`Vocabulary`](@ref) parses the corpus once and accumulates per-token statistics
([`TextConfig`](@ref)`()` defaults to word unigrams). A [`VectorModel`](@ref) then turns
that vocabulary into a weighting scheme — here, classic TF-IDF — and
[`vectorize_corpus`](@ref) applies it to every paragraph, producing one sparse
`SparseVector` per document.

```@example gutenberg
voc = Vocabulary(TextConfig(), CASK_OF_AMONTILLADO; verbose=false)
vocsize(voc), trainsize(voc)
```

```@example gutenberg
model = VectorModel(IdfWeighting(), TfWeighting(), voc)
vecs = vectorize_corpus(model, CASK_OF_AMONTILLADO)
vecs[1]
```

### Pruning the vocabulary by minimum frequency

`798` tokens out of a ~4000-word story is a lot — most of them are one-off words
([hapax legomena](https://en.wikipedia.org/wiki/Hapax_legomenon)) that add noise more
than signal. Each entry of a [`Vocabulary`](@ref) carries its own `occs` (total
occurrence count across the corpus) and `ndocs` (number of documents it appears in), so
[`filter_tokens`](@ref) can prune by either:

```@example gutenberg
hapax_count = count(t -> t.occs == 1, voc[i] for i in eachindex(voc))
vocsize(voc), hapax_count
```

```@example gutenberg
pruned_voc = filter_tokens(t -> t.occs >= 3, voc)
vocsize(pruned_voc)
```

`filter_tokens` returns a brand new [`Vocabulary`](@ref) — `voc` itself is untouched, so
the rest of this tutorial keeps using the original, unpruned vocabulary. To actually
build a model on the pruned vocabulary instead, just use `pruned_voc` in place of `voc`
from here on (e.g. `VectorModel(IdfWeighting(), TfWeighting(), pruned_voc)`); words
below the frequency cutoff are treated as out-of-vocabulary from then on, the same as
any other unseen word.

Filtering by document frequency (`t.ndocs`) instead of raw occurrence count is often a
better cutoff for longer/multi-document corpora — it discards words that are common
within a single document but never recur elsewhere, which raw frequency alone wouldn't
catch:

```@example gutenberg
pruned_by_docfreq = filter_tokens(t -> t.ndocs >= 3, voc)
vocsize(pruned_by_docfreq)
```

### Searching with a raw inverted file (vector-space ranking)

[`WeightedInvertedFile`](@ref) indexes the weight vectors directly and ranks by a
distance over them — cosine here, via `NormCosine` (SimilaritySearch's cosine distance, re-exported by TextSearch). This is the same kind of
index you'd use for any sparse vector search, not just text.

```@example gutenberg
wif = WeightedInvertedFile(vocsize(voc))
wctx = quietctx()
append_items!(wif, wctx, VectorDatabase(vecs))

res = knnqueue(KnnSorted, 5)
search(wif, wctx, vecs[1], res)
collect(IdView(res))
```

The first hit is paragraph 1 itself (distance 0 — a document is always its own nearest
neighbor); the rest are the paragraphs whose TF-IDF vectors are closest to it. Querying
with free text instead of an existing document's vector works the same way, through
[`vectorize`](@ref):

```@example gutenberg
qvec = vectorize(model, "vector search library")
res = knnqueue(KnnSorted, 5)
search(wif, wctx, qvec, res)
[(id, first(CASK_OF_AMONTILLADO[id], 60)) for id in collect(IdView(res))]
```

Only one hit came back even though we asked for 5 — an inverted file can only rank
documents that share at least one token with the query, and here just one paragraph
happens to contain "search". Unsurprisingly, a 19th-century short story has nothing to
do with vector search libraries anyway; every score here is essentially noise. Try a
query drawn from the story itself, like `"amontillado nitre"` or `"trowel wall"`, to
see closer, more meaningful matches.

### Vector arithmetic: dot products, centroids, and normalization

Since paragraph and query vectors are `SparseVector`s (from `SparseArrays.jl`), ordinary
`LinearAlgebra`/`SparseArrays` operations work on them directly — `+`, `-`, [`dot`](@ref),
`norm`, [`normalize!`](@ref), and scalar `*`/`/` all work out of the box, with no extra
glue code from TextSearch. Two things make this useful: comparing documents/queries
directly via [`dot`](@ref), and building a query that represents *more than one* idea at
once.

```@example gutenberg
using LinearAlgebra, SparseArrays

q_wine = vectorize(model, "amontillado wine")
q_damp = vectorize(model, "nitre damp catacombs")
norm(q_wine), norm(q_damp)
```

`vectorize` normalizes its output to unit length by default (`normalize=true`) — this is
what makes `dot` directly meaningful as a similarity score: for unit ("spherical")
vectors, the dot product *is* the cosine similarity, bounded the same way cosine
similarity is. It also matches what [`WeightedInvertedFile`](@ref) itself assumes —
its distance is a cheap running dot-product sum (see `Dist.NormCosine`'s docstring),
valid only when the vectors being compared are already unit length.

```@example gutenberg
dot(q_wine, q_damp)
```

Zero — these two queries share no vocabulary at all, so as far as the dot product is
concerned they're unrelated (not literally "opposite", just orthogonal). Compare that to
two paragraphs that are actually about the same scene:

```@example gutenberg
dot(vecs[51], vecs[52])  # two consecutive paragraphs of Fortunato's manic "ha! ha!" laughter
```

To search for *both* ideas at once — say, a query that's part "the wine", part "the
damp vaults" — combine the two query vectors into their [`centroid`](@ref) and search
with that, instead of running two separate queries and merging results by hand:

```@example gutenberg
q_both = centroid([q_wine, q_damp])
norm(q_both)
```

```@example gutenberg
res = knnqueue(KnnSorted, 5)
search(wif, wctx, q_both, res)
[(id, first(CASK_OF_AMONTILLADO[id], 60)) for id in collect(IdView(res))]
```

The results blend both themes, rather than only ever matching one or the other.

#### Why `normalize!` matters here

[`centroid`](@ref) normalizes its *output*, but that alone doesn't make a fair blend —
it fixes the final vector's length, not the direction that a plain sum already baked
in. If the inputs going in aren't themselves unit vectors, whichever one happens to have
the larger magnitude dominates the sum, and the final `normalize!` just rescales that
already-skewed direction to length 1. A minimal example makes this concrete:

```@example gutenberg
a = sparsevec([1], [1.0f0], 2)  # a unit vector, pointing along "axis 1"
b = sparsevec([2], [1.0f0], 2)  # a unit vector, pointing along "axis 2"
centroid([a, b])                # evenly split between both directions, as expected
```

```@example gutenberg
b_scaled = b * 5.0f0   # same direction as `b`, but 5x the magnitude
norm(b_scaled)
```

```@example gutenberg
centroid([a, b_scaled])  # direction is pulled almost entirely toward b, not an even blend
```

`vectorize`'s default `normalize=true` is precisely what saves you from this: every
vector TextSearch itself produces already lies on the unit sphere, so summing any
number of them and normalizing the result gives a genuinely even blend, as in
`q_both` above. The failure mode above only bites when vectors come from somewhere
`vectorize` didn't touch — built with `normalize=false`, assembled by hand, or imported
from a different pipeline entirely. The fix is always the same: call [`normalize!`](@ref)
on each vector individually before combining them, so every input to a centroid is on
equal footing before it's summed.

### Searching with BM25 (probabilistic ranking)

[`BM25InvertedFile`](@ref) is a different index entirely: instead of building explicit
weight vectors, it indexes the corpus's bags of words and a [`BM25Scorer`](@ref)
directly, ranking by the Okapi BM25 formula. There's no separate `VectorModel`/
`vectorize_corpus` step — `append_items!` takes raw text (or [`TokenizedText`](@ref),
or a pre-computed [`BOW`](@ref)) and computes everything it needs from `voc`.

```@example gutenberg
bm25idx = BM25InvertedFile(voc)
bctx = quietctx()
append_items!(bm25idx, bctx, CASK_OF_AMONTILLADO)

res = knnqueue(KnnSorted, 5)
search(bm25idx, bctx, "amontillado nitre", res)
[(id, first(CASK_OF_AMONTILLADO[id], 60)) for id in collect(IdView(res))]
```

Both index types answer top-k queries the same way (`append_items!`/`push_item!` to
build, `search` to query), so switching between them is mostly a matter of which one
matches your ranking needs: `WeightedInvertedFile` for vector-space similarity over any
weighting scheme you've built, `BM25InvertedFile` when you want BM25's document-length
normalization and term-saturation behavior without hand-building vectors first.

## Saving and loading indexes with JLD2

Every type used above — [`Vocabulary`](@ref), [`VectorModel`](@ref),
[`BM25InvertedFile`](@ref), [`WeightedInvertedFile`](@ref) — is a plain Julia struct, so
[`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl) can save and load them directly with no
special glue code.

```@example gutenberg
using JLD2

path = tempname() * ".jld2"
jldsave(path; voc, model, bm25idx)
```

```@example gutenberg
loaded = load(path)
voc2, model2, bm25idx2 = loaded["voc"], loaded["model"], loaded["bm25idx"]
vectorize(model2, CASK_OF_AMONTILLADO[1]) == vecs[1]
```

```@example gutenberg
res = knnqueue(KnnSorted, 5)
search(bm25idx2, quietctx(), "amontillado nitre", res)
collect(IdView(res))
```

The reloaded `BM25InvertedFile` answers the same query with the same ranking as the
original — nothing needs to be rebuilt or refit.

## Working with Paragraphs, Sentences, and External Tokenizers

`TextSearch`'s own tokenizer ([`TextConfig`](@ref)/[`tokenize`](@ref)) is fast, thread-safe, and tuned for domain-agnostic and informal text. For long documents or specialised NLP tasks, you may want to segment text into paragraphs or sentences, skip redundant text normalization, or replace TextSearch's tokenizer entirely with an **external tokenizer**.

### Paragraph and Sentence Tokenization

TextSearch provides built-in helper functions [`tokenize_paragraphs`](@ref) and [`tokenize_sentences`](@ref) to split long documents or text corpora into finer-grained chunks prior to vocabulary building or indexing:

```@example gutenberg
# Splitting long text into sentences using TextSearch's built-in sentence tokenizer:
all_sentences = tokenize_sentences(CASK_OF_AMONTILLADO)
length(all_sentences), all_sentences[1]
```

```@example gutenberg
sentence_voc = Vocabulary(TextConfig(), all_sentences; verbose=false)
vocsize(sentence_voc), trainsize(sentence_voc)
```

You can also use external sentence segmenters like `WordTokenizers.split_sentences` as a preprocessing step:

```@example gutenberg
using WordTokenizers

sentences = String[]
for paragraph in CASK_OF_AMONTILLADO
    for s in split_sentences(paragraph)
        push!(sentences, String(s))
    end
end

length(sentences), sentences[1]
```

### Avoiding Redundant Normalization with `isnormalized`

When text has already been normalized (for example, during paragraph or sentence extraction), you can pass `isnormalized=true` to [`tokenize`](@ref), [`Vocabulary`](@ref), [`bagofwords`](@ref), or [`vectorize`](@ref) to skip character-level case-folding, diacritics removal, and regex preprocessing:

```@example gutenberg
cfg = TextConfig(normalization=NormalizationConfig(lc=true, del_punc=true))
norm_sentences = tokenize_sentences(cfg, CASK_OF_AMONTILLADO)

# Pass isnormalized=true to skip repeating the normalization step:
norm_voc = Vocabulary(cfg, norm_sentences; isnormalized=true, verbose=false)
vocsize(norm_voc)
```

### Integrating External Tokenizers with `TokenizedText`

If you want to use an **external tokenizer** (such as `WordTokenizers.jl`, HuggingFace / BPE / WordPiece subword tokenizers, spaCy, or custom regex tokenizers), wrap your token list in a [`TokenizedText`](@ref).

`TokenizedText` is TextSearch's universal contract for pre-tokenized documents:
[`Vocabulary`](@ref), [`bagofwords`](@ref), [`vectorize`](@ref), and [`append_items!`](@ref) all recognize `TokenizedText` and skip TextSearch's internal normalization and tokenization steps entirely, consuming your external tokens as-is.

```@example gutenberg
# Tokenize using WordTokenizers.jl (or any external tokenizer):
wt_docs = [TokenizedText(String.(WordTokenizers.tokenize(lowercase(p)))) for p in CASK_OF_AMONTILLADO]
collect(wt_docs[1])[1:8]
```

```@example gutenberg
# Build Vocabulary directly from external TokenizedText documents:
wt_voc = Vocabulary(TextConfig(), wt_docs; verbose=false)
vocsize(wt_voc)
```

`TextConfig()` is still passed when constructing `Vocabulary` — TextSearch retains it for later query processing — but because every document arrives as a `TokenizedText`, none of `TextConfig`'s tokenization rules alter how your documents were split; the tokenization was performed entirely by your external tokenizer.

## Dense Semantic Representations & Dimensionality Reduction

While inverted files (like `InvertedFile` and `BM25InvertedFile`) excel at sparse keyword-based retrieval, many natural language processing workflows benefit from **dense semantic embeddings** or **compact binary sketches**.

`TextSearch.jl` provides two complementary techniques for dimensionality reduction and dense indexing:
1. **Latent Semantic Indexing (LSI)**: Low-rank matrix factorization via truncated SVD.
2. **Random Indexing (RI)**: Fast projection via random matrices, with direct pipelines for 8-bit scalar quantization (`SQu8`, `SQgu8`) and binary bit sketches (`BitSketch`).

---

### Latent Semantic Indexing (LSI)

#### Where is LSI useful?
- **Synonymy and polysemy resolution**: By projecting the term-document matrix onto its principal singular vectors, LSI groups words that frequently co-occur into shared latent semantic dimensions. Documents using different words for the same concept (e.g., *"wine"* and *"amontillado"*) are mapped close together in the dense space.
- **Noise reduction and compact representations**: Reduces large vocabularies (e.g., tens of thousands of terms) into a dense, low-dimensional space (typically 64 to 300 dimensions, default `maxoutdim=128`).
- **Dense index compatibility**: Creates dense `MatrixDatabase{Matrix{Float32}}` collections that can be indexed with approximate nearest-neighbor graph structures like [`SearchGraph`](https://github.com/sadit/SimilaritySearch.jl).

#### Building and Querying an LSI Model

You can construct a [`LatentSemanticIndexing`](@ref) model directly from a [`VectorModel`](@ref) and a training corpus, or via convenience constructors:

```@example gutenberg
# Train an LSI model with 16 latent dimensions on our Gutenberg corpus
lsi = LatentSemanticIndexing(CASK_OF_AMONTILLADO; maxoutdim=16, verbose=false)
lsi
```

```@example gutenberg
# Project a single query document into a dense Float32 vector:
q_vec = vectorize(lsi, "wine vaults and connoisseur")
(length(q_vec), typeof(q_vec))
```

```@example gutenberg
# Vectorize the entire corpus into a dense MatrixDatabase in parallel:
lsi_db = vectorize_corpus(lsi, CASK_OF_AMONTILLADO; verbose=false)
size(lsi_db.matrix)
```

Now we can build a graph index (`SearchGraph`) or exact search (`ExhaustiveSearch`) from `SimilaritySearch.jl` using cosine distance:

```@example gutenberg
# Index and search the dense LSI space:
sctx = SearchGraphContext()
lsi_index = SearchGraph(Dist.NormCosine(), lsi_db)
index!(lsi_index, sctx)

res = knnqueue(KnnSorted, 3)
search(lsi_index, sctx, q_vec, res)
[(id, first(CASK_OF_AMONTILLADO[id], 60) * "...") for id in collect(IdView(res))]
```

#### Word Embeddings and a Synonym Network

`lsi.P` is a `(outdim(lsi), vocsize(lsi))` projection matrix -- column `t` is already the LSI
embedding of vocabulary token `t` (a document's vector is just a weighted sum of its tokens'
columns), so the same trained model doubles as a source of **word embeddings**, with no extra
training needed. [`wordvectors`](@ref) returns them as a `MatrixDatabase`, ready for the same
kind of nearest-neighbor search used above for documents:

```@example gutenberg
W = wordvectors(lsi)
size(W.matrix)
```

Running [`allknn`](https://sadit.github.io/SimilaritySearch.jl/dev/) over that word-embedding
space is exactly how the "synonymy resolution" mentioned earlier becomes concrete: words that
tend to co-occur in similar contexts end up with nearby embeddings. [`synonyms`](@ref) wraps this
into a `token => [(neighbor, distance), ...]` network in one call:

```@example gutenberg
net = synonyms(lsi, 5; verbose=false)
net["wine"]
```

For a small demo corpus like this one, don't expect polished synonym pairs -- a handful of short
paragraphs isn't enough text for the co-occurrence statistics LSI relies on to fully separate
content words from frequent function words. On a real corpus (thousands of documents, a pruned
vocabulary), the same call is a quick way to get a first synonym/related-terms network without
training a dedicated word-embedding model.

---

### Random Indexing (RI) & Quantization Pipelines

#### Where is Random Indexing useful?
- **Streaming / incremental scenarios without global matrix factorization**: Unlike LSI (which computes SVD across an entire static corpus), Random Indexing assigns fixed, pseudo-orthogonal random projection vectors to each vocabulary term. New documents can be projected immediately on the fly without re-training or re-factoring any matrix.
- **Fast computation with Johnson-Lindenstrauss guarantees**: Preserves pairwise cosine distances with high probability while being computationally lightweight (`:gaussian`, `:qr`, or sparse ternary `:sparse_random`).
- **Extreme memory compression and hardware acceleration**:
  - **`SQu8` / `SQgu8` Scalar Quantization**: Compresses embeddings from 32-bit `Float32` down to 8-bit `UInt8` (a 4× memory reduction) while retaining search accuracy.
  - **`BitSketch` (SimHash)**: Compresses projections into 64-bit packed bit words (`UInt64`), turning similarity search into lightning-fast **Hamming distance** evaluations using CPU `POPCNT` and `XOR` instructions (`Dist.Bits.Hamming()`).

#### 1. Dense Random Indexing (`Float32`)

```@example gutenberg
# Build a Random Indexing model from the corpus (default maxoutdim=1024; using 64 here for illustration)
ri = RandomIndexing(CASK_OF_AMONTILLADO; maxoutdim=64, method=:gaussian, verbose=false)
ri
```

```@example gutenberg
# Vectorize corpus into dense Float32 MatrixDatabase
ri_db = vectorize_corpus(ri, CASK_OF_AMONTILLADO; verbose=false)
size(ri_db.matrix)
```

#### 2. 8-Bit Scalar Quantization (`SQu8` & `SQgu8`)

For memory-constrained environments, `vectorize_corpus` can directly produce quantized databases:

```@example gutenberg
using SimilaritySearch.ScalarQuant: SQu8, SQgu8

# Per-column 8-bit quantization (SQu8):
squ8_db = vectorize_corpus(SQu8, ri, CASK_OF_AMONTILLADO; verbose=false)

# Single query quantized to SQu8:
q_squ8 = vectorize(SQu8, ri, "damp vaults and catacombs")

# Search using SQu8.NormCosine():
squ8_index = ExhaustiveSearch(SQu8.NormCosine(), squ8_db)
res_squ8 = knnqueue(KnnSorted, 2)
search(squ8_index, GenericContext(), q_squ8, res_squ8)
[(id, first(CASK_OF_AMONTILLADO[id], 60) * "...") for id in collect(IdView(res_squ8))]
```

#### 3. Binary BitSketches & Hamming Search (`BitSketch`)

`BitSketch` uses SimHash-style sign-packing to generate compact binary fingerprints:

```@example gutenberg
# Create a 512-dimension RI model (packed into 8 UInt64 words (512 bits) per document)
ri_bits = RandomIndexing(CASK_OF_AMONTILLADO; maxoutdim=512, verbose=false)

# Vectorize entire corpus into packed bit words (MatrixDatabase{Matrix{UInt64}}):
bits_db = bitsketch(ri_bits, CASK_OF_AMONTILLADO; verbose=false)
(typeof(bits_db), size(bits_db.matrix))
```

```@example gutenberg
# Query bit sketch:
q_bits = bitsketch(ri_bits, "damp vaults and catacombs")

# Ultra-fast Hamming distance search:
bit_index = ExhaustiveSearch(Dist.Bits.Hamming(), bits_db)
res_bits = knnqueue(KnnSorted, 2)
search(bit_index, GenericContext(), q_bits, res_bits)
[(id, first(CASK_OF_AMONTILLADO[id], 60) * "...") for id in collect(IdView(res_bits))]
```

## A small tweet-like corpus

`TextConfig` has several options aimed specifically at short, informal, social-media
text: grouping `@mentions`, URLs, and emoji into single normalized tokens instead of
leaving them as noisy character soup. The messages below are a small illustrative set
written to exercise these options (not scraped from a live feed, so the example needs
no network access and no data-license considerations).

```@example tweets
using TextSearch, SimilaritySearch

quietctx() = InvertedFileContext(logger=SimilaritySearch.LogList(SimilaritySearch.AbstractLog[]))

tweets = [
    "Just landed in Mexico City!! 🎉 cant wait to try the tacos @VisitMexico #travel",
    "Ugh, stuck in traffic again on the highway :( #mondayblues",
    "New paper on approximate similarity search is out! check it out https://example.org/paper",
    "@juli_ai loved your talk on vector databases today, so insightful #ai #ml",
    "Rainy day, perfect for reading a good book ☕📚",
    "Why does @united keep cancelling flights?? this is the third time this month #travelfail",
    "Excited to announce our new open source vector search release! https://github.com/example/repo #julialang",
    "lol this meme is too real 😂😂😂 #mood",
    "Can anyone recommend a good vector search library for Julia? asking for a friend @julialang",
    "Beautiful sunset over the bay tonight 🌅 #nofilter",
]

cfg = TextConfig(normalization=NormalizationConfig(group_usr=true, group_url=true, group_emo=true, del_punc=false))
collect(TextSearch.tokenize(cfg, tweets[1]))
```

`@VisitMexico` collapsed to `_usr`, and the 🎉 emoji collapsed to `👾` — with
`group_emo=true`, every emoji character is replaced by this single placeholder glyph
before tokenization, so any emoji becomes the same token instead of each distinct emoji
being its own rare, one-off token. `#travel` stayed intact — hashtags are treated as
regular content, not stripped, since they usually carry meaning.

```@example tweets
voc = Vocabulary(cfg, tweets; verbose=false)
bm25idx = BM25InvertedFile(voc)
ctx = quietctx()
append_items!(bm25idx, ctx, tweets)

res = knnqueue(KnnSorted, 3)
search(bm25idx, ctx, "vector search library", res)
[(id, tweets[id]) for id in collect(IdView(res))]
```

The top matches are exactly the three tweets that actually mention vector search —
BM25 ranks them by how much of the query they cover and how rare/salient those terms
are across the small corpus.

## Next steps

See the [TextSearch API](@ref) page for the full reference — every function and type used above
(and many more, including the lower-level building blocks in `TextSearch.Intersections`
and `TextSearch.InvertedFiles`) is documented there with its own runnable example.
