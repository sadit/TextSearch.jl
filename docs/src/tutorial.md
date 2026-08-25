```@meta
CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```

# Tutorial

This tutorial provides a comprehensive guide to text preprocessing, vector representation, inverted indexing, semantic dimensionality reduction, and profile management in `TextSearch.jl`, in integration with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl).

All code blocks in this tutorial are executed during documentation generation to guarantee consistency between explanations and output.

The following optional packages are used in specific sections:

```julia
] add JLD2 WordTokenizers
```

- [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl): For serializing and deserializing indexes and models to disk.
- [`WordTokenizers.jl`](https://github.com/JuliaText/WordTokenizers.jl): An external NLP tokenization library used to demonstrate pipeline extensibility.

!!! note "Namespace Resolution for `tokenize`"
    Both `TextSearch` and `WordTokenizers` export a function named `tokenize`. When both packages are imported into the same session, qualify calls explicitly (`TextSearch.tokenize(...)` or `WordTokenizers.tokenize(...)`) to prevent method ambiguity.

---

## The Reference Corpus: *The Cask of Amontillado*

To illustrate text retrieval workflows on real text, we use Edgar Allan Poe's short story *The Cask of Amontillado* (1846) from [Project Gutenberg](https://www.gutenberg.org/ebooks/1063), partitioned into its 54 constituent paragraphs. Each paragraph represents an individual document.

```@setup gutenberg
using TextSearch, SimilaritySearch

# Configure a quiet InvertedFileContext by suppressing informational progress lines during batch operations
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

---

## Vocabulary Building and Vector Models

### 1. Vocabulary Extraction

A [`Vocabulary`](@ref) processes a text corpus according to a [`TextConfig`](@ref) specification (which defaults to character normalization and word unigram extraction) and accumulates global token statistics:
- `t.occs`: Total frequency of occurrences across the entire corpus.
- `t.ndocs`: Number of distinct documents containing the token (document frequency).

```@example gutenberg
voc = Vocabulary(TextConfig(), CASK_OF_AMONTILLADO; verbose=false)
vocsize(voc), gettrainsize(voc)
```

### 2. Term Weighting and Vector Models

A [`VectorModel`](@ref) maps bag-of-words token counts into numeric weight vectors using term weighting schemes. Here, we instantiate standard Term Frequency - Inverse Document Frequency (TF-IDF) weighting using `TfWeighting()` and `IdfWeighting()`:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left( 1 + \frac{|D|}{\text{DF}(t)} \right)$$

Using [`vectorize_corpus`](@ref), the entire corpus is transformed into a collection of sparse vectors (`SparseVector{Float32, Int32}`):

```@example gutenberg
model = VectorModel(IdfWeighting(), TfWeighting(), voc)
vecs = vectorize_corpus(model, CASK_OF_AMONTILLADO)
vecs[1]
```

---

## Vocabulary Pruning

In natural language corpora, a significant fraction of terms appear only once ([hapax legomena](https://en.wikipedia.org/wiki/Hapax_legomenon)). These low-frequency terms increase vocabulary dimensionality without contributing generalizable discriminative information.

The [`filter_tokens`](@ref) function produces a pruned [`Vocabulary`](@ref) using predicate functions evaluated on token occurrences:

```@example gutenberg
hapax_count = count(t -> t.occs == 1, voc[i] for i in eachindex(voc))
vocsize(voc), hapax_count
```

```@example gutenberg
# Prune terms with fewer than 3 total occurrences across the corpus
pruned_voc = filter_tokens(t -> t.occs >= 3, voc)
vocsize(pruned_voc)
```

`filter_tokens` returns a new `Vocabulary` without modifying the original instance. In multi-document collections, filtering by document frequency (`t.ndocs`) is often preferable to discard words that are frequent within a single document but absent elsewhere:

```@example gutenberg
# Prune terms appearing in fewer than 3 distinct documents
pruned_by_docfreq = filter_tokens(t -> t.ndocs >= 3, voc)
vocsize(pruned_by_docfreq)
```

---

## Vector-Space Information Retrieval: `WeightedInvertedFile`

[`WeightedInvertedFile`](@ref) indexes sparse weight vectors and evaluates similarity under a specified distance metric, defaulting to cosine distance via `Dist.NormCosine` from `SimilaritySearch.jl`:

```@example gutenberg
wif = WeightedInvertedFile(vocsize(voc))
wctx = quietctx()
append_items!(wif, wctx, VectorDatabase(vecs))

res = knnqueue(KnnSorted, 5)
search(wif, wctx, vecs[1], res)
collect(IdView(res))
```

In the output, the first match is Document 1 at distance `0.0` (self-match), followed by the documents whose TF-IDF profiles are closest in the cosine space.

### Free-Text Querying

To query the index with arbitrary unstructured text, transform the string into a sparse vector using [`vectorize`](@ref):

```@example gutenberg
qvec = vectorize(model, "vector search library")
res = knnqueue(KnnSorted, 5)
search(wif, wctx, qvec, res)
[(id, first(CASK_OF_AMONTILLADO[id], 60)) for id in collect(IdView(res))]
```

An inverted index generates candidate documents sharing at least one token with the query vector. If a query shares minimal vocabulary with the indexed corpus, the candidate set is appropriately restricted.

---

## Vector Algebra: Dot Products, Centroids, and Normalization

Because document vectors are represented as `SparseVector` instances from `SparseArrays.jl`, standard linear algebra operations (`+`, `-`, `dot`, `norm`, `normalize!`) apply directly.

```@example gutenberg
using LinearAlgebra, SparseArrays

q_wine = vectorize(model, "amontillado wine")
q_damp = vectorize(model, "nitre damp catacombs")
norm(q_wine), norm(q_damp)
```

By default, `vectorize` scales vectors to unit Euclidean length ($\|v\|_2 = 1$). For unit-normalized vectors, the inner product equals the cosine similarity:

$$\langle u, v \rangle = \cos(\theta) = 1 - d_{\text{Cosine}}(u, v)$$

```@example gutenberg
dot(q_wine, q_damp)  # Evaluates to 0.0 because the two queries have disjoint term supports
```

```@example gutenberg
dot(vecs[51], vecs[52])  # Non-zero similarity between consecutive related paragraphs
```

### Composite Query Formulation via Centroids

To construct a composite query representing multiple thematic aspects simultaneously, compute their spherical [`centroid`](@ref):

```@example gutenberg
q_both = centroid([q_wine, q_damp])
norm(q_both)
```

```@example gutenberg
res = knnqueue(KnnSorted, 5)
search(wif, wctx, q_both, res)
[(id, first(CASK_OF_AMONTILLADO[id], 60)) for id in collect(IdView(res))]
```

### Mathematical Requirement for Input Normalization

The [`centroid`](@ref) function computes the sum of input vectors and normalizes the resultant vector to unit length:

$$c = \frac{\sum_{i=1}^m v_i}{\left\| \sum_{i=1}^m v_i \right\|_2}$$

If input vectors do not possess identical $L_2$ norms, the resultant direction is biased toward the vector with larger magnitude:

```@example gutenberg
a = sparsevec([1], [1.0f0], 2)  # Unit vector along coordinate 1
b = sparsevec([2], [1.0f0], 2)  # Unit vector along coordinate 2
centroid([a, b])                # Balanced combination with equal weights
```

```@example gutenberg
b_scaled = b * 5.0f0            # Vector along coordinate 2 with 5x magnitude
centroid([a, b_scaled])         # Direction is skewed predominantly toward coordinate 2
```

Because `vectorize` produces unit-normalized vectors by default, queries combined via `centroid` maintain equal weighting. When combining vectors generated externally, ensure `normalize!(v)` is called prior to centroid computation.

---

## Probabilistic Information Retrieval: `BM25InvertedFile`

[`BM25InvertedFile`](@ref) implements the Okapi BM25 ranking function. Unlike vector space models that evaluate geometric cosine distance over pre-computed sparse vectors, BM25 models term saturation and document length normalization directly:

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

`BM25InvertedFile` ingests raw strings, pre-tokenized structures, or bag-of-words representations directly without requiring an intermediate `VectorModel`:

```@example gutenberg
bm25idx = BM25InvertedFile(voc)
bctx = quietctx()
append_items!(bm25idx, bctx, CASK_OF_AMONTILLADO)

res = knnqueue(KnnSorted, 5)
search(bm25idx, bctx, "amontillado nitre", res)
[(id, first(CASK_OF_AMONTILLADO[id], 60)) for id in collect(IdView(res))]
```

### Selection Summary: Vector Space vs. BM25

- **[`WeightedInvertedFile`](@ref)**: Use when ranking under customized term weighting schemes (TF, IDF, TF-IDF) or when operating on general sparse feature vectors.
- **[`BM25InvertedFile`](@ref)**: Use for standard full-text document retrieval tasks benefiting from non-linear term saturation ($k_1$) and document-length penalization ($b$).

---

## Index Persistence with JLD2

All primary structures ([`Vocabulary`](@ref), [`VectorModel`](@ref), [`BM25InvertedFile`](@ref), [`WeightedInvertedFile`](@ref)) are concrete Julia types compatible with [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl) serialization:

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

Deserializing an index restores its posting lists and scoring parameters, enabling immediate query execution without retraining.

---

## Granular Segmentation and External Tokenizers

### 1. Paragraph and Sentence Segmentation

`TextSearch.jl` includes utility functions [`tokenize_paragraphs`](@ref) and [`tokenize_sentences`](@ref) to segment long documents into fine-grained passages before index creation:

```@example gutenberg
all_sentences = tokenize_sentences(CASK_OF_AMONTILLADO)
length(all_sentences), all_sentences[1]
```

```@example gutenberg
sentence_voc = Vocabulary(TextConfig(), all_sentences; verbose=false)
vocsize(sentence_voc), gettrainsize(sentence_voc)
```

External sentence splitters (such as `WordTokenizers.split_sentences`) can also be integrated into preprocessing pipelines:

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

### 2. Bypassing Redundant Normalization with `isnormalized`

When text has already undergone character normalization (e.g., lowercasing, punctuation stripping), passing `isnormalized=true` skips redundant transformation passes during vocabulary building and vectorization:

```@example gutenberg
cfg = TextConfig(normalization=NormalizationConfig(lc=true, del_punc=true))
norm_sentences = tokenize_sentences(cfg, CASK_OF_AMONTILLADO)

norm_voc = Vocabulary(cfg, norm_sentences; isnormalized=true, verbose=false)
vocsize(norm_voc)
```

### 3. Integrating External Tokenizers with `TokenizedText`

To integrate external subword tokenizers (such as BPE, WordPiece, SentencePiece, or `WordTokenizers.jl`), wrap pre-tokenized string arrays in a [`TokenizedText`](@ref) container.

When functions such as [`Vocabulary`](@ref), [`bagofwords`](@ref), [`vectorize`](@ref), and [`append_items!`](@ref) receive a `TokenizedText`, they consume the supplied tokens directly and bypass internal tokenization:

```@example gutenberg
# Tokenize documents using an external tokenizer
wt_docs = [TokenizedText(String.(WordTokenizers.tokenize(lowercase(p)))) for p in CASK_OF_AMONTILLADO]
collect(wt_docs[1])[1:8]
```

```@example gutenberg
# Construct Vocabulary directly from pre-tokenized documents
wt_voc = Vocabulary(TextConfig(), wt_docs; verbose=false)
vocsize(wt_voc)
```

---

## Dense Semantic Representations and Dimensionality Reduction

While inverted files provide exact retrieval for sparse representations, dense vector embeddings map semantically related terms and documents to continuous vector spaces.

`TextSearch.jl` implements two dimensionality reduction paradigms:
1. **Latent Semantic Indexing (LSI)**: Low-rank matrix approximation via truncated Singular Value Decomposition (SVD).
2. **Random Indexing (RI)**: Randomized projections governed by the Johnson-Lindenstrauss lemma, supporting scalar quantization and binary bit sketches.

---

### Latent Semantic Indexing (LSI)

#### Theoretical Formulation

Let $X \in \mathbb{R}^{v \times n}$ denote the term-document matrix. Truncated SVD computes the rank-$k$ approximation:

$$X \approx U_k \Sigma_k V_k^T$$

The projection matrix $P = \Sigma_k^{-1} U_k^T$ maps sparse document vectors $d \in \mathbb{R}^v$ into dense semantic coordinates $z = P d \in \mathbb{R}^k$.

#### Training and Indexing with LSI

```@example gutenberg
# Fit an LSI model with k = 16 latent dimensions
lsi = LatentSemanticIndexing(CASK_OF_AMONTILLADO; maxoutdim=16, verbose=false)
lsi
```

```@example gutenberg
# Project query string into a dense 16-dimensional vector
q_vec = vectorize(lsi, "wine vaults and connoisseur")
(length(q_vec), typeof(q_vec))
```

```@example gutenberg
# Vectorize entire corpus into a dense MatrixDatabase
lsi_db = vectorize_corpus(lsi, CASK_OF_AMONTILLADO; verbose=false)
size(lsi_db.matrix)
```

Dense LSI databases can be indexed using graph-based approximate indexes such as [`SearchGraph`](https://github.com/sadit/SimilaritySearch.jl):

```@example gutenberg
sctx = SearchGraphContext()
lsi_index = SearchGraph(Dist.NormCosine(), lsi_db)
index!(lsi_index, sctx)

res = knnqueue(KnnSorted, 3)
search(lsi_index, sctx, q_vec, res)
[(id, first(CASK_OF_AMONTILLADO[id], 60) * "...") for id in collect(IdView(res))]
```

#### Word Embeddings and Query Expansion Networks

The column vectors of the projection matrix correspond to dense word embeddings. The function [`wordvectors`](@ref) extracts these embeddings into a `MatrixDatabase`:

```@example gutenberg
W = wordvectors(lsi)
size(W.matrix)
```

The function [`query_expansion`](@ref) computes an all-pairs nearest-neighbor graph over the vocabulary embeddings to produce a semantic term-expansion network:

```@example gutenberg
net = query_expansion(lsi, 5; verbose=false)
net.query_expansion["wine"]
```

---

### Random Indexing and Quantization Pipelines

#### Properties of Random Indexing

Random Indexing assigns a fixed, pseudo-orthogonal random vector $r_t \in \mathbb{R}^k$ ($k \ll v$) to each vocabulary token $t$. A document vector is constructed incrementally as the linear combination of its constituent token vectors:

$$z = \sum_{t \in D} w(t, D) r_t$$

Advantages include:
- **Streaming Computation**: Incremental projection of new documents without requiring full-matrix SVD refactoring.
- **Distance Preservation**: Bounds metric distortion in accordance with the Johnson-Lindenstrauss lemma.
- **Compression Compatibility**: Direct integration with 8-bit scalar quantization and binary bit sketches.

#### 1. Dense Random Indexing (`Float32`)

```@example gutenberg
ri = RandomIndexing(CASK_OF_AMONTILLADO; maxoutdim=64, method=:gaussian, verbose=false)
ri_db = vectorize_corpus(ri, CASK_OF_AMONTILLADO; verbose=false)
size(ri_db.matrix)
```

#### 2. 8-Bit Scalar Quantization (`SQu8` / `SQgu8`)

Scalar quantization compresses 32-bit floating point dimensions into 8-bit unsigned integers (`UInt8`), reducing memory requirements by 4$\times$:

```@example gutenberg
using SimilaritySearch.ScalarQuant: SQu8, SQgu8

# Quantize entire corpus representation
squ8_db = vectorize_corpus(SQu8, ri, CASK_OF_AMONTILLADO; verbose=false)

# Quantize single query vector
q_squ8 = vectorize(SQu8, ri, "damp vaults and catacombs")

# Search over quantized representations using SQu8.NormCosine()
squ8_index = ExhaustiveSearch(SQu8.NormCosine(), squ8_db)
res_squ8 = knnqueue(KnnSorted, 2)
search(squ8_index, GenericContext(), q_squ8, res_squ8)
[(id, first(CASK_OF_AMONTILLADO[id], 60) * "...") for id in collect(IdView(res_squ8))]
```

#### 3. Binary Bit Sketches and Hamming Search (`BitSketch`)

`BitSketch` applies random hyperplane projections, packing projection signs into 64-bit unsigned integers (`UInt64`). Distance evaluation is computed via hardware-accelerated bitwise Hamming distance:

```@example gutenberg
# Project corpus into 512-bit binary signatures (8 × UInt64 words per document)
ri_bits = RandomIndexing(CASK_OF_AMONTILLADO; maxoutdim=512, verbose=false)
bits_db = bitsketch(ri_bits, CASK_OF_AMONTILLADO; verbose=false)
(typeof(bits_db), size(bits_db.matrix))
```

```@example gutenberg
# Query bit sketch generation
q_bits = bitsketch(ri_bits, "damp vaults and catacombs")

# Exact Hamming distance search
bit_index = ExhaustiveSearch(Dist.Bits.Hamming(), bits_db)
res_bits = knnqueue(KnnSorted, 2)
search(bit_index, GenericContext(), q_bits, res_bits)
[(id, first(CASK_OF_AMONTILLADO[id], 60) * "...") for id in collect(IdView(res_bits))]
```

---

## Portable Text Profiles (`TextProfile`)

A [`TextProfile`](@ref) encapsulates the statistical and linguistic artifacts estimated from a corpus (vocabulary frequencies, term weightings, stopword sets, lemma mappings, and expansion networks) into a portable, inspectable specification.

### Architecture: Policy vs. Artifacts

- **Policy ([`TextConfig`](@ref))**: Declarative rules governing text normalization and token extraction (independent of corpus statistics).
- **Artifacts ([`TextProfile`](@ref))**: Empirical models estimated from data. The profile derives its active tokenizer configuration directly from its declared policy and active artifacts.

```@example gutenberg
stop = Set(stopword_candidates(voc, 0.5))
lemmas = lemma_clusters(voc, W)

profile = TextProfile(model;
                      stopwords=stop, lemmas,
                      query_expansion=net.query_expansion,
                      query_expansion_distances=net.distances,
                      applied=AppliedArtifacts(stopwords=true),
                      lineage=[LineageStep(:fit; trainsize=length(CASK_OF_AMONTILLADO), outdim=16)])

(stopwords=length(profile.stopwords), lemmas=length(profile.lemmas),
 expansion=length(profile.query_expansion), base=isbase(profile))
```

The function [`fit_profile`](@ref) provides a single-call pipeline to estimate all profile components:

```@example gutenberg
oneshot = fit_profile(TextConfig(), CASK_OF_AMONTILLADO;
                      stopwords=(; doc_freq_threshold=0.5),
                      encoder=(; outdim=16), expansion=(; k=5), verbose=false)
(vocsize=vocsize(oneshot.model.voc), stopwords=length(oneshot.stopwords),
 expansion=length(oneshot.query_expansion), base=isbase(oneshot))
```

### JSON Serialization and Inspection

Profiles serialize to standard JSON files via [`save_profile`](@ref) and [`zip_profile`](@ref), enabling cross-platform inspection and version control without executing arbitrary serialized code:

```@example gutenberg
dir = mktempdir()
save_profile(dir, profile)
sort(readdir(dir))
```

### Applied vs. Carried Artifacts

Artifacts can be stored in a profile as reference data without being activated in the tokenization pipeline. The function [`with_applied`](@ref) dynamically activates or deactivates artifacts:

```@example gutenberg
p = load_profile(dir)
gettextconfig(p).pipeline, p.applied
```

```@example gutenberg
# Activate lemmatization in the profile's tokenization pipeline
q = with_applied(p; lemmas=true)
gettextconfig(q).pipeline.lemmas !== nothing, q.applied
```

### Merging Partitioned Profiles: `merge_profiles`

When processing large-scale corpora in distributed partitions, [`merge_profiles`](@ref) combines independent batch profiles into a single unified profile. Vocabulary counts across disjoint subsets are additive, guaranteeing mathematically exact inverse document frequencies:

```@example gutenberg
half = length(CASK_OF_AMONTILLADO) ÷ 2
onebatch(docs) = TextProfile(VectorModel(IdfWeighting(), TfWeighting(),
                                         Vocabulary(TextConfig(), docs; verbose=false)))
a = onebatch(CASK_OF_AMONTILLADO[1:half])
b = onebatch(CASK_OF_AMONTILLADO[half+1:end])
merged = merge_profiles([a, b])

(a=gettrainsize(a.model.voc), b=gettrainsize(b.model.voc),
 merged=gettrainsize(merged.model.voc), lineage=lineage_summary(merged))
```

### Profile Adaptation: `refit_profile`

[`refit_profile`](@ref) adapts an existing base profile to a specialized target domain using Bayesian-style updating. The base profile serves as a prior weighted by parameter `kappa` relative to empirical observations in the adaptation sample:

```@example gutenberg
tuned = refit_profile(p, CASK_OF_AMONTILLADO[1:6]; verbose=false)
(tuned=istuned(tuned), vocsize=vocsize(tuned.model.voc), lineage=lineage_summary(tuned))
```

The predicates [`isbase`](@ref) and [`istuned`](@ref) verify model provenance directly from recorded lineage history.

---

## Query Token Resolution and Spelling Normalization

When a corpus preserves case and diacritics, queries entered in un-normalized form may fail to match exact vocabulary entries:

```@example gutenberg
cased = TextConfig(normalization=NormalizationConfig(lc=false))
cvoc = Vocabulary(cased, CASK_OF_AMONTILLADO; verbose=false)
token2id(cvoc, "amontillado"), token2id(cvoc, "Amontillado")  # ID 0 indicates out-of-vocabulary
```

The function [`derive_variants`](@ref) computes an auxiliary dictionary of non-derivable surface variants:

```@example gutenberg
variants = derive_variants(cvoc)
```

The function [`resolve_query_tokens`](@ref) maps query tokens to their most probable vocabulary forms:

```@example gutenberg
r = resolve_query_tokens(cvoc, ["amontillado", "wine"], variants)
r.tokens, explain(r)
```

The [`QueryPolicy`](@ref) structure controls query resolution behavior, permitting manual disabling of spelling substitution or query expansion:

```@example gutenberg
resolve_query_tokens(cvoc, ["amontillado"], variants, QueryPolicy(correction=:off)).tokens
```

```@example gutenberg
expansion_sources(r)
```

---

## Social Media and Informal Text Processing

[`TextConfig`](@ref) includes dedicated normalization options for informal and social media text:
- `group_usr`: Maps user handles (`@username`) to the canonical placeholder token `_usr`.
- `group_url`: Maps web links (`https://...`) to `_url`.
- `group_emo`: Maps emoji characters to a unified symbol (`👾`), preventing vocabulary fragmentation across sparse emojis.
- Hashtags (`#topic`) are preserved as informative semantic tokens.

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

```@example tweets
voc = Vocabulary(cfg, tweets; verbose=false)
bm25idx = BM25InvertedFile(voc)
ctx = quietctx()
append_items!(bm25idx, ctx, tweets)

res = knnqueue(KnnSorted, 3)
search(bm25idx, ctx, "vector search library", res)
[(id, tweets[id]) for id in collect(IdView(res))]
```

BM25 scores reflect query term coverage and document-level term salience across the informal collection.

---

## Summary and API Reference

For complete function signatures and algorithmic details, consult the [TextSearch API](@ref) reference.
