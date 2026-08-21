const FIT_CONFIG_TEMPLATE = """
# textsearch fit configuration -- edit, save, and exit your editor to proceed.

[input]
format = "jsonl"          # "plaintext" | "csv" | "jsonl" | "json" | "parquet"
path = "corpus.jsonl"
text_key = "text"          # ignored for format="plaintext"

[output]
dir = "./profiles"
prefix = "corpus"
batch_size = 10000          # max docs per output profile zip; 0 = unbounded (single profile)
# Skip parts whose .zip already exists instead of refitting them. Each part is written as
# soon as it is fitted, so an interrupted run leaves the finished ones behind and this
# resumes from there. Off by default: reusing a profile fitted under different settings
# silently would be worse than redoing the work.
resume = false

[normalization]
del_diac = true
del_dup = false
# Punctuation is only ~0.5% of a real vocabulary but over half of all token *occurrences*,
# so keeping it inflates numtokens/avgdoclen (which BM25 normalizes by) roughly 2x and
# spends the stopword detector's budget on "." and ",".
del_punc = true
group_num = true
group_url = true
group_usr = false
group_emo = false
lc = true

[tokenization]
nlist = [1]
mark_token_type = true

[vocabulary]
# Drop tokens appearing in fewer than this many documents. 1 keeps everything; on a real
# corpus most of the vocabulary is single-document noise (typos, IDs, foreign words) whose
# embeddings and synonyms are meaningless, and since the synonym network is an all-pairs
# search over the vocabulary, pruning it cuts that cost quadratically.
min_ndocs = 1

[stopwords]
enabled = true
doc_freq_threshold = 0.5

[encoder]
kind = "lsi"                # "lsi" | "external"
outdim = 256
scaling = "none"            # "none" | "inv_singular_values" | "singular_values"
external_path = ""          # kind="external": path to a token->vector JSON mapping
# How LSI's truncated SVD is computed. Both options are exact, so this is purely a cost
# choice. "full" builds a dense Gram matrix and takes a complete eigendecomposition --
# O(min(vocab, docs)^3) time and a squared allocation regardless of outdim -- which wins
# only for small batches. "lanczos" (ARPACK) never forms that matrix and is far faster at
# scale. "auto" picks full up to a few thousand documents per batch, lanczos above.
factorization = "auto"       # "auto" | "lanczos" | "full"

[synonyms]
k = 8
# The synonym network is an all-pairs kNN over the vocabulary. "auto" uses an approximate
# autotuned index once the vocabulary is large enough for the exact O(vocabulary^2) search
# to hurt, and the exact one below that (where exact is both fast and, well, exact).
# "always"/"never" force one or the other. The recalls are the autotuning targets.
approx = "auto"              # "auto" | "always" | "never"
construction_recall = 0.97
search_recall = 0.9

[lemmas]
algorithm = "fft"           # "fft" | "dnet" | "randsel" | "multirandsel"
num_clusters = 0             # 0 = auto (sqrt(vocsize))
# The selector also seeds the grouping, so it decides more than ties: "shortest" lets a
# short misspelling become a seed and fragment the family around it, while "most_frequent"
# seeds on the form the corpus actually uses.
selector = "most_frequent"   # "most_frequent" | "shortest" | "shortest_then_most_frequent"
# Embeddings alone give topical neighbours, not inflections ("guerra" lands next to
# "belico", not "guerras"), so surface similarity is what actually groups a lemma family and
# the lemma is elected per family. morphology_threshold is a normalized distance (lower =
# stricter); min_common_prefix additionally demands that many shared leading characters,
# which is what stops position-blind n-gram similarity from merging "abioticos" with
# "bioticos" -- set it to 0 for languages that do not inflect by suffix.
morphology = "jaccard"       # "jaccard" | "levenshtein" | "none"
morphology_threshold = 0.3
qgram = 2
min_common_prefix = 3
# "morphology_first" groups surface-similar families over the whole vocabulary and lets
# embeddings only split a family whose members mean different things; "semantic_first"
# clusters by embedding first, which fragments inflection families across clusters (~1/3 the
# coverage, ~10x slower) and is the only order that uses algorithm/num_clusters above.
# semantic_threshold is a distance under cosine, so on [0, 2]: tightening it deletes correct
# inflections rather than adding precision.
order = "morphology_first"   # "morphology_first" | "semantic_first"
semantic_threshold = 1.0
"""

"""
    edit_toml_config(; template::AbstractString=FIT_CONFIG_TEMPLATE) -> Dict

Writes `template` to a temporary file, launches `\$EDITOR` on it (falling back to `vi`
with a printed notice if `EDITOR` isn't set), blocks until the editor exits, then parses
and returns the edited file with `TOML.parsefile`. This is the "visudo-style" flow:
`fit`'s options are edited as a TOML document rather than passed as a wall of flags.
"""
function edit_toml_config(; template::AbstractString=FIT_CONFIG_TEMPLATE)
    path = tempname() * ".toml"
    write(path, template)
    editor = get(ENV, "EDITOR", "")
    if isempty(editor)
        editor = "vi"
        println(stderr, "EDITOR not set, falling back to 'vi'")
    end
    run(`$editor $path`)
    TOML.parsefile(path)
end

"""
    load_fit_config(config_path::Union{Nothing,AbstractString}) -> Dict

`config_path === nothing`: runs the visudo-style editor flow ([`edit_toml_config`](@ref)).
Otherwise reads `config_path` directly with `TOML.parsefile`, skipping the editor
entirely -- the non-interactive/scriptable escape hatch used by `--config` and by tests.
"""
function load_fit_config(config_path::Union{Nothing,AbstractString})
    config_path === nothing ? edit_toml_config() : TOML.parsefile(config_path)
end
