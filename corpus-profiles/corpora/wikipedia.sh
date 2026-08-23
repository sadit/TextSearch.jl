#!/usr/bin/env bash
#
# Build TextSearch profiles from a Wikipedia dump, via the HuggingFace
# `wikimedia/wikipedia` dataset (https://huggingface.co/datasets/wikimedia/wikipedia).
#
# That dataset ships one parquet config per (snapshot, language), e.g. `20231101.es`, with
# columns id/url/title/text -- one row per article, already stripped of wiki markup.
#
# Usage:
#   corpora/wikipedia.sh --lang es
#   corpora/wikipedia.sh --lang es --limit 20000          # smoke test on a slice
#   corpora/wikipedia.sh --lang de --snapshot 20231101
#   corpora/wikipedia.sh --lang es --parts 32             # more, smaller parts
#   corpora/wikipedia.sh --lang es --resume               # continue an interrupted run
#   corpora/wikipedia.sh --lang es --steps fetch          # download only
#
# Produces profiles/wiki<SNAPSHOT>-<LANG>/wiki<SNAPSHOT>-<LANG>-NNNN.zip -- one independent
# profile per part (see --parts), which `textsearch merge` folds back into one.
#
# Only downloads and fits; nothing is published. Verify a profile (textsearch info, a few
# textsearch search queries) before attaching it to a release -- see ../README.md.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

HF_REPO="wikimedia/wikipedia"
HF_API="https://huggingface.co/api/datasets/$HF_REPO"
HF_RESOLVE="https://huggingface.co/datasets/$HF_REPO/resolve/main"

LANG_CODE=""
SNAPSHOT=""
LIMIT=0
PARTS=16
BATCH_SIZE=""      # empty: derived from PARTS once the document count is known
RESUME=false
OUTDIM=256
# Wikipedia is edited prose, so its diacritics are trustworthy and worth keeping: "año" and
# "ano" are different words. (Queries typed without accents are a query-side problem, to be
# handled by spelling correction rather than by destroying the distinction in the profile.)
DEL_DIAC=false
DEL_PUNC=true
# Keep casing, for the same reason as `del_diac`: folding destroys sense distinctions the corpus
# actually makes, and it cannot be undone once fitted. Measured on 272,466 Spanish paragraphs,
# keeping it costs x1.12 vocabulary and recovers senses no statistical method reached -- `granada`
# unfolded returns the heraldic charge (`gules azur bordura`) as well as the city, likewise `cuba`
# (the barrel), `leon` (the animal), `sol`, `palma`, `iglesia`. A query typed folded gets back to
# the corpus spellings through the profile's variant map, which `fit` derives here; a profile
# fitted with `--lc` has no such recourse. Pass --lc for a folded profile.
LC=false
QUERY_EXPANSION_K=8
# UNEXAMINED, and it silently discards a lot: verified against the parquet row counts, 200
# characters drops 62,946 Spanish articles (3.4%), 323,825 English (5.1%) and 129,605
# Portuguese (11.7%) -- Portuguese Wikipedia is full of very short freguesia stubs. Unlike
# `del_diac` or `min_ndocs` this value was never measured or argued for, and it biases
# `avgdoclen` upward by dropping the shortest documents, which is the quantity BM25 normalizes
# every document length by. A profile fitted this way therefore tells BM25 the average document
# is longer than it is in the corpus it will index. Set it to 0 to keep everything.
MIN_CHARS=200
MIN_NDOCS=5
# One document per PARAGRAPH instead of per article, each prefixed with the article title.
# Measured on 10,000 Spanish articles: 272,466 paragraph documents, and document frequency over
# paragraphs separates real stopwords from Wikipedia artifacts by a factor of 55 where article
# frequency puts them within 1.2 (`de` 0.998 -> 0.934 against `enlaces` 0.826 -> 0.017). A fit
# under identical settings then detects 11 stopwords -- all function words -- against 46 that
# included `enlaces externos referencias vease anio parte dos`. Query expansion also improve, since
# co-occurring in an 87-token paragraph is a far tighter semantic window than in a 2,300-token
# article: `planeta` goes from `larense protoplanetas haumea verrier` to `marte saturno neptuno
# urano jupiter`. See ../README.md.
SPLIT_PARAGRAPHS=false
# Minimum tokens (whitespace words) a record must have. Replaces --min-chars as the useful floor
# once paragraphs are the unit: "Referencias" is 11 characters and no content, "no fue asi" is 10
# and a sentence. Bare section headings are folded into the following paragraph rather than
# dropped, so this discards very little -- 1,438 records out of 272,466, against 66,363 when the
# headings were dropped instead.
MIN_TOKENS=4
STOPWORDS=true
DOC_FREQ_THRESHOLD=""   # empty: per-language default from the table below
LEMMA_ALG=fft
LEMMA_SEL=most_frequent
STEPS="fetch,prepare,fit"
FORCE=0

usage() {
  # the leading comment block (everything after the shebang, up to the first code line)
  awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
  cat << 'EOF'

Options:
  --lang CODE            wikipedia language code (required), e.g. es, en, de
  --snapshot DATE        dump snapshot, e.g. 20231101 (default: newest available)
  --limit N              stop after N articles (smoke tests; 0 = all)
  --parts N              split the output into N profile files (default 16). Batch size is
                         derived from the document count, so this bounds peak memory and
                         gives N snapshots: each part is written as it finishes, so an
                         interrupted run keeps the completed ones. FEWER parts also make the
                         first part a better sample of the corpus, which matters because
                         stopwords are detected once on it and reused -- see the note in
                         `textsearch fit`'s [stopwords] section. Wikipedia orders articles
                         longest-first so its first part is already the densest in function
                         words; a corpus in arbitrary order should use fewer, larger parts
  --batch-size N         articles per part, overriding --parts (0 = one single profile)
  --resume               skip parts whose .zip already exists instead of refitting them
  --outdim N             LSI dimension (default 256)
  --del-diac B           strip diacritics: true|false (default false -- Wikipedia's accents
                         are reliable, so "año" is kept distinct from "ano")
  --del-punc B           drop punctuation: true|false (default true -- it is ~half of all
                         token occurrences and only ~0.5% of the vocabulary)
  --query-expansion-k N              query_expansion per token (default 8)
  --min-chars N          skip articles shorter than this (default 200, drops stubs)
  --lc                   lowercase while normalizing, giving up the case-based sense
                         separation and the variant map that makes it searchable. Not
                         reversible: refit blends counters in the space the base was fitted in.
  --split-paragraphs     emit one document per paragraph (title-prefixed) instead of per
                         article -- see the note in the script; changes what a "document" is,
                         so stopword detection and avgdoclen change with it
  --min-tokens N         drop records with fewer than N whitespace words (default 4)
  --min-ndocs N          drop tokens in fewer than N documents (default 5; 1 keeps all).
                         The query_expansion network is an all-pairs search over the vocabulary,
                         so this cuts fit cost quadratically -- see ../README.md
  --no-stopwords         disable stopword detection/removal (on by default)
  --doc-freq-threshold F stopword document-frequency cutoff. Default is per-language (see
                         the table in the script); pass a value to override it
  --lemma-algorithm A    fft | dnet | randsel | multirandsel (default fft)
  --lemma-selector S     most_frequent (default) | shortest | shortest_then_most_frequent
  --steps LIST           comma list of fetch,prepare,fit (default all)
  --force                re-download / re-convert even if outputs exist
  -h, --help             this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang)                 LANG_CODE="$2"; shift 2 ;;
    --snapshot)             SNAPSHOT="$2"; shift 2 ;;
    --limit)                LIMIT="$2"; shift 2 ;;
    --parts)                PARTS="$2"; shift 2 ;;
    --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
    --resume)               RESUME=true; shift ;;
    --outdim)               OUTDIM="$2"; shift 2 ;;
    --del-diac)             DEL_DIAC="$2"; shift 2 ;;
    --del-punc)             DEL_PUNC="$2"; shift 2 ;;
    --query-expansion-k)                QUERY_EXPANSION_K="$2"; shift 2 ;;
    --min-chars)            MIN_CHARS="$2"; shift 2 ;;
    --split-paragraphs)     SPLIT_PARAGRAPHS=true; shift ;;
    --lc)                   LC=true; shift ;;
    --min-tokens)           MIN_TOKENS="$2"; shift 2 ;;
    --min-ndocs)            MIN_NDOCS="$2"; shift 2 ;;
    --no-stopwords)         STOPWORDS=false; shift ;;
    --doc-freq-threshold)   DOC_FREQ_THRESHOLD="$2"; shift 2 ;;
    --lemma-algorithm)      LEMMA_ALG="$2"; shift 2 ;;
    --lemma-selector)       LEMMA_SEL="$2"; shift 2 ;;
    --steps)                STEPS="$2"; shift 2 ;;
    --force)                FORCE=1; shift ;;
    -h|--help)              usage; exit 0 ;;
    *)                      die "unknown option: $1 (try --help)" ;;
  esac
done

[[ -n "$LANG_CODE" ]] || { usage; die "--lang is required"; }

# ── per-language stopword threshold ──────────────────────────────────────────
#
# `doc_freq_threshold` is a document-frequency RATIO relative to the batch, and that ratio is
# not scale-invariant: a bigger, more diverse batch dilutes every token, so a threshold
# calibrated on a small probe means something different in production. Measured on Portuguese,
# `area` sits at 0.520 over the first 10k articles and at 0.353 over 122,831 -- so a 10k probe
# says "0.5 is needed to protect it" about a word that is nowhere near the cutoff at the size
# the fit actually runs. Calibrate at the production batch size or not at all.
#
# What decides the value is what sits in the (0.4, 0.5] band at that size, because that band is
# exactly what 0.4 removes and 0.5 keeps:
#
#   pt  0.4   ao das mais a sao ligacoes externas seu ou pela          -- function + boilerplate
#   en  0.4   first one this has after are were be two his its new or  -- function only
#   es  0.5   ... e historia otros ha ano cuando esta vease gran ya le uno asi nombre
#
# Spanish is the exception because four of those twenty-one are content (`historia`, `ano`,
# `anos`, `nombre`), and the error is not symmetric: removing a token deletes it from the base
# vocabulary, where no later `refit` can recover it, while keeping one costs almost nothing
# because idf already drives a high-document-frequency token's weight toward zero. So drop the
# threshold where the band is pure function, and leave it where the band holds content.
#
# The gain from removing function words is a cost gain, not a relevance one: they dominate
# `numtokens`/`avgdoclen` (which BM25 normalizes by), spend the all-pairs query_expansion kNN budget,
# and crowd the leading LSI dimensions.
if [[ -z "$DOC_FREQ_THRESHOLD" && "$SPLIT_PARAGRAPHS" == "true" ]]; then
  # The per-language values below were calibrated with ARTICLES as documents and are simply the
  # wrong scale here: measured on 272,466 Spanish paragraphs, 0.5 flags 11 tokens, 0.4 flags 14,
  # 0.3 flags 19 and 0.1 flags 31 -- and the first 30 by document frequency are all function
  # words, while the highest content word sits at 0.069 (`anio`) and the highest Wikipedia
  # artifact at 0.019 (`vease`). So 0.5 leaves roughly twenty legitimate function words in, and
  # the gap below them is wide enough that the exact value stops being a tuning problem: this is
  # the main practical benefit of paragraphs as documents.
  DOC_FREQ_THRESHOLD=0.1
  log "stopword threshold for paragraph units: $DOC_FREQ_THRESHOLD (measured on es; the band"
  log "  between function words and content is wide here, unlike at article level)"
fi
# The query-expansion head cut has the same scale problem and the same answer: it is a document
# frequency, so it only means something once the document unit is fixed. It also has to sit BELOW
# the stopword threshold to do anything at all, since tokens above that are no longer in the
# vocabulary. With stopwords at 0.1, a cut of 0.05 denies a list to the strip in between -- 37
# tokens in es, 45 in pt, 49 in en, measured on 10k articles of each -- which is the function
# words plus the likes of `anos ano parte forma ciudad`, none of which needs enriching. The
# article-level equivalent was never measured, so article runs leave it off.
if [[ -z "${TS_HEAD_DF:-}" && "$SPLIT_PARAGRAPHS" == "true" ]]; then
  TS_HEAD_DF=0.05
  log "query-expansion head cut for paragraph units: $TS_HEAD_DF"
fi
if [[ -z "$DOC_FREQ_THRESHOLD" ]]; then
  case "$LANG_CODE" in
    es)       DOC_FREQ_THRESHOLD=0.5 ;;
    pt|en)    DOC_FREQ_THRESHOLD=0.4 ;;
    # Untuned: 0.5 is the conservative end, which for an unmeasured language is the right
    # default -- it removes less. Do not assume it transfers: Arabic flagged 19 candidates at
    # 0.5 against Spanish's 46 on the same 10k, because its function words are proclitics glued
    # inside other tokens rather than separate words at all.
    *)        DOC_FREQ_THRESHOLD=0.5 ;;
  esac
  log "stopword threshold for '$LANG_CODE': $DOC_FREQ_THRESHOLD (per-language default)"
fi
require_cmd curl python3 julia

has_step() { [[ ",$STEPS," == *",$1,"* ]]; }

# ── snapshot discovery ───────────────────────────────────────────────────────

if [[ -z "$SNAPSHOT" ]]; then
  log "discovering newest snapshot for lang=$LANG_CODE ..."
  SNAPSHOT="$(curl -sSf -m 120 "$HF_API" | python3 -c "
import json,sys
d = json.load(sys.stdin)
lang = '$LANG_CODE'
dates = sorted({
    p.split('/')[0].split('.', 1)[0]
    for s in d.get('siblings', [])
    for p in [s['rfilename']]
    if '/' in p and p.split('/')[0].endswith('.' + lang)
})
if not dates:
    sys.exit(f'no snapshot found for language {lang!r} in $HF_REPO')
print(dates[-1])
")" || die "snapshot discovery failed"
fi

CONFIG_DIR="${SNAPSHOT}.${LANG_CODE}"
PROFILE_NAME="wiki${SNAPSHOT}-${LANG_CODE}"
SHARD_DIR="$RAW_DIR/wikipedia/$CONFIG_DIR"
# The suffix is not cosmetic: `prepare` reuses an existing JSONL, so without it a
# --split-paragraphs run would silently fit on an article-level conversion left by an earlier run.
if [[ "$SPLIT_PARAGRAPHS" == "true" ]]; then
  JSONL="$WORK_DIR/wikipedia/${PROFILE_NAME}-paragraphs.jsonl"
else
  JSONL="$WORK_DIR/wikipedia/${PROFILE_NAME}.jsonl"
fi
OUT_DIR="$PROFILES_DIR/$PROFILE_NAME"
FIT_CFG="$WORK_DIR/wikipedia/${PROFILE_NAME}.fit.toml"

log "corpus=wikipedia snapshot=$SNAPSHOT lang=$LANG_CODE -> profile '$PROFILE_NAME'"
[[ "$SPLIT_PARAGRAPHS" == "true" ]] && log "unit=paragraph (min-tokens=$MIN_TOKENS)" || true
log "normalization: lc=$LC del_diac=$DEL_DIAC del_punc=$DEL_PUNC"

# ── fetch ────────────────────────────────────────────────────────────────────

if has_step fetch; then
  mkdir -p "$SHARD_DIR"
  log "listing shards for $CONFIG_DIR ..."
  # "<relative path>\t<expected size in bytes>" per shard
  mapfile -t SHARDS < <(curl -sSf -m 120 "$HF_API/tree/main/$CONFIG_DIR" | python3 -c "
import json,sys
for e in json.load(sys.stdin):
    if e.get('type') == 'file' and e['path'].endswith('.parquet'):
        size = e.get('size') or (e.get('lfs') or {}).get('size') or 0
        print(e['path'], size, sep='\t')
") || die "could not list shards for $CONFIG_DIR"

  [[ ${#SHARDS[@]} -gt 0 ]] || die "no parquet shards found for $CONFIG_DIR"
  log "found ${#SHARDS[@]} shard(s)"

  for entry in "${SHARDS[@]}"; do
    rel="${entry%%$'\t'*}"; want="${entry##*$'\t'}"
    dest="$SHARD_DIR/$(basename "$rel")"
    if [[ $FORCE -eq 0 && -f "$dest" ]]; then
      have=$(stat -c %s "$dest")
      if [[ "$have" == "$want" ]]; then
        log "  have $(basename "$dest") ($((want/1000000)) MB)"
        continue
      fi
      log "  resuming $(basename "$dest") ($((have/1000000))/$((want/1000000)) MB)"
    fi
    log "  downloading $(basename "$dest") ($((want/1000000)) MB)"
    curl -sSfL --retry 5 --retry-delay 5 -C - -o "$dest" "$HF_RESOLVE/$rel" \
      || die "download failed: $rel"
    have=$(stat -c %s "$dest")
    [[ "$have" == "$want" ]] || die "size mismatch for $rel: got $have, expected $want"
  done
  log "fetch complete: $SHARD_DIR"
fi

# ── prepare (parquet -> jsonl) ───────────────────────────────────────────────

if has_step prepare; then
  shopt -s nullglob
  shards=("$SHARD_DIR"/*.parquet)
  shopt -u nullglob
  [[ ${#shards[@]} -gt 0 ]] || die "no parquet shards in $SHARD_DIR -- run with --steps fetch first"

  if [[ $FORCE -eq 0 && -s "$JSONL" ]]; then
    log "reusing existing $JSONL ($(du -h "$JSONL" | cut -f1)); pass --force to rebuild"
  else
    mkdir -p "$(dirname "$JSONL")"
    log "converting ${#shards[@]} shard(s) -> $JSONL (min-chars=$MIN_CHARS limit=$LIMIT)"
    extra=()
    [[ "$LIMIT" != "0" ]] && extra+=(--limit "$LIMIT")
    if [[ "$SPLIT_PARAGRAPHS" == "true" ]]; then
      extra+=(--split-paragraphs --title-column title --min-tokens "$MIN_TOKENS")
    fi
    ts_julia "$CP_ROOT/lib/parquet_to_jsonl.jl" "$JSONL" "${shards[@]}" \
      --text-column text --min-chars "$MIN_CHARS" --keep-columns id,title,url "${extra[@]}"
  fi
fi

# ── fit ──────────────────────────────────────────────────────────────────────

if has_step fit; then
  [[ -s "$JSONL" ]] || die "missing $JSONL -- run with --steps prepare first"
  mkdir -p "$OUT_DIR"

  # Derive the batch size from the requested number of parts. Batch size is no longer a
  # time knob -- measured on 100k Spanish articles, 10/30/100k batches all took ~575s in
  # total -- so it is chosen to bound peak memory and to decide how many snapshots a long
  # run leaves behind, not to make the run faster.
  if [[ -z "$BATCH_SIZE" ]]; then
    [[ "$PARTS" -ge 1 ]] || die "--parts must be >= 1, got $PARTS"
    ndocs=$(wc -l < "$JSONL")
    [[ "$ndocs" -gt 0 ]] || die "$JSONL is empty"
    BATCH_SIZE=$(( (ndocs + PARTS - 1) / PARTS ))
    log "$ndocs documents / $PARTS parts -> batch_size=$BATCH_SIZE"
  else
    log "batch_size=$BATCH_SIZE (explicit, --parts ignored)"
  fi

  TS_JSONL="$JSONL" TS_OUTDIR="$OUT_DIR" TS_PREFIX="$PROFILE_NAME" TS_BATCH="$BATCH_SIZE" \
  TS_RESUME="$RESUME" TS_MIN_NDOCS="$MIN_NDOCS" TS_STOPWORDS="$STOPWORDS" \
  TS_DOC_FREQ_THRESHOLD="$DOC_FREQ_THRESHOLD" TS_OUTDIM="$OUTDIM" TS_QUERY_EXPANSION_K="$QUERY_EXPANSION_K" \
  TS_LEMMA_ALG="$LEMMA_ALG" TS_LEMMA_SEL="$LEMMA_SEL" TS_LANGUAGE="$LANG_CODE" \
  TS_HEAD_DF="${TS_HEAD_DF:-0.0}" \
  TS_DEL_DIAC="$DEL_DIAC" TS_DEL_PUNC="$DEL_PUNC" TS_LC="$LC" \
  TS_VARIANTS_MIN_NDOCS="${TS_VARIANTS_MIN_NDOCS:-20}" \
    ts_render_fit_config "$FIT_CFG"
  ts_fit "$FIT_CFG"
  log "profiles in $OUT_DIR:"
  ls -la "$OUT_DIR" >&2
  log "NOT published. Verify first (see ../README.md), e.g.:"
  log "  textsearch install $OUT_DIR/${PROFILE_NAME}-0001.zip $PROFILE_NAME && textsearch info $PROFILE_NAME"
fi
