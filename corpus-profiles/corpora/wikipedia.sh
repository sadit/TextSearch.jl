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
#   corpora/wikipedia.sh --lang es --steps fetch          # download only
#
# Produces profiles/wiki<SNAPSHOT>-<LANG>/wiki<SNAPSHOT>-<LANG>-NNNN.zip -- one independent
# profile per --batch-size articles.
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
BATCH_SIZE=25000
OUTDIM=128
SYN_K=8
MIN_CHARS=200
MIN_NDOCS=5
STOPWORDS=true
DOC_FREQ_THRESHOLD=0.5
LEMMA_ALG=fft
LEMMA_SEL=shortest
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
  --batch-size N         articles per output profile (default 25000; 0 = single profile)
  --outdim N             LSI dimension (default 128)
  --syn-k N              synonyms per token (default 8)
  --min-chars N          skip articles shorter than this (default 200, drops stubs)
  --min-ndocs N          drop tokens in fewer than N documents (default 5; 1 keeps all).
                         The synonym network is an all-pairs search over the vocabulary,
                         so this cuts fit cost quadratically -- see ../README.md
  --no-stopwords         disable stopword detection/removal (on by default)
  --doc-freq-threshold F stopword document-frequency cutoff (default 0.5)
  --lemma-algorithm A    fft | dnet | randsel | multirandsel (default fft)
  --lemma-selector S     shortest | most_frequent | shortest_then_most_frequent
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
    --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
    --outdim)               OUTDIM="$2"; shift 2 ;;
    --syn-k)                SYN_K="$2"; shift 2 ;;
    --min-chars)            MIN_CHARS="$2"; shift 2 ;;
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
JSONL="$WORK_DIR/wikipedia/${PROFILE_NAME}.jsonl"
OUT_DIR="$PROFILES_DIR/$PROFILE_NAME"
FIT_CFG="$WORK_DIR/wikipedia/${PROFILE_NAME}.fit.toml"

log "corpus=wikipedia snapshot=$SNAPSHOT lang=$LANG_CODE -> profile '$PROFILE_NAME'"

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
    ts_julia "$CP_ROOT/lib/parquet_to_jsonl.jl" "$JSONL" "${shards[@]}" \
      --text-column text --min-chars "$MIN_CHARS" --keep-columns id,title,url "${extra[@]}"
  fi
fi

# ── fit ──────────────────────────────────────────────────────────────────────

if has_step fit; then
  [[ -s "$JSONL" ]] || die "missing $JSONL -- run with --steps prepare first"
  mkdir -p "$OUT_DIR"
  ts_render_fit_config "$FIT_CFG" "$JSONL" "$OUT_DIR" "$PROFILE_NAME" "$BATCH_SIZE" \
    "$STOPWORDS" "$DOC_FREQ_THRESHOLD" "$OUTDIM" "$SYN_K" "$LEMMA_ALG" "$LEMMA_SEL" \
    "$MIN_NDOCS"
  ts_fit "$FIT_CFG"
  log "profiles in $OUT_DIR:"
  ls -la "$OUT_DIR" >&2
  log "NOT published. Verify first (see ../README.md), e.g.:"
  log "  textsearch install $OUT_DIR/${PROFILE_NAME}-0001.zip $PROFILE_NAME && textsearch info $PROFILE_NAME"
fi
