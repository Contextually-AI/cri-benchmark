#!/usr/bin/env bash
set -euo pipefail

# CRI Benchmark — Docker Runner
#
# Usage:
#   ./run.sh                                        All adapters, all datasets
#   ./run.sh --limit 50                             Smoke test (50 messages)
#   ./run.sh --adapter rag                          Single adapter, all datasets
#   ./run.sh --adapter rag --dataset datasets/canonical/persona-1-basic
#   ./run.sh --adapter rag --adapter full-context   Multiple adapters
#   ./run.sh --format markdown --verbose            Extra flags forwarded to cri

IMAGE="cri-benchmark"
RUN_ID="$(date +%Y_%m_%d_%H_%M_%S)"
RESULTS_DIR="$(pwd)/logs/${RUN_ID}"
DEFAULT_ADAPTERS=("no-memory" "full-context" "rag")
SCRIPT_START=$SECONDS

# ── Helpers ──────────────────────────────────────────────────────
ts() { date +%H:%M:%S; }
elapsed() {
    local diff=$(( SECONDS - $1 ))
    printf "%dm%02ds" $((diff / 60)) $((diff % 60))
}
log() { echo "[$(ts)] $*"; }

# ── Parse arguments ──────────────────────────────────────────────
ADAPTERS=()
DATASETS=()
LIMIT=""
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --adapter)
            ADAPTERS+=("$2"); shift 2 ;;
        --dataset)
            DATASETS+=("$2"); shift 2 ;;
        --limit)
            LIMIT="$2"; shift 2 ;;
        --output|--format|--profile|--dimensions|--judge-runs)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --verbose|--scale-test)
            EXTRA_ARGS+=("$1"); shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Defaults
if [ ${#ADAPTERS[@]} -eq 0 ]; then
    ADAPTERS=("${DEFAULT_ADAPTERS[@]}")
fi

# ── Setup logging ────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Build ────────────────────────────────────────────────────────
STEP_START=$SECONDS
log "Building CRI Benchmark container..."
docker build -t "$IMAGE" .
log "Build complete ($(elapsed $STEP_START))"

# ── Discover datasets if none specified ──────────────────────────
if [ ${#DATASETS[@]} -eq 0 ]; then
    STEP_START=$SECONDS
    log "Discovering datasets..."
    DISCOVERED=$(docker run --rm \
        --user "$(id -u):$(id -g)" \
        -v "$(pwd)/../.auth_token:/.auth_token:ro" \
        "$IMAGE" list-datasets 2>/dev/null \
        | grep -oP '(?<=│ )datasets/\S+' || true)

    if [ -z "$DISCOVERED" ]; then
        DISCOVERED=$(find datasets/canonical -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
    fi

    if [ -z "$DISCOVERED" ]; then
        log "ERROR: No datasets found. Pass --dataset explicitly."
        exit 1
    fi

    for ds in $DISCOVERED; do
        DATASETS+=("$ds")
    done
    log "Found ${#DATASETS[@]} dataset(s) ($(elapsed $STEP_START))"
fi

# ── Build common flags ───────────────────────────────────────────
COMMON_FLAGS=()
if [ -n "$LIMIT" ]; then
    COMMON_FLAGS+=("--limit" "$LIMIT")
fi

# Add defaults only if not specified by user
HAS_FORMAT=false
HAS_VERBOSE=false
HAS_PROFILE=false
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    case "$arg" in
        --format)   HAS_FORMAT=true ;;
        --verbose)  HAS_VERBOSE=true ;;
        --profile)  HAS_PROFILE=true ;;
    esac
done
if [ "$HAS_FORMAT" = false ]; then
    COMMON_FLAGS+=("--format" "json")
fi
if [ "$HAS_VERBOSE" = false ]; then
    COMMON_FLAGS+=("--verbose")
fi
if [ "$HAS_PROFILE" = false ]; then
    COMMON_FLAGS+=("--profile" "extended")
fi

# ── Run ──────────────────────────────────────────────────────────
log "Adapters: ${ADAPTERS[*]}"
log "Datasets: ${DATASETS[*]}"
[ -n "$LIMIT" ] && log "Limit: $LIMIT messages per dataset"
log "Log file: $LOG_FILE"
echo ""

for ds in "${DATASETS[@]}"; do
    for adapter in "${ADAPTERS[@]}"; do
        STEP_START=$SECONDS
        echo "──────────────────────────────────────────────"
        log "Starting adapter=$adapter dataset=$ds"
        echo "──────────────────────────────────────────────"

        # Auto-generate --output if user didn't provide one
        OUTPUT_FLAG=()
        HAS_OUTPUT=false
        for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
            if [ "$arg" = "--output" ]; then HAS_OUTPUT=true; break; fi
        done
        if [ "$HAS_OUTPUT" = false ]; then
            OUTPUT_FLAG=("--output" "/app/results/${adapter}_$(basename "$ds")")
        fi

        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -v "$RESULTS_DIR:/app/results" \
            -v "$(pwd)/../.auth_token:/.auth_token:ro" \
            "$IMAGE" run \
                --adapter "$adapter" \
                --dataset "$ds" \
                "${OUTPUT_FLAG[@]}" \
                "${COMMON_FLAGS[@]}" \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
            || log "WARNING: Adapter '$adapter' failed on '$ds' (skipping)"

        log "Finished adapter=$adapter dataset=$ds ($(elapsed $STEP_START))"
        echo ""
    done
done

log "All runs complete (total: $(elapsed $SCRIPT_START))"
log "Results in $RESULTS_DIR/"

# ── Generate summary ─────────────────────────────────────────────
SUMMARY="$RESULTS_DIR/summary.md"
log "Generating summary..."

python3 -c "
import json, os, sys

results_dir = sys.argv[1]
dims = ['PAS','DBU','MEI','TC','CRQ','QRP','SFC','LNC','ARS']

# Collect results
rows = []
for name in sorted(os.listdir(results_dir)):
    rpath = os.path.join(results_dir, name, 'result.json')
    if not os.path.isfile(rpath):
        continue
    with open(rpath) as f:
        data = json.load(f)
    cri = data['cri_result']
    # Parse adapter and dataset from directory name
    parts = name.split('_', 1)
    adapter = parts[0] if len(parts) > 0 else name
    dataset = parts[1] if len(parts) > 1 else ''
    rows.append({
        'adapter': cri.get('system_name', adapter),
        'dataset': data.get('dataset_id', dataset),
        'cri': cri.get('cri', 0),
        **{d.lower(): cri.get(d.lower(), None) for d in dims}
    })

if not rows:
    print('No results found.', file=sys.stderr)
    sys.exit(0)

# Group by dataset
datasets = sorted(set(r['dataset'] for r in rows))
adapters = sorted(set(r['adapter'] for r in rows))

lines = ['# CRI Benchmark Summary', '']

# Overview table: CRI composite per adapter x dataset
lines.append('## Composite CRI')
lines.append('')
header = '| Adapter |' + '|'.join(f' {ds} ' for ds in datasets) + '|'
sep = '|---------|' + '|'.join(':------:' for _ in datasets) + '|'
lines.append(header)
lines.append(sep)
for adapter in adapters:
    cells = []
    for ds in datasets:
        match = [r for r in rows if r['adapter'] == adapter and r['dataset'] == ds]
        if match:
            cells.append(f' **{match[0][\"cri\"]:.4f}** ')
        else:
            cells.append(' — ')
    lines.append(f'| {adapter} |' + '|'.join(cells) + '|')
lines.append('')

# Per-dimension tables
for dim in dims:
    dim_lower = dim.lower()
    lines.append(f'## {dim}')
    lines.append('')
    lines.append(header)
    lines.append(sep)
    for adapter in adapters:
        cells = []
        for ds in datasets:
            match = [r for r in rows if r['adapter'] == adapter and r['dataset'] == ds]
            if match and match[0].get(dim_lower) is not None:
                val = match[0][dim_lower]
                cells.append(f' {val:.4f} ')
            else:
                cells.append(' — ')
        lines.append(f'| {adapter} |' + '|'.join(cells) + '|')
    lines.append('')

with open(sys.argv[2], 'w') as f:
    f.write('\n'.join(lines))
" "$RESULTS_DIR" "$SUMMARY"

log "Summary written to $SUMMARY"
