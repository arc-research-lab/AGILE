#!/bin/bash
#
# Sweep benchmark: generate BFS & PR graphs with k=200,
# write each to a separate SSD (offset 0), and test with AGILE and BAM.
#
# 8 SSDs, 8 test cases (4 BFS + 4 PR), k=200
#
#   BFS (neighbors on SSD, each ~2GB):
#     bfs-g20-1  0000:01:00.0  Kronecker  scale=20  k=200
#     bfs-g20-2  0000:02:00.0  Kronecker  scale=20  k=200
#     bfs-u20-1  0000:03:00.0  Uniform    scale=20  k=200
#     bfs-u20-2  0000:04:00.0  Uniform    scale=20  k=200
#
#   PR (neighbors + weights on SSD, each ~2GB total):
#     pr-g19-1   0000:e1:00.0  Kronecker  scale=19  k=200
#     pr-g19-2   0000:e2:00.0  Kronecker  scale=19  k=200
#     pr-u19-1   0000:e3:00.0  Uniform    scale=19  k=200
#     pr-u19-2   0000:e4:00.0  Uniform    scale=19  k=200
#
# Graphs stored in: demo/graphs/<name>/
# Results stored in: demo/graphs/<name>/results/
#
# Usage:
#   sudo ./scripts/sweep_bench.sh
#   sudo ./scripts/sweep_bench.sh --gen-only
#   sudo ./scripts/sweep_bench.sh --skip-write

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_DIR="$REPO_ROOT/demo"
GRAPH_DIR="$DEMO_DIR/graphs"
GRAPH_GEN="$DEMO_DIR/build/tools/graph-gen/agile_graph_gen"
TEST_BFS="$SCRIPT_DIR/test_bfs.sh"
TEST_PR="$SCRIPT_DIR/test_pr.sh"

GEN_ONLY=false
SKIP_WRITE=false

for arg in "$@"; do
    case "$arg" in
        --gen-only)   GEN_ONLY=true ;;
        --skip-write) SKIP_WRITE=true ;;
        -h|--help)
            echo "Usage: sudo $0 [--gen-only] [--skip-write]"
            echo "  --gen-only    Only generate graphs, skip SSD write and testing"
            echo "  --skip-write  Skip writing to SSD (assume already written)"
            exit 0 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "Error: run with sudo"
    exit 1
fi

if [[ ! -x "$GRAPH_GEN" ]]; then
    echo "Error: graph-gen not found: $GRAPH_GEN"
    exit 1
fi

mkdir -p "$GRAPH_DIR"

# ── Test configurations ──
# NAME      APP  SSD              GEN  SCALE
TESTS=(
  "bfs-g20-1  bfs  0000:01:00.0     -g   20"
#   "bfs-g20-2  bfs  0000:02:00.0     -g   20"
  "bfs-u20-1  bfs  0000:03:00.0     -u   20"
#   "bfs-u20-2  bfs  0000:04:00.0     -u   20"
  "pr-g19-1   pr   0000:e1:00.0     -g   19"
#   "pr-g19-2   pr   0000:e2:00.0     -g   19"
  "pr-u19-1   pr   0000:e3:00.0     -u   19"
#   "pr-u19-2   pr   0000:e4:00.0     -u   19"
)

TOTAL=${#TESTS[@]}
IDX=0
FAILED=()

for entry in "${TESTS[@]}"; do
    read -r NAME APP SSD GEN SCALE <<< "$entry"
    IDX=$((IDX + 1))

    DIR="$GRAPH_DIR/$NAME"
    PREFIX="$DIR/graph"
    mkdir -p "$DIR/results"

    echo ""
    echo "========================================================"
    echo " [$IDX/$TOTAL] $NAME"
    echo "   app=$APP  ssd=$SSD  gen=$GEN  scale=$SCALE"
    echo "========================================================"
    echo ""

    # ── Phase 1: Generate graph ──
    if [[ ! -f "${PREFIX}.info.txt" ]]; then
        echo "--- Generating graph ---"
        GEN_ARGS="$GEN $SCALE -k 200 -o $PREFIX"
        [[ "$APP" == "pr" ]] && GEN_ARGS="$GEN_ARGS -w"
        echo "  $GRAPH_GEN $GEN_ARGS"
        $GRAPH_GEN $GEN_ARGS
    else
        echo "--- Graph exists, skipping generation ---"
    fi

    echo ""
    cat "${PREFIX}.info.txt"
    echo ""

    # Show file sizes
    for f in "$DIR"/graph.*.bin; do
        if [[ -f "$f" ]]; then
            SIZE=$(stat -c%s "$f")
            echo "  $(basename "$f"): $SIZE bytes ($((SIZE / 1048576)) MB)"
        fi
    done
    echo ""

    if [[ "$GEN_ONLY" == true ]]; then
        continue
    fi

    # ── Phase 2+3: Write to SSD and test ──
    LOG="$DIR/results/test.log"

    pushd "$DIR/results" > /dev/null

    TEST_ARGS=(-b "$SSD" -p "$PREFIX" -m both)
    [[ "$SKIP_WRITE" == true ]] && TEST_ARGS+=(-S)

    RC=0
    if [[ "$APP" == "bfs" ]]; then
        "$TEST_BFS" "${TEST_ARGS[@]}" 2>&1 | tee "$LOG" || RC=$?
    elif [[ "$APP" == "pr" ]]; then
        "$TEST_PR" "${TEST_ARGS[@]}" 2>&1 | tee "$LOG" || RC=$?
    fi

    popd > /dev/null

    if [[ $RC -ne 0 ]]; then
        echo "  *** [$NAME] FAILED (exit code $RC) ***"
        FAILED+=("$NAME")
    else
        echo "  [$NAME] complete. Results in $DIR/results/"
    fi

    # ── Emit per-test JSON result (consumed by websocket server) ──
    LOG="$DIR/results/test.log"
    if [[ -f "$LOG" ]]; then
        if [[ "$APP" == "bfs" ]]; then
            _TAG="BFS"; _TAG_BAM="BFS-BAM"; _TAG_GPU="BFS-GPU"
        else
            _TAG="PR"; _TAG_BAM="PR-BAM"; _TAG_GPU="PR-GPU"
        fi
        _AC=""; _AW=""; _BC=""; _BW=""; _GT=""; _MD5=""
        _LINE=$(grep "\[${_TAG}\].*COLD kernel:" "$LOG" 2>/dev/null | head -1)
        if [[ -n "$_LINE" ]]; then
            _AC=$(echo "$_LINE" | sed -n 's/.*COLD kernel: \([0-9.]*\) s.*/\1/p')
            _AW=$(echo "$_LINE" | sed -n 's/.*WARM kernel: \([0-9.]*\) s.*/\1/p')
        fi
        _LINE=$(grep "\[${_TAG_BAM}\].*COLD kernel:" "$LOG" 2>/dev/null | head -1)
        if [[ -n "$_LINE" ]]; then
            _BC=$(echo "$_LINE" | sed -n 's/.*COLD kernel: \([0-9.]*\) s.*/\1/p')
            _BW=$(echo "$_LINE" | sed -n 's/.*WARM kernel: \([0-9.]*\) s.*/\1/p')
        fi
        _LINE=$(grep "\[${_TAG_GPU}\].*kernel time" "$LOG" 2>/dev/null | head -1)
        if [[ -n "$_LINE" ]]; then
            _GT=$(echo "$_LINE" | sed -n 's/.*kernel time[=:] *\([0-9.]*\) s.*/\1/p')
        fi
        if grep -q "MATCH" "$LOG" 2>/dev/null; then _MD5="MATCH"
        elif grep -q "DIFFER" "$LOG" 2>/dev/null; then _MD5="DIFFER"
        fi
        echo "__JSON_RESULT__{\"test\":\"$NAME\",\"app\":\"$APP\",\"ssd\":\"$SSD\",\"agile-cold\":\"$_AC\",\"agile-warm\":\"$_AW\",\"bam-cold\":\"$_BC\",\"bam-warm\":\"$_BW\",\"gpu\":\"$_GT\",\"md5\":\"$_MD5\"}"
    fi
done

# ── Summary ──
echo ""
echo "========================================================"
echo " Sweep complete: $TOTAL tests"
echo "========================================================"

if [[ "$GEN_ONLY" == true ]]; then
    echo " (gen-only mode — no tests were run)"
    echo " Graphs: $GRAPH_DIR/"
else
    # ── Detailed summary table ──
    echo ""
    printf " %-12s  %-10s %-10s  %-10s %-10s  %-10s  %-10s %-10s  %-6s\n" \
        "NAME" "AGILE-COLD" "AGILE-WARM" "BAM-COLD" "BAM-WARM" "GPU" "COLD-SPD" "WARM-SPD" "MD5"
    printf " %-12s  %-10s %-10s  %-10s %-10s  %-10s  %-10s %-10s  %-6s\n" \
        "----" "----------" "----------" "--------" "--------" "---" "--------" "--------" "---"

    for entry in "${TESTS[@]}"; do
        read -r NAME APP SSD _ _ <<< "$entry"
        LOG="$GRAPH_DIR/$NAME/results/test.log"

        # Determine log tag based on app type
        if [[ "$APP" == "bfs" ]]; then
            TAG="BFS"; TAG_BAM="BFS-BAM"; TAG_GPU="BFS-GPU"
        else
            TAG="PR"; TAG_BAM="PR-BAM"; TAG_GPU="PR-GPU"
        fi

        A_COLD="-"; A_WARM="-"
        B_COLD="-"; B_WARM="-"
        G_TIME="-"
        COLD_SPD="-"; WARM_SPD="-"
        MD5_STATUS="-"

        if [[ -f "$LOG" ]]; then
            # Parse AGILE line
            LINE=$(grep "\[${TAG}\].*COLD kernel:" "$LOG" 2>/dev/null | head -1)
            if [[ -n "$LINE" ]]; then
                A_COLD=$(echo "$LINE" | sed -n 's/.*COLD kernel: \([0-9.]*\) s.*/\1/p')
                A_WARM=$(echo "$LINE" | sed -n 's/.*WARM kernel: \([0-9.]*\) s.*/\1/p')
            fi
            # Parse BAM line
            LINE=$(grep "\[${TAG_BAM}\].*COLD kernel:" "$LOG" 2>/dev/null | head -1)
            if [[ -n "$LINE" ]]; then
                B_COLD=$(echo "$LINE" | sed -n 's/.*COLD kernel: \([0-9.]*\) s.*/\1/p')
                B_WARM=$(echo "$LINE" | sed -n 's/.*WARM kernel: \([0-9.]*\) s.*/\1/p')
            fi
            # Parse GPU time (BFS: "kernel time=X.X s", PR: "Total kernel time: X.X s")
            LINE=$(grep "\[${TAG_GPU}\].*kernel time" "$LOG" 2>/dev/null | head -1)
            if [[ -n "$LINE" ]]; then
                G_TIME=$(echo "$LINE" | sed -n 's/.*kernel time[=:] *\([0-9.]*\) s.*/\1/p')
            fi
            # Compute AGILE-over-BAM speedups
            if [[ "$A_COLD" != "-" && "$B_COLD" != "-" ]]; then
                COLD_SPD=$(awk "BEGIN { printf \"%.2fx\", $B_COLD / $A_COLD }")
            fi
            if [[ "$A_WARM" != "-" && "$B_WARM" != "-" ]]; then
                WARM_SPD=$(awk "BEGIN { printf \"%.2fx\", $B_WARM / $A_WARM }")
            fi
            # Parse MD5 match
            if grep -q "MATCH" "$LOG" 2>/dev/null; then
                MD5_STATUS="MATCH"
            elif grep -q "DIFFER" "$LOG" 2>/dev/null; then
                MD5_STATUS="DIFFER"
            fi
        fi

        # Check if failed
        STATUS=""
        for f in "${FAILED[@]+"${FAILED[@]}"}"; do
            [[ "$f" == "$NAME" ]] && STATUS=" FAIL"
        done

        printf " %-12s  %-10s %-10s  %-10s %-10s  %-10s  %-10s %-10s  %-6s%s\n" \
            "$NAME" "$A_COLD" "$A_WARM" "$B_COLD" "$B_WARM" "$G_TIME" "$COLD_SPD" "$WARM_SPD" "$MD5_STATUS" "$STATUS"
    done

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo ""
        echo " FAILED: ${FAILED[*]}"
    else
        echo ""
        echo " All tests passed."
    fi

    # ── Write JSON results for HTML UI ──
    JSON_DIR="$DEMO_DIR/html/data"
    JSON_FILE="$JSON_DIR/sweep_results.json"
    mkdir -p "$JSON_DIR"

    echo "[" > "$JSON_FILE"
    FIRST=true
    for entry in "${TESTS[@]}"; do
        read -r NAME APP SSD _ _ <<< "$entry"
        LOG="$GRAPH_DIR/$NAME/results/test.log"

        if [[ "$APP" == "bfs" ]]; then
            TAG="BFS"; TAG_BAM="BFS-BAM"; TAG_GPU="BFS-GPU"
        else
            TAG="PR"; TAG_BAM="PR-BAM"; TAG_GPU="PR-GPU"
        fi

        A_COLD=""; A_WARM=""
        B_COLD=""; B_WARM=""
        G_TIME=""
        MD5_STATUS=""

        if [[ -f "$LOG" ]]; then
            LINE=$(grep "\[${TAG}\].*COLD kernel:" "$LOG" 2>/dev/null | head -1)
            if [[ -n "$LINE" ]]; then
                A_COLD=$(echo "$LINE" | sed -n 's/.*COLD kernel: \([0-9.]*\) s.*/\1/p')
                A_WARM=$(echo "$LINE" | sed -n 's/.*WARM kernel: \([0-9.]*\) s.*/\1/p')
            fi
            LINE=$(grep "\[${TAG_BAM}\].*COLD kernel:" "$LOG" 2>/dev/null | head -1)
            if [[ -n "$LINE" ]]; then
                B_COLD=$(echo "$LINE" | sed -n 's/.*COLD kernel: \([0-9.]*\) s.*/\1/p')
                B_WARM=$(echo "$LINE" | sed -n 's/.*WARM kernel: \([0-9.]*\) s.*/\1/p')
            fi
            LINE=$(grep "\[${TAG_GPU}\].*kernel time" "$LOG" 2>/dev/null | head -1)
            if [[ -n "$LINE" ]]; then
                G_TIME=$(echo "$LINE" | sed -n 's/.*kernel time[=:] *\([0-9.]*\) s.*/\1/p')
            fi
            if grep -q "MATCH" "$LOG" 2>/dev/null; then
                MD5_STATUS="MATCH"
            elif grep -q "DIFFER" "$LOG" 2>/dev/null; then
                MD5_STATUS="DIFFER"
            fi
        fi

        [[ "$FIRST" == true ]] && FIRST=false || echo "," >> "$JSON_FILE"
        cat >> "$JSON_FILE" <<JSONEOF
  {"test":"$NAME","app":"$APP","ssd":"$SSD","agile-cold":"$A_COLD","agile-warm":"$A_WARM","bam-cold":"$B_COLD","bam-warm":"$B_WARM","gpu":"$G_TIME","md5":"$MD5_STATUS"}
JSONEOF
    done
    echo "" >> "$JSON_FILE"
    echo "]" >> "$JSON_FILE"

    echo ""
    echo " JSON results: $JSON_FILE"

    echo ""
    echo " Results: $GRAPH_DIR/*/results/"
    echo " Logs:    $GRAPH_DIR/*/results/test.log"
fi
