#!/bin/bash
#
# Write graph data for PageRank to SSD and run PR with AGILE and/or BAM.
#
# Usage:
#   sudo ./test_pr.sh -b <BDF> -p <prefix> [options]
#   sudo ./test_pr.sh -b <BDF> -i <info_file> [options]
#
# Examples:
#   sudo ./scripts/test_pr.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20
#   sudo ./scripts/test_pr.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20 -m agile
#   sudo ./scripts/test_pr.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20 -m bam -S
#   sudo ./scripts/test_pr.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20 -o 1024

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWITCH_SCRIPT="$SCRIPT_DIR/switch_nvme_driver.sh"

# Demo PR binaries
PR_AGILE="$REPO_ROOT/demo/build/examples/pr-agile/agile_demo_pr"
PR_BAM="$REPO_ROOT/demo/build/examples/pr-bam/agile_demo_pr_bam"
PR_GPU="$REPO_ROOT/demo/build/examples/pr-gpu/agile_demo_pr_gpu"

# gdrcopy library path (needed by AGILE runtime)
GDRCOPY_LIB="$REPO_ROOT/driver/gdrcopy/src"

# Defaults
BDF=""
INFO_FILE=""
PREFIX=""
SSD_OFFSET_PAGES=0
MODE="both"        # agile, bam, or both
QUEUE_NUM=15
QUEUE_DEPTH=512
MAX_ITR=20
SKIP_WRITE=false

usage() {
    echo "Usage: sudo $0 -b <BDF> {-i <info_file> | -p <prefix>} [options]"
    echo ""
    echo "Options:"
    echo "  -b <BDF>        PCIe BDF of target SSD (required)"
    echo "  -i <info_file>  Path to .info.txt from graph-gen"
    echo "  -p <prefix>     Graph file prefix (auto-appends .info.txt)"
    echo "  -o <offset>     SSD offset for data in 4KB pages (default: 0)"
    echo "  -m <mode>       Test mode: agile, bam, or both (default: both)"
    echo "  -q <queues>     Number of NVMe queues (default: 15)"
    echo "  -n <max_itr>    Max PR iterations (default: 20)"
    echo "  -S              Skip writing data to SSD (assume already written)"
    echo "  -h              Show help"
    exit 1
}

while getopts "b:i:p:o:m:q:n:Sh" opt; do
    case "$opt" in
        b) BDF="$OPTARG" ;;
        i) INFO_FILE="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        o) SSD_OFFSET_PAGES="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        q) QUEUE_NUM="$OPTARG" ;;
        n) MAX_ITR="$OPTARG" ;;
        S) SKIP_WRITE=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Resolve info file from prefix
if [[ -n "$PREFIX" && -z "$INFO_FILE" ]]; then
    INFO_FILE="${PREFIX}.info.txt"
fi

# Validation
if [[ -z "$BDF" ]]; then
    echo "Error: -b <BDF> is required."
    usage
fi
if [[ -z "$INFO_FILE" ]]; then
    echo "Error: -i <info_file> or -p <prefix> is required."
    usage
fi
if [[ ! -f "$INFO_FILE" ]]; then
    echo "Error: info file not found: $INFO_FILE"
    exit 1
fi
if [[ "$MODE" != "agile" && "$MODE" != "bam" && "$MODE" != "both" ]]; then
    echo "Error: -m must be agile, bam, or both"
    usage
fi
if [[ $EUID -ne 0 ]]; then
    echo "Error: this script must be run as root (sudo)."
    exit 1
fi
if [[ ! -d "/sys/bus/pci/devices/$BDF" ]]; then
    echo "Error: PCI device $BDF does not exist."
    exit 1
fi

# Derive prefix from info file if not set
if [[ -z "$PREFIX" ]]; then
    PREFIX="${INFO_FILE%.info.txt}"
fi

NEIGHBORS_FILE="${PREFIX}.neighbors.bin"
WEIGHTS_FILE="${PREFIX}.weights.bin"

if [[ ! -f "$NEIGHBORS_FILE" ]]; then
    echo "Error: neighbors file not found: $NEIGHBORS_FILE"
    exit 1
fi
if [[ ! -f "$WEIGHTS_FILE" ]]; then
    echo "Error: weights file not found: $WEIGHTS_FILE"
    echo "PR requires a weighted graph. Regenerate with: graph-gen -g <scale> -o <prefix> -w"
    exit 1
fi

NEIGHBORS_SIZE=$(stat -c%s "$NEIGHBORS_FILE")
WEIGHTS_SIZE=$(stat -c%s "$WEIGHTS_FILE")
WEIGHT_OFFSET_ELEMS=$((NEIGHBORS_SIZE / 4))
TOTAL_SSD_PAGES=$(( (NEIGHBORS_SIZE + WEIGHTS_SIZE) / 4096 ))

echo "============================================"
echo " AGILE PageRank Test"
echo "============================================"
echo "BDF              : $BDF"
echo "Info file        : $INFO_FILE"
echo "Neighbors        : $NEIGHBORS_FILE ($NEIGHBORS_SIZE bytes)"
echo "Weights          : $WEIGHTS_FILE ($WEIGHTS_SIZE bytes)"
echo "SSD offset       : ${SSD_OFFSET_PAGES} pages"
echo "Weight offset    : $WEIGHT_OFFSET_ELEMS uint32 elements"
echo "Total SSD pages  : $TOTAL_SSD_PAGES"
echo "Mode             : $MODE"
echo "Max iterations   : $MAX_ITR"
echo ""

# ── Step 1: Write neighbors + weights to SSD ──
if [[ "$SKIP_WRITE" == false ]]; then
    echo "=== Step 1: Writing PR data to SSD ==="

    # Switch to stock nvme driver for dd access
    "$SWITCH_SCRIPT" nvme "$BDF"
    sleep 2

    # Find the /dev/nvmeXn1 block device
    NVME_DEV=""
    for ctrl in /sys/class/nvme/nvme*; do
        if [[ -L "$ctrl" ]]; then
            ctrl_addr=$(cat "$ctrl/address" 2>/dev/null || true)
            if [[ "$ctrl_addr" == "$BDF" ]]; then
                ctrl_name=$(basename "$ctrl")
                for ns in /dev/${ctrl_name}n*; do
                    if [[ -b "$ns" ]]; then
                        NVME_DEV="$ns"
                        break
                    fi
                done
                break
            fi
        fi
    done

    if [[ -z "$NVME_DEV" ]]; then
        echo "Error: could not find block device for $BDF"
        exit 1
    fi

    echo "  Block device: $NVME_DEV"

    # Write neighbors
    echo "  [1/2] neighbors at page offset $SSD_OFFSET_PAGES"
    dd if="$NEIGHBORS_FILE" of="$NVME_DEV" bs=4096 oflag=direct seek="$SSD_OFFSET_PAGES" status=progress
    echo ""

    # Write weights immediately after neighbors
    WEIGHTS_PAGE_OFFSET=$((SSD_OFFSET_PAGES + NEIGHBORS_SIZE / 4096))
    echo "  [2/2] weights at page offset $WEIGHTS_PAGE_OFFSET"
    dd if="$WEIGHTS_FILE" of="$NVME_DEV" bs=4096 oflag=direct seek="$WEIGHTS_PAGE_OFFSET" status=progress
    echo ""

    echo "  Write complete."
else
    echo "=== Step 1: Skipped (data already on SSD) ==="
fi
echo ""

# ── Step 2: Run PR with AGILE driver ──
if [[ "$MODE" == "agile" || "$MODE" == "both" ]]; then
    echo "=== Step 2: PageRank with AGILE driver ==="

    if [[ ! -x "$PR_AGILE" ]]; then
        echo "Error: AGILE PR binary not found at $PR_AGILE"
        echo "Build: cd $REPO_ROOT/demo/build && make agile_demo_pr"
        exit 1
    fi

    "$SWITCH_SCRIPT" agile-nvme "$BDF"
    sleep 1

    AGILE_DEV="/dev/AGILE-NVMe-${BDF}"
    if [[ ! -e "$AGILE_DEV" ]]; then
        echo "Error: AGILE device node $AGILE_DEV not found"
        exit 1
    fi

    echo "  Device: $AGILE_DEV"
    echo "  Running: agile_demo_pr --info $INFO_FILE -d $AGILE_DEV --blk-offset $SSD_OFFSET_PAGES --weight-offset $WEIGHT_OFFSET_ELEMS --ssd-blocks $TOTAL_SSD_PAGES --max-itr $MAX_ITR -q $QUEUE_NUM"
    echo ""

    LD_LIBRARY_PATH="$GDRCOPY_LIB" "$PR_AGILE" \
        --info "$INFO_FILE" \
        -d "$AGILE_DEV" \
        --blk-offset "$SSD_OFFSET_PAGES" \
        --weight-offset "$WEIGHT_OFFSET_ELEMS" \
        --ssd-blocks "$TOTAL_SSD_PAGES" \
        --max-itr "$MAX_ITR" \
        -q "$QUEUE_NUM" \
        --queue-depth "$QUEUE_DEPTH"

    echo ""
    echo "  AGILE PR complete."
    echo ""
fi

# ── Step 3: Run PR with BAM driver ──
if [[ "$MODE" == "bam" || "$MODE" == "both" ]]; then
    echo "=== Step 3: PageRank with BAM driver ==="

    if [[ ! -x "$PR_BAM" ]]; then
        echo "Error: BAM PR binary not found at $PR_BAM"
        echo "Build: cd $REPO_ROOT/demo/build && make agile_demo_pr_bam"
        exit 1
    fi

    "$SWITCH_SCRIPT" bam-nvme "$BDF"
    sleep 1

    # Find the BAM device node
    BAM_DEV=""
    for dev in /dev/libnvm*; do
        if [[ -c "$dev" ]]; then
            BAM_DEV="$dev"
            break
        fi
    done

    if [[ -z "$BAM_DEV" ]]; then
        echo "Error: no /dev/libnvm* device node found"
        exit 1
    fi

    echo "  Device: $BAM_DEV"
    echo "  Running: agile_demo_pr_bam --info $INFO_FILE -d $BAM_DEV --weight-offset $WEIGHT_OFFSET_ELEMS --max-itr $MAX_ITR -q $QUEUE_NUM"
    echo ""

    "$PR_BAM" \
        --info "$INFO_FILE" \
        -d "$BAM_DEV" \
        --weight-offset "$WEIGHT_OFFSET_ELEMS" \
        --max-itr "$MAX_ITR" \
        -q "$QUEUE_NUM" \
        --queue-depth "$QUEUE_DEPTH"

    echo ""
    echo "  BAM PR complete."
    echo ""
fi

# ── Step 4: Run PR on GPU (no SSD) ──
if [[ "$MODE" == "both" ]]; then
    echo "=== Step 4: PageRank on GPU memory ==="

    if [[ ! -x "$PR_GPU" ]]; then
        echo "  Warning: GPU PR binary not found at $PR_GPU — skipping"
    else
        echo "  Running: agile_demo_pr_gpu --info $INFO_FILE --max-itr $MAX_ITR"
        echo ""

        "$PR_GPU" \
            --info "$INFO_FILE" \
            --max-itr "$MAX_ITR" \
            -o "res-pr-gpu.bin"

        echo ""
        echo "  GPU PR complete."
    fi
    echo ""
fi

# ── Step 5: Restore to agile-nvme ──
echo "=== Restoring $BDF to agile-nvme ==="
"$SWITCH_SCRIPT" agile-nvme "$BDF"

# ── Step 6: Compare results if both ran ──
if [[ "$MODE" == "both" ]]; then
    echo ""
    echo "=== Comparing results ==="
    AGILE_RES="res-pr.bin"
    BAM_RES="res-pr-bam.bin"
    if [[ -f "$AGILE_RES" && -f "$BAM_RES" ]]; then
        AGILE_MD5=$(md5sum "$AGILE_RES" | awk '{print $1}')
        BAM_MD5=$(md5sum "$BAM_RES" | awk '{print $1}')
        echo "  AGILE: $AGILE_MD5  ($AGILE_RES)"
        echo "  BAM:   $BAM_MD5  ($BAM_RES)"
        if [[ "$AGILE_MD5" == "$BAM_MD5" ]]; then
            echo "  MATCH: results are identical"
        else
            echo "  DIFFER: results differ (may be due to floating-point ordering)"
        fi
    else
        echo "  Could not compare: one or both result files missing"
    fi
fi

echo ""
echo "============================================"
echo " Done."
echo "============================================"
