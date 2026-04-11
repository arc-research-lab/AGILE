#!/bin/bash
#
# Generate a graph and write it to a specific NVMe SSD.
#
# This script:
#   1. Generates a graph using agile_graph_gen (Kronecker or Uniform)
#   2. Switches the target SSD to the stock nvme driver (via switch_nvme_driver.sh)
#   3. Writes the selected binary files to the SSD using dd
#   4. Switches the SSD back to the original driver
#
# Application modes (-a):
#   bfs  — writes offsets + neighbors to SSD (default)
#   pr   — writes neighbors + weights to SSD (forces -w)
#
# Usage:
#   sudo ./gen_graph_to_ssd.sh -b <BDF> -g <scale> [-a bfs|pr] [-k <degree>] [-w] [-d <driver>]
#   sudo ./gen_graph_to_ssd.sh -b <BDF> -u <scale> [-a bfs|pr] [-k <degree>] [-w] [-d <driver>]
#   sudo ./gen_graph_to_ssd.sh -b <BDF> -i <prefix> [-a bfs|pr] [-d <driver>]
#
# Examples:
#   sudo ./scripts/gen_graph_to_ssd.sh -b 0000:e1:00.0 -g 20
#   sudo ./scripts/gen_graph_to_ssd.sh -b 0000:e1:00.0 -g 20 -a pr
#   sudo ./scripts/gen_graph_to_ssd.sh -b 0000:e1:00.0 -i /tmp/agile-graph/graph-s20 -a pr -o 1024

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWITCH_SCRIPT="$SCRIPT_DIR/switch_nvme_driver.sh"
GRAPH_GEN="$REPO_ROOT/demo/build/tools/graph-gen/agile_graph_gen"

# Defaults
BDF=""
SCALE=""
GEN_TYPE=""     # -g or -u
DEGREE=""
WEIGHTED=false
INPUT_PREFIX=""
SSD_OFFSET_PAGES=0
RESTORE_DRIVER="agile-nvme"
TMPDIR="/tmp/agile-graph"
APP="bfs"       # bfs or pr

usage() {
    echo "Usage: sudo $0 -b <BDF> {-g <scale> | -u <scale> | -i <prefix>} [options]"
    echo ""
    echo "Options:"
    echo "  -b <BDF>      PCIe BDF of target SSD (required)"
    echo "  -a <app>      Application: bfs (default) or pr"
    echo "  -g <scale>    Generate Kronecker graph (2^scale vertices)"
    echo "  -u <scale>    Generate Uniform random graph (2^scale vertices)"
    echo "  -k <degree>   Average degree (default: 16)"
    echo "  -w            Generate weighted graph (forced on for pr)"
    echo "  -i <prefix>   Use existing graph files (skip generation)"
    echo "  -o <offset>   SSD start offset in 4KB pages (default: 0)"
    echo "  -d <driver>   Driver to restore after write (default: agile-nvme)"
    echo "  -t <tmpdir>   Temp directory for graph files (default: /tmp/agile-graph)"
    echo "  -h            Show help"
    exit 1
}

while getopts "b:a:g:u:k:wi:o:d:t:h" opt; do
    case "$opt" in
        b) BDF="$OPTARG" ;;
        a) APP="$OPTARG" ;;
        g) GEN_TYPE="-g"; SCALE="$OPTARG" ;;
        u) GEN_TYPE="-u"; SCALE="$OPTARG" ;;
        k) DEGREE="$OPTARG" ;;
        w) WEIGHTED=true ;;
        i) INPUT_PREFIX="$OPTARG" ;;
        o) SSD_OFFSET_PAGES="$OPTARG" ;;
        d) RESTORE_DRIVER="$OPTARG" ;;
        t) TMPDIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# PR always needs weights
if [[ "$APP" == "pr" ]]; then
    WEIGHTED=true
fi

# Validation
if [[ -z "$BDF" ]]; then
    echo "Error: -b <BDF> is required."
    usage
fi

if [[ -z "$INPUT_PREFIX" && -z "$GEN_TYPE" ]]; then
    echo "Error: specify -g <scale>, -u <scale>, or -i <prefix>."
    usage
fi

if [[ "$APP" != "bfs" && "$APP" != "pr" ]]; then
    echo "Error: -a must be bfs or pr."
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

# ── Step 1: Generate graph (or use existing) ──
PREFIX=""
if [[ -n "$INPUT_PREFIX" ]]; then
    PREFIX="$INPUT_PREFIX"
    echo "=== Using existing graph: $PREFIX ==="
    if [[ ! -f "${PREFIX}.info.txt" ]]; then
        echo "Error: ${PREFIX}.info.txt not found."
        exit 1
    fi
else
    if [[ ! -x "$GRAPH_GEN" ]]; then
        echo "Error: graph-gen not found at $GRAPH_GEN"
        echo "Build it first: cd $REPO_ROOT/demo/build && make agile_graph_gen"
        exit 1
    fi

    mkdir -p "$TMPDIR"
    PREFIX="$TMPDIR/graph-s${SCALE}"

    echo "=== Generating graph (app=$APP) ==="
    GEN_ARGS="$GEN_TYPE $SCALE -o $PREFIX"
    [[ -n "$DEGREE" ]] && GEN_ARGS="$GEN_ARGS -k $DEGREE"
    [[ "$WEIGHTED" == true ]] && GEN_ARGS="$GEN_ARGS -w"
    echo "  $GRAPH_GEN $GEN_ARGS"
    $GRAPH_GEN $GEN_ARGS
fi

echo ""
echo "=== Graph info ==="
cat "${PREFIX}.info.txt"
echo ""

# Collect binary files to write based on app mode
FILES=()
if [[ "$APP" == "bfs" ]]; then
    FILES+=("${PREFIX}.offsets.bin")
    FILES+=("${PREFIX}.neighbors.bin")
    if [[ "$WEIGHTED" == true && -f "${PREFIX}.weights.bin" ]]; then
        FILES+=("${PREFIX}.weights.bin")
    fi
elif [[ "$APP" == "pr" ]]; then
    if [[ ! -f "${PREFIX}.neighbors.bin" ]]; then
        echo "Error: ${PREFIX}.neighbors.bin not found."
        exit 1
    fi
    if [[ ! -f "${PREFIX}.weights.bin" ]]; then
        echo "Error: ${PREFIX}.weights.bin not found. PR requires a weighted graph (-w)."
        exit 1
    fi
    FILES+=("${PREFIX}.neighbors.bin")
    FILES+=("${PREFIX}.weights.bin")
fi

# ── Step 2: Switch SSD to stock nvme driver ──
echo "=== Switching $BDF to nvme driver for dd access ==="
"$SWITCH_SCRIPT" nvme "$BDF"
sleep 2

# Find the /dev/nvmeXn1 block device for this BDF
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
    echo "Error: could not find block device for $BDF after switching to nvme driver."
    echo "Restoring driver..."
    "$SWITCH_SCRIPT" "$RESTORE_DRIVER" "$BDF"
    exit 1
fi

echo "  Block device: $NVME_DEV"

# ── Step 3: Write graph files to SSD using dd ──
echo ""
echo "=== Writing files to $NVME_DEV (app=$APP, offset=${SSD_OFFSET_PAGES} pages) ==="
OFFSET_PAGES=$SSD_OFFSET_PAGES
IDX=0
TOTAL=${#FILES[@]}
for f in "${FILES[@]}"; do
    IDX=$((IDX + 1))
    if [[ ! -f "$f" ]]; then
        echo "  WARNING: $f not found, skipping"
        continue
    fi
    FSIZE=$(stat -c%s "$f")
    echo "  [$IDX/$TOTAL] $(basename "$f") at page offset $OFFSET_PAGES ($FSIZE bytes)"
    dd if="$f" of="$NVME_DEV" bs=4096 oflag=direct seek="$OFFSET_PAGES" status=progress
    OFFSET_PAGES=$((OFFSET_PAGES + FSIZE / 4096))
    echo ""
done

TOTAL_SSD_PAGES=$((OFFSET_PAGES - SSD_OFFSET_PAGES))

# Write a layout info file for later reference
LAYOUT_FILE="${PREFIX}.ssd_layout.txt"
{
    echo "app= $APP"
    echo "ssd_bdf= $BDF"
    echo "ssd_dev= $NVME_DEV"
    echo "ssd_offset_pages= $SSD_OFFSET_PAGES"
    OFFSET_PAGES=$SSD_OFFSET_PAGES
    for f in "${FILES[@]}"; do
        if [[ -f "$f" ]]; then
            FSIZE=$(stat -c%s "$f")
            echo "$(basename "$f")= offset_pages=$OFFSET_PAGES size=$FSIZE"
            OFFSET_PAGES=$((OFFSET_PAGES + FSIZE / 4096))
        fi
    done
    echo "total_ssd_pages= $TOTAL_SSD_PAGES"
    if [[ "$APP" == "pr" ]]; then
        NEIGHBORS_SIZE=$(stat -c%s "${PREFIX}.neighbors.bin")
        echo "weight_offset_elems= $((NEIGHBORS_SIZE / 4))"
    fi
} > "$LAYOUT_FILE"
echo "=== SSD layout saved to $LAYOUT_FILE ==="
cat "$LAYOUT_FILE"

# ── Step 4: Restore SSD to original driver ──
echo ""
echo "=== Restoring $BDF to $RESTORE_DRIVER driver ==="
"$SWITCH_SCRIPT" "$RESTORE_DRIVER" "$BDF"

echo ""
echo "Done. Graph written to $NVME_DEV ($BDF) for $APP."
