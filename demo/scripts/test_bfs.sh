#!/bin/bash
#
# Write graph to SSD and run BFS with AGILE and/or BAM drivers.
#
# Usage:
#   sudo ./test_bfs.sh -b <BDF> -i <info_file> [options]
#   sudo ./test_bfs.sh -b <BDF> -p <prefix>    [options]
#
# Examples:
#   sudo ./scripts/test_bfs.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20
#   sudo ./scripts/test_bfs.sh -b 0000:e1:00.0 -i /tmp/agile-graph/graph-s20.info.txt -o 1024
#   sudo ./scripts/test_bfs.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20 -m agile
#   sudo ./scripts/test_bfs.sh -b 0000:e1:00.0 -p /tmp/agile-graph/graph-s20 -m bam

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWITCH_SCRIPT="$SCRIPT_DIR/switch_nvme_driver.sh"

# Demo BFS binaries
BFS_AGILE="$REPO_ROOT/demo/build/examples/bfs-agile/agile_demo_bfs"
BFS_BAM="$REPO_ROOT/demo/build/examples/bfs-bam/agile_demo_bfs_bam"
BFS_GPU="$REPO_ROOT/demo/build/examples/bfs-gpu/agile_demo_bfs_gpu"

# gdrcopy library path (needed by AGILE runtime)
GDRCOPY_LIB="$REPO_ROOT/driver/gdrcopy/src"

# Defaults
BDF=""
INFO_FILE=""
PREFIX=""
SSD_OFFSET_PAGES=0
MODE="both"        # agile, bam, or both
START_NODE=0
QUEUE_NUM=15
QUEUE_DEPTH=512
SKIP_WRITE=false

usage() {
    echo "Usage: sudo $0 -b <BDF> {-i <info_file> | -p <prefix>} [options]"
    echo ""
    echo "Options:"
    echo "  -b <BDF>        PCIe BDF of target SSD (required)"
    echo "  -i <info_file>  Path to .info.txt from graph-gen"
    echo "  -p <prefix>     Graph file prefix (auto-appends .info.txt)"
    echo "  -o <offset>     SSD offset for neighbors data in 4KB pages (default: 0)"
    echo "  -m <mode>       Test mode: agile, bam, or both (default: both)"
    echo "  -s <node>       BFS start node (default: 0)"
    echo "  -q <queues>     Number of NVMe queues (default: 15)"
    echo "  -S              Skip writing graph to SSD (assume already written)"
    echo "  -h              Show help"
    exit 1
}

while getopts "b:i:p:o:m:s:q:Sh" opt; do
    case "$opt" in
        b) BDF="$OPTARG" ;;
        i) INFO_FILE="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        o) SSD_OFFSET_PAGES="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        s) START_NODE="$OPTARG" ;;
        q) QUEUE_NUM="$OPTARG" ;;
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
if [[ ! -f "$NEIGHBORS_FILE" ]]; then
    echo "Error: neighbors file not found: $NEIGHBORS_FILE"
    exit 1
fi

echo "============================================"
echo " AGILE BFS Test"
echo "============================================"
echo "BDF           : $BDF"
echo "Info file     : $INFO_FILE"
echo "Neighbors     : $NEIGHBORS_FILE"
echo "SSD offset    : ${SSD_OFFSET_PAGES} pages ($((SSD_OFFSET_PAGES * 4096)) bytes)"
echo "Mode          : $MODE"
echo "Start node    : $START_NODE"
echo ""

# ── Step 1: Write neighbors to SSD ──
if [[ "$SKIP_WRITE" == false ]]; then
    echo "=== Step 1: Writing neighbors to SSD ==="

    # Switch to stock nvme driver for dd access
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
        echo "Error: could not find block device for $BDF"
        exit 1
    fi

    FSIZE=$(stat -c%s "$NEIGHBORS_FILE")
    echo "  Device: $NVME_DEV"
    echo "  dd if=$NEIGHBORS_FILE of=$NVME_DEV bs=4096 seek=$SSD_OFFSET_PAGES (size=$FSIZE)"
    dd if="$NEIGHBORS_FILE" of="$NVME_DEV" bs=4096 oflag=direct seek="$SSD_OFFSET_PAGES" status=progress
    echo ""
    echo "  Write complete."
else
    echo "=== Step 1: Skipped (graph already on SSD) ==="
fi
echo ""

# ── Step 2: Run BFS with AGILE driver ──
if [[ "$MODE" == "agile" || "$MODE" == "both" ]]; then
    echo "=== Step 2: BFS with AGILE driver ==="

    if [[ ! -x "$BFS_AGILE" ]]; then
        echo "Error: AGILE BFS binary not found at $BFS_AGILE"
        echo "Build: cd $REPO_ROOT/demo/build && make agile_demo_bfs"
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
    echo "  Running: agile_demo_bfs --info $INFO_FILE -d $AGILE_DEV --blk-offset $SSD_OFFSET_PAGES -s $START_NODE -q $QUEUE_NUM"
    echo ""

    LD_LIBRARY_PATH="$GDRCOPY_LIB" "$BFS_AGILE" \
        --info "$INFO_FILE" \
        -d "$AGILE_DEV" \
        --blk-offset "$SSD_OFFSET_PAGES" \
        -s "$START_NODE" \
        -q "$QUEUE_NUM" \
        --queue-depth "$QUEUE_DEPTH"

    echo ""
    echo "  AGILE BFS complete."
    echo ""
fi

# ── Step 3: Run BFS with BAM driver ──
if [[ "$MODE" == "bam" || "$MODE" == "both" ]]; then
    echo "=== Step 3: BFS with BAM driver ==="

    if [[ ! -x "$BFS_BAM" ]]; then
        echo "Error: BAM BFS binary not found at $BFS_BAM"
        echo "Build: cd $REPO_ROOT/demo/build && make agile_demo_bfs_bam"
        exit 1
    fi

    "$SWITCH_SCRIPT" bam-nvme "$BDF"
    sleep 1

    # Find the BAM device node (created by switch_nvme_driver.sh)
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
    echo "  Running: agile_demo_bfs_bam --info $INFO_FILE -d $BAM_DEV -s $START_NODE -q $QUEUE_NUM"
    echo ""

    "$BFS_BAM" \
        --info "$INFO_FILE" \
        -d "$BAM_DEV" \
        -s "$START_NODE" \
        -q "$QUEUE_NUM" \
        --queue-depth "$QUEUE_DEPTH"

    echo ""
    echo "  BAM BFS complete."
    echo ""
fi

# ── Step 4: Run BFS on GPU (no SSD) ──
if [[ "$MODE" == "both" ]]; then
    echo "=== Step 4: BFS on GPU memory ==="

    if [[ ! -x "$BFS_GPU" ]]; then
        echo "  Warning: GPU BFS binary not found at $BFS_GPU — skipping"
    else
        echo "  Running: agile_demo_bfs_gpu --info $INFO_FILE -s $START_NODE"
        echo ""

        "$BFS_GPU" \
            --info "$INFO_FILE" \
            -s "$START_NODE" \
            -o "res-bfs-gpu.bin"

        echo ""
        echo "  GPU BFS complete."
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
    AGILE_RES="res-bfs.bin"
    BAM_RES="res-bfs-bam.bin"
    if [[ -f "$AGILE_RES" && -f "$BAM_RES" ]]; then
        AGILE_MD5=$(md5sum "$AGILE_RES" | awk '{print $1}')
        BAM_MD5=$(md5sum "$BAM_RES" | awk '{print $1}')
        echo "  AGILE: $AGILE_MD5  ($AGILE_RES)"
        echo "  BAM:   $BAM_MD5  ($BAM_RES)"
        if [[ "$AGILE_MD5" == "$BAM_MD5" ]]; then
            echo "  MATCH: results are identical"
        else
            echo "  DIFFER: results differ"
        fi
    else
        echo "  Could not compare: one or both result files missing"
    fi
fi

echo ""
echo "============================================"
echo " Done."
echo "============================================"
