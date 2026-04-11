#!/bin/bash
#
# Switch a single NVMe SSD driver between: nvme (kernel stock), agile-nvme, bam-nvme
#
# Usage:
#   sudo ./switch_nvme_driver.sh <driver> <BDF>
#   ./switch_nvme_driver.sh list
#
# Examples:
#   sudo ./switch_nvme_driver.sh agile-nvme 0000:e1:00.0
#   sudo ./switch_nvme_driver.sh nvme       0000:e1:00.0
#   sudo ./switch_nvme_driver.sh bam-nvme   0000:e4:00.0
#   ./switch_nvme_driver.sh list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Paths to driver kernel modules (try versioned dirs first, then generic)
AGILE_NVME_KO=""
BAM_NVME_KO=""

for d in "$REPO_ROOT"/driver/agile_nvme_driver-linux-*/kernel "$REPO_ROOT/driver/agile_nvme_driver/kernel"; do
    if [[ -f "$d/agile_nvme_driver.ko" ]]; then
        AGILE_NVME_KO="$d/agile_nvme_driver.ko"
        break
    fi
done

for d in "$REPO_ROOT"/driver/bam_nvme_driver-linux-*; do
    if [[ -f "$d/libnvm.ko" ]]; then
        BAM_NVME_KO="$d/libnvm.ko"
        break
    fi
done

# Sysfs driver names (must match .name in struct pci_driver)
SYSFS_DRV_NVME="nvme"
SYSFS_DRV_AGILE="AGILE NVMe SSD Driver"
SYSFS_DRV_BAM="libnvm helper"

# Kernel module names
MOD_AGILE="agile_nvme_driver"
MOD_BAM="libnvm"

usage() {
    echo "Usage: sudo $0 <driver> <BDF>"
    echo "       sudo $0 list"
    echo ""
    echo "Drivers:"
    echo "  nvme        - switch back to kernel stock NVMe driver"
    echo "  agile-nvme  - switch to AGILE NVMe driver"
    echo "  bam-nvme    - switch to BAM NVMe driver (libnvm)"
    echo "  list        - list NVMe PCI devices and current drivers"
    echo ""
    echo "BDF: PCIe Bus:Device.Function address, e.g. 0000:e1:00.0"
    exit 1
}

# List all NVMe-class PCI devices (class 0x0108)
list_nvme_devices() {
    local bdfs=()
    for dev in /sys/bus/pci/devices/*/class; do
        local class
        class=$(cat "$dev" 2>/dev/null || true)
        # NVMe class: 0x010802
        if [[ "$class" == 0x0108* ]]; then
            bdfs+=("$(basename "$(dirname "$dev")")")
        fi
    done
    echo "${bdfs[@]}"
}

# Get current driver for a BDF
get_current_driver() {
    local bdf="$1"
    local driver_link="/sys/bus/pci/devices/$bdf/driver"
    if [[ -L "$driver_link" ]]; then
        basename "$(readlink "$driver_link")"
    else
        echo "(none)"
    fi
}

# Print NVMe device info
print_device_info() {
    local bdfs
    read -ra bdfs <<< "$(list_nvme_devices)"
    if [[ ${#bdfs[@]} -eq 0 ]]; then
        echo "No NVMe PCI devices found."
        return
    fi
    printf "%-16s %-24s %s\n" "BDF" "DRIVER" "DEVICE"
    printf "%-16s %-24s %s\n" "---" "------" "------"
    for bdf in "${bdfs[@]}"; do
        local drv
        drv=$(get_current_driver "$bdf")
        local dev_info
        dev_info=$(lspci -s "$bdf" 2>/dev/null | sed "s/^$bdf //" || echo "unknown")
        printf "%-16s %-24s %s\n" "$bdf" "$drv" "$dev_info"
    done
}

# Unbind a device from its current driver
unbind_device() {
    local bdf="$1"
    local drv
    drv=$(get_current_driver "$bdf")
    if [[ "$drv" == "(none)" ]]; then
        echo "  $bdf: not bound to any driver, skipping unbind"
        return
    fi
    # If unbinding from BAM driver, remove the /dev/libnvmN node
    if [[ "$drv" == "$SYSFS_DRV_BAM" ]]; then
        local node
        node=$(dmesg | grep "Character device /dev/libnvm" | tail -1 | grep -oP '/dev/\S+(?= created)')
        if [[ -n "$node" && -e "$node" ]]; then
            echo "  Removing BAM device node $node"
            rm -f "$node"
        fi
    fi
    echo "  $bdf: unbinding from $drv"
    echo "$bdf" | tee /sys/bus/pci/devices/"$bdf"/driver/unbind
}

# Bind a device to a driver using driver_override + drivers_probe
bind_device() {
    local bdf="$1"
    local drv_name="$2"
    echo "  $bdf: setting driver_override to '$drv_name'"
    echo "$drv_name" | tee /sys/bus/pci/devices/"$bdf"/driver_override
    echo "  $bdf: triggering probe"
    echo "$bdf" | tee /sys/bus/pci/drivers_probe
}

# Clear driver_override for a device
clear_driver_override() {
    local bdf="$1"
    echo "" | tee /sys/bus/pci/devices/"$bdf"/driver_override > /dev/null
}

# Ensure a kernel module is loaded
ensure_module_loaded() {
    local mod="$1"
    local ko_path="$2"
    local extra_args="${3:-}"
    if lsmod | grep -qw "$mod"; then
        echo "  Module $mod already loaded"
    else
        echo "  Loading $ko_path $extra_args"
        insmod "$ko_path" $extra_args
    fi
}

# ---- Main ----

if [[ $# -lt 1 ]]; then
    usage
fi

TARGET="$1"
shift

if [[ "$TARGET" == "list" ]]; then
    print_device_info
    exit 0
fi

if [[ "$TARGET" != "nvme" && "$TARGET" != "agile-nvme" && "$TARGET" != "bam-nvme" ]]; then
    echo "Error: unknown driver '$TARGET'"
    usage
fi

if [[ $# -ne 1 ]]; then
    echo "Error: exactly one BDF address is required."
    usage
fi

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "Error: this script must be run as root (sudo)."
    exit 1
fi

BDF="$1"

if [[ ! -d "/sys/bus/pci/devices/$BDF" ]]; then
    echo "Error: PCI device $BDF does not exist."
    exit 1
fi

echo "Target driver : $TARGET"
echo "Device        : $BDF"
echo ""

# Step 1: Unbind device from current driver
echo "=== Unbinding device ==="
unbind_device "$BDF"

# Step 2: Ensure target module is loaded, then bind device
echo ""
echo "=== Binding to target driver ==="
case "$TARGET" in
    nvme)
        modprobe nvme 2>/dev/null || true
        bind_device "$BDF" "$SYSFS_DRV_NVME"
        ;;
    agile-nvme)
        if [[ -z "$AGILE_NVME_KO" ]]; then
            echo "Error: agile_nvme_driver.ko not found under $REPO_ROOT/driver/"
            exit 1
        fi
        ensure_module_loaded "$MOD_AGILE" "$AGILE_NVME_KO"
        bind_device "$BDF" "$SYSFS_DRV_AGILE"
        ;;
    bam-nvme)
        if [[ -z "$BAM_NVME_KO" ]]; then
            echo "Error: libnvm.ko not found under $REPO_ROOT/driver/"
            exit 1
        fi
        ensure_module_loaded "$MOD_BAM" "$BAM_NVME_KO" "max_num_ctrls=64"
        bind_device "$BDF" "$SYSFS_DRV_BAM"

        # Auto-create /dev/libnvmN device node via mknod
        echo ""
        echo "=== Creating BAM device node ==="
        sleep 1
        line=$(dmesg | grep "Character device /dev/libnvm" | tail -1)
        if [[ -n "$line" ]]; then
            dev_name=$(echo "$line" | grep -oP '/dev/\S+(?= created)')
            major=$(echo "$line" | grep -oP 'mknod /dev/\S+ c \K\d+')
            minor=$(echo "$line" | grep -oP 'mknod /dev/\S+ c \d+ \K\d+')
            if [[ -n "$dev_name" && -n "$major" && -n "$minor" ]]; then
                [[ -e "$dev_name" ]] && rm -f "$dev_name"
                echo "  mknod $dev_name c $major $minor"
                mknod "$dev_name" c "$major" "$minor"
                chmod 666 "$dev_name"
            fi
        else
            echo "  WARNING: no BAM device node found in dmesg"
        fi
        ;;
esac

# Step 4: Verify
echo ""
echo "=== Current state ==="
sleep 1
drv=$(get_current_driver "$BDF")
dev_info=$(lspci -s "$BDF" 2>/dev/null | sed "s/^$BDF //" || echo "unknown")
printf "%-16s %-24s %s\n" "BDF" "DRIVER" "DEVICE"
printf "%-16s %-24s %s\n" "---" "------" "------"
printf "%-16s %-24s %s\n" "$BDF" "$drv" "$dev_info"

echo ""
echo "Done."
