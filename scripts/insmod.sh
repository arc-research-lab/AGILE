#!/bin/bash
cd driver/gdrcopy
sudo bash insmod.sh
cd ../agile_mem
sudo insmod agile_mem.ko
# echo "0000:4b:00.0" | sudo tee /sys/bus/pci/devices/0000:4b:00.0/driver/unbind
# echo "0000:b1:00.0" | sudo tee /sys/bus/pci/devices/0000:b1:00.0/driver/unbind
# echo "0000:98:00.0" | sudo tee /sys/bus/pci/devices/0000:98:00.0/driver/unbind
cd ../agile_nvme
# sudo insmod agile_nvme.ko