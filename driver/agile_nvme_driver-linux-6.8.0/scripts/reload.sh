#!/bin/bash

echo "0000:f1:00.0" | sudo tee /sys/bus/pci/devices/0000:f1:00.0/driver/unbind
echo "0000:f2:00.0" | sudo tee /sys/bus/pci/devices/0000:f2:00.0/driver/unbind
echo "0000:f3:00.0" | sudo tee /sys/bus/pci/devices/0000:f3:00.0/driver/unbind
echo "0000:f4:00.0" | sudo tee /sys/bus/pci/devices/0000:f4:00.0/driver/unbind


echo "0000:e1:00.0" | sudo tee /sys/bus/pci/devices/0000:e1:00.0/driver/unbind
echo "0000:e2:00.0" | sudo tee /sys/bus/pci/devices/0000:e2:00.0/driver/unbind
echo "0000:e3:00.0" | sudo tee /sys/bus/pci/devices/0000:e3:00.0/driver/unbind
echo "0000:e4:00.0" | sudo tee /sys/bus/pci/devices/0000:e4:00.0/driver/unbind

sudo rmmod agile_nvme_driver
sudo insmod ./kernel/agile_nvme_driver.ko