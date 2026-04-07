#!/bin/bash
make
sudo rmmod agile_gpu_krnl
sudo insmod agile_gpu_krnl.ko
ls /dev/AGILE*