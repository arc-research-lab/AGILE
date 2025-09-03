#!/bin/bash

make
sudo rmmod agile_kernel
sudo insmod agile_kernel.ko