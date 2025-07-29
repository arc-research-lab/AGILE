#!/bin/bash

make
sudo rmmod agile_nvme_driver
sudo insmod agile_nvme_driver.ko