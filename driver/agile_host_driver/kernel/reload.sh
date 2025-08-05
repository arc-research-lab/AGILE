#!/bin/bash

make
sudo rmmod agile_host
sudo insmod agile_host.ko