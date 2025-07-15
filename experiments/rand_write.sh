#!/bin/bash

req_start=1
req_end=$((131072*2))
repeat=8

save_path=./results/rand_write
mkdir -p ${save_path}


for((i=req_start;i<=req_end;i=i*2))
do
    mkdir -p ${save_path}/one-ssd/req_${i}
    mkdir -p ${save_path}/two-ssd/req_${i}
    mkdir -p ${save_path}/three-ssd/req_${i}
    for((j=0;j<repeat;j++))
    do
        # one SSD
        echo "./scripts/run.sh bin/bench-write -so 0 -sn 1 -rn ${i} | tee ${save_path}/one-ssd/req_${i}/dell-${j}.log"
        ./scripts/run.sh bin/bench-write -so 0 -sn 1 -rn ${i} | tee ${save_path}/one-ssd/req_${i}/dell-${j}.log

        # two SSDs
        echo "./scripts/run.sh bin/bench-write -so 0 -sn 2 -rn ${i} | tee ${save_path}/two-ssd/req_${i}/dell-sx0-${j}.log"
        ./scripts/run.sh bin/bench-write -so 0 -sn 2 -rn ${i} | tee ${save_path}/two-ssd/req_${i}/dell-sx0-${j}.log

        # three SSDs
        echo "./scripts/run.sh bin/bench-write -so 0 -sn 3 -rn ${i} | tee ${save_path}/three-ssd/req_${i}/dell-sx0-sx1-${j}.log"
        ./scripts/run.sh bin/bench-write -so 0 -sn 3 -rn ${i} | tee ${save_path}/three-ssd/req_${i}/dell-sx0-sx1-${j}.log
    done
done