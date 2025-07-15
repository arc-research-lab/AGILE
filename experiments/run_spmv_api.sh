#!/bin/bash

# settings

datasets=/home/zhuoping/workspace/git/gapbs/pr-graph/g3
bam=baseline/bam/build/bin

nvme_blk_size=8
queue_num=32
queue_depth=256
cache_page_size=$((4*1024*1024 / ${nvme_blk_size}))
max_itr=20
save_path=./results/pr/${nvme_blk_size}

repeat=8


max_ssd_page=$((2*1024*1024 / ${nvme_blk_size}))

# run PR compute core

for i in ${datasets}/*
do
    echo ${i}
    u=${i##*/}
    rm -rf ${save_path}/${u}
    mkdir -p ${save_path}/${u}/hbm
    info=($(cat ${i}/pr-info.txt))
    echo "node ${info[4]} edge ${info[7]} $u" 

    for((r = 0; r < repeat; r++))
    do
        echo "./bin/test-pr -nn ${info[4]} -en ${info[7]} -it 1 -i ${i}/pr-rows.bin -ef ${i}/pr-cols.bin -wf ${i}/pr-vals.bin | tee log"
        ./bin/test-pr -nn ${info[4]} -en ${info[7]} -it 1 -i ${i}/pr-rows.bin -ef ${i}/pr-cols.bin -wf ${i}/pr-vals.bin | tee log
        mv log ${save_path}/${u}/hbm/cuda-res-pr-${r}.log
        mv cuda-pr.txt ${save_path}/${u}/hbm/cuda-pr-${r}.txt
    done
done


