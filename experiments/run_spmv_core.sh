#!/bin/bash

# settings

datasets=/home/zhuoping/workspace/git/gapbs/bfs-graph/g
bam=baseline/bam/build/bin

nvme_blk_size=8
queue_num=32
queue_depth=256
cache_page_size=$((4*1024*1024 / ${nvme_blk_size}))
save_path=./results/bfs/${nvme_blk_size}

repeat=8


max_ssd_page=$((2*1024*1024 / ${nvme_blk_size}))

# run BFS compute core

for i in ${datasets}/*
do
    u=${i##*/}
    echo ${u}
    size=${u##*-}
    info=($(cat ${i}/bfs-info.txt))
    echo "node ${info[4]} edge ${info[7]} $u" 
    mkdir -p ${save_path}/${u}/hbm/

    for((r = 0; r < repeat; r++))
    do
        echo "./bin/test-bfs -nn ${info[4]} -en ${info[7]} -i ${i}/bfs-rows.bin -ie ${i}/bfs-cols.bin | tee log"
        ./bin/test-bfs -nn ${info[4]} -en ${info[7]} -i ${i}/bfs-rows.bin -ie ${i}/bfs-cols.bin | tee log
        md5sum cuda-bfs.bin >> log
        mv log ${save_path}/${u}/hbm/cuda-res-bfs-${r}.log
    done
done


