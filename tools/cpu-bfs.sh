#!/bin/bash
datasets=/home/zhuoping/workspace/git/gapbs/bfs-graph

for i in ${datasets}/u/*
do
    info=($(head -n 1 ${i}/bfs-info.txt))
    echo "node ${info[4]} edge ${info[7]}" 
    echo "./bin/cpu-bfs -nn ${info[4]} -en ${info[7]} -nf ${i}/bfs-rows.bin -ef ${i}/bfs-cols.bin | tee log"
    ./bin/cpu-bfs -nn ${info[4]} -en ${info[7]} -nf ${i}/bfs-rows.bin -ef ${i}/bfs-cols.bin | tee log
    # mv log ${i}/
    mv cpu-res-bfs.bin ${i}/
done