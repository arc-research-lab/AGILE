#!/bin/bash
datasets=/mnt/nvme/zhuoping/gapbs/pr-graph/g3

size=512

thread_pr(){
    i=${1}
    u=${i##*/}
    # echo ${u}
    info=($(cat ${i}/pr-info.txt))
    echo "node ${info[4]} edge ${info[7]}" 
    echo ./bin/cpu-pr -nn ${info[4]} -en ${info[7]} -nf ${i}/pr-rows.bin -ef ${i}/pr-cols.bin -o cpu-res-pr-${u}.txt -onv ${i}/pr-vals.bin -mi 20
    ./bin/cpu-pr -nn ${info[4]} -en ${info[7]} -nf ${i}/pr-rows.bin -ef ${i}/pr-cols.bin -o cpu-res-pr-${u}.txt -onv ${i}/pr-vals.bin -mi 20 | tee pr-log-${u}
    mv pr-log-${u} ${i}/
    mv cpu-res-pr-${u}.txt ${i}/cpu-res-pr.txt
    truncate -s ${size}M ${i}/pr-vals.bin
    size=$((${size} * 2))
    break;
}

for i in ${datasets}/*
do
    thread_pr ${i}
done

wait