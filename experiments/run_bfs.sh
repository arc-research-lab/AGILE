#!/bin/bash

datasets=/home/zhuoping/workspace/git/gapbs/bfs-graph/u
bam=baseline/bam/build/bin


nvme_blk_size=8
queue_num=32
queue_depth=256
cache_page_size=$((4*1024*1024 / ${nvme_blk_size}))
save_path=./results/bfs/${nvme_blk_size}

repeat=8

max_ssd_page=$((2*1024*1024 / ${nvme_blk_size}))

for i in ${datasets}/*
do
    # echo ${i}
    u=${i##*/}
    echo ${u}
    size=${u##*-}
    # if [ ${size} -lt 22 ]
    # then
    #     max_ssd_page=$((max_ssd_page*2))
    #     continue;
    # fi
    echo "./scripts/run.sh bin/block-write -i ${i}/bfs-cols.bin -ss $((${nvme_blk_size} * 512)) -bn ${max_ssd_page} -qn 128 -gsn 65536"
    ./scripts/run.sh bin/block-write -i ${i}/bfs-cols.bin -ss $((${nvme_blk_size} * 512)) -bn ${max_ssd_page} -qn 128 -gsn 65536
    
    mkdir -p ${save_path}/${u}/bam/
    mkdir -p ${save_path}/${u}/agile/
    info=($(cat ${i}/bfs-info.txt))
    echo "node ${info[4]} edge ${info[7]} $u" 

    for((r = 0; r < repeat; r++))
    do 
        echo "${bam}/nvm-test-bfs-bench -n ${info[4]} -e ${info[7]} -i ${i}/bfs-rows.bin --page_size=$((512 * ${nvme_blk_size})) --pages=${cache_page_size} --queue_depth=${queue_depth} --num_queues=${queue_num}"
        sudo ${bam}/nvm-test-bfs-bench -n ${info[4]} -e ${info[7]} -i ${i}/bfs-rows.bin --page_size=$((512 * ${nvme_blk_size})) --pages=${cache_page_size} --queue_depth=${queue_depth} --num_queues=${queue_num} | tee log
        # sudo mv gpu-res-bfs.bin ${save_path}/${u}/gpu-res-bfs-${r}.bin
        md5sum gpu-res-bfs.bin >> log
        sudo mv log ${save_path}/${u}/bam/log-${r}
        sleep 1
        echo "./scripts/run.sh bin/bfs -nn ${info[4]} -en ${info[7]} -i ${i}/bfs-rows.bin -gsn ${cache_page_size}  -qn ${queue_num} -qd ${queue_depth} -bd 128 -ss $((${nvme_blk_size} * 512)) -sbn ${max_ssd_page}"
        ./scripts/run.sh bin/bfs -nn ${info[4]} -en ${info[7]} -i ${i}/bfs-rows.bin -gsn ${cache_page_size}  -qn ${queue_num} -qd ${queue_depth} -bd 128 -ss $((${nvme_blk_size} * 512)) -sbn ${max_ssd_page} | tee log
        md5sum res-bfs.bin >> log
        sudo mv log ${save_path}/${u}/agile/log-${r}
    done


    max_ssd_page=$((max_ssd_page*2))
done