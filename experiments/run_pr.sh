#!/bin/bash
datasets=/home/zhuoping/workspace/git/gapbs/pr-graph/g3
bam=baseline/bam/build/bin

nvme_blk_size=8
queue_num=32
queue_depth=256
cache_page_size=$((4*1024*1024 / ${nvme_blk_size}))
max_itr=2
save_path=./results/pr/${nvme_blk_size}

repeat=8




max_ssd_page=$((1024*1024 / ${nvme_blk_size}))

for i in ${datasets}/*
do  
    echo ${i}
    u=${i##*/}

    mkdir -p ${save_path}/${u}/bam
    mkdir -p ${save_path}/${u}/agile

    info=($(cat ${i}/pr-info.txt))
    echo "node ${info[4]} edge ${info[7]} $u" 
    write graph to ssd
    echo "./scripts/run.sh bin/block-write -i ${i}/pr-cols.bin -ss $((${nvme_blk_size} * 512)) -bn ${max_ssd_page} -qn 128 -gsn 65536"
    ./scripts/run.sh bin/block-write -i ${i}/pr-cols.bin -ss $((${nvme_blk_size} * 512)) -bn ${max_ssd_page} -qn 128 -gsn 65536

    #write val to ssd
    echo "./scripts/run.sh bin/block-write -i ${i}/pr-vals.bin -ss $((${nvme_blk_size} * 512)) -bn ${max_ssd_page} -bo ${max_ssd_page} -qn 128 -gsn 65536"
    ./scripts/run.sh bin/block-write -i ${i}/pr-vals.bin -ss $((${nvme_blk_size} * 512)) -bn ${max_ssd_page} -bo ${max_ssd_page} -qn 128 -gsn 65536

    for((r = 0; r < repeat; r++))
    do 
        echo sudo ${bam}/nvm-test-pagerank-bench -n ${info[4]} -e ${info[7]} -i ${i}/pr-rows.bin -s ${max_ssd_page} --page_size=$((512 * ${nvme_blk_size})) --pages=${cache_page_size} --queue_depth=${queue_depth} --num_queues=${queue_num} --threads=768 -m ${max_itr} -o bam-pr-res.txt | tee log
        sudo ${bam}/nvm-test-pagerank-bench -n ${info[4]} -e ${info[7]} -i ${i}/pr-rows.bin -s ${max_ssd_page} --page_size=$((512 * ${nvme_blk_size})) --pages=${cache_page_size} --queue_depth=${queue_depth} --num_queues=${queue_num} --threads=768 -m ${max_itr} -o bam-pr-res.txt | tee log
        # sudo mv bam-pr-res.txt ${save_path}/bam/${u}/bam-pr-res-${r}.txt
        sudo mv log ${save_path}/${u}/bam/log-${r}
        # sleep 5
        echo "./scripts/run.sh bin/pr -nn ${info[4]} -en ${info[7]} -it ${max_itr} -i ${i}/pr-rows.bin -gsn ${cache_page_size} -wo $((${max_ssd_page} * ${nvme_blk_size} * 512 / 4))  -qn ${queue_num} -qd ${queue_depth} -bd 128 -td 768 -ss $((${nvme_blk_size} * 512)) -sbn $((${max_ssd_page} * 2)) | tee log"
        ./scripts/run.sh bin/pr -nn ${info[4]} -en ${info[7]} -it ${max_itr} -i ${i}/pr-rows.bin -gsn ${cache_page_size} -wo $((${max_ssd_page} * ${nvme_blk_size} * 512 / 4))  -qn ${queue_num} -qd ${queue_depth} -bd 128 -td 768 -ss $((${nvme_blk_size} * 512)) -sbn $((${max_ssd_page} * 2)) | tee log
        # sudo mv res-pr.txt ${save_path}/agile/${u}/agile-pr-res-${r}.txt
        sudo mv log ${save_path}/${u}/agile/log-${r}

    done
    # break;
    # fi
    max_ssd_page=$((max_ssd_page*2))
done
