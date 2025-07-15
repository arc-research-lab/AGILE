#!/bin/bash

bam_bench=baseline/bam/build/bin/nvm-test-dlrm-criteo-bam-bench

itr_start=1000
itr_end=1000
batch_size=2048
embedding_vec_dim=1024
gpu_cahce_lines=$((65536*8))
slot_size=4096
num_queue=32
queue_depth=256
repeat=1

bottom_layer="512-512-512"
top_layer="1024-1024-1024"

save_path=./results/dlrm/BS_${batch_size}_EVD_${embedding_vec_dim}_GPU_CACHE_${gpu_cahce_lines}_SLOT_${slot_size}_NUM_QUEUE_${num_queue}_QUEUE_DEPTH_${queue_depth}_BL_${bottom_layer}_TL_${top_layer}/
mkdir -p ${save_path}

for((i = itr_start; i <= itr_end; i=i*10))
do
    mkdir ${save_path}/itr_${i}
    for((j = 0; j < repeat; j++))
    do
        # # sync version disable load 
        # echo "./scripts/run.sh bin/app-dlrm-criteo-sync -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} -el 0 | tee ${save_path}/itr_${i}/sync-compute_repeat_${j}.log"
        # ./scripts/run.sh bin/app-dlrm-criteo-sync -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} -el 0 | tee ${save_path}/itr_${i}/sync-compute_repeat_${j}.log

        # # sync version disable compute
        # echo "./scripts/run.sh bin/app-dlrm-criteo-sync -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} -ec 0 | tee ${save_path}/itr_${i}/sync-load_repeat_${j}.log"
        # ./scripts/run.sh bin/app-dlrm-criteo-sync -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} -ec 0 | tee ${save_path}/itr_${i}/sync-load_repeat_${j}.log

        # sync 
        echo "./scripts/run.sh bin/app-dlrm-criteo-sync -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} | tee ${save_path}/itr_${i}/sync_repeat_${j}.log"
        ./scripts/run.sh bin/app-dlrm-criteo-sync -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} | tee ${save_path}/itr_${i}/sync_repeat_${j}.log

        # async
         echo "./scripts/run.sh bin/app-dlrm-criteo-async -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} | tee ${save_path}/itr_${i}/async_repeat_${j}.log"
        ./scripts/run.sh bin/app-dlrm-criteo-async -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} | tee ${save_path}/itr_${i}/async_repeat_${j}.log

        # bam
        sudo ${bam_bench} -itr ${i} -bs ${batch_size} -evd ${embedding_vec_dim} -gsn ${gpu_cahce_lines} -qn ${num_queue} -qd ${queue_depth} -bl ${bottom_layer} -tl ${top_layer} | tee ${save_path}/itr_${i}/bam_repeat_${j}.log

    done
done




