#!/bin/bash

# --buf-per-blk 8 -i 5000 -t 32
# 10 to 1000 with step 10

nvme=(01 02 03 04 e1 e2 e3 e4)

for i in ${nvme[@]}
do

echo "compute_itr,sync_time,async_time,ctc,speedup" > sweep_ctc_${i}.csv

for ((j=200; j<=1350; j = j+10))
do
    sudo ./build/examples/ctc/agile_demo_ctc -d /dev/AGILE-NVMe-0000:${i}:00.0 -q 2 --queue-depth 512 --buf-per-blk 2 -i 1000 -t 32 --compute-itr ${j} | tee log
    # ...
    # [INFO][CTC]      === Summary ===
    # [INFO][CTC]      Compute only: 39151 us
    # [INFO][CTC]      Load only:    232608 us
    # [INFO][CTC]      Sync (L+C):   266704 us
    # [INFO][CTC]      Async (L+C):  229240 us
    # [INFO][CTC]      CTC (Compute-to-Communication): 0.1683
    # [INFO][CTC]      Async speedup over sync: 1.1634x
    # [INFO][CTC]      kernel finish

    # parse log
    sync_time=$(grep "Sync (L+C)" log | awk '{print $4}')
    async_time=$(grep "Async (L+C)" log | awk '{print $4}')
    ctc=$(grep "CTC (Compute-to-Communication)" log | awk '{print $4}')
    speedup=$(grep "Async speedup over sync" log | awk '{print $6}' | sed 's/x//')
    echo "${j},${sync_time},${async_time},${ctc},${speedup}" >> sweep_ctc_${i}.csv
done

done