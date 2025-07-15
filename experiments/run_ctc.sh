#!/bin/bash
itr=10000
bpblk=64
qn=64
td=1024
bd=1
start=100
end=8000
step=100
echo "citr, Sync load time, Sync compute time, Sync time, Sync CTC, Sync Speedup, Async load time, Async compute time, Async time, Async Speedup, Async CTC, Speedup (Async vs Sync)" > ./results/ctc-block-itr-${itr}-bpblk-${bpblk}-qn-${qn}-td-${td}-bd-${bd}-start-${start}-end-${end}-step-${step}.csv

function get_results(){
    data=($(tail -n 3 log))
    sync_time=${data[2]}
    async_time=${data[6]}
    speedup=${data[9]}
    echo "$sync_time $async_time $speedup"
}

for((i=${start};i<${end};i=i+${step}))
do
    # disable load and measure compute time only
    echo "./scripts/run.sh bin/ctc-block -itr ${itr} -bpblk ${bpblk} -qn ${qn} -td ${td} -bd ${bd} -el 0 -citr ${i}"
    ./scripts/run.sh bin/ctc-block -itr ${itr} -bpblk ${bpblk} -qn ${qn} -td ${td} -bd ${bd} -el 0 -citr ${i} | tee log
    read sync_compute async_compute compute_speedup <<< $(get_results)

    # disable compute and measure load time only
    echo "./scripts/run.sh bin/ctc-block -itr ${itr} -bpblk ${bpblk} -qn ${qn} -td ${td} -bd ${bd} -ec 0 -citr ${i}"
    ./scripts/run.sh bin/ctc-block -itr ${itr} -bpblk ${bpblk} -qn ${qn} -td ${td} -bd ${bd} -ec 0 -citr ${i} | tee log
    read sync_load async_load load_speedup <<< $(get_results)

    # measure both load and compute time
    echo "./scripts/run.sh bin/ctc-block -itr ${itr} -bpblk ${bpblk} -qn ${qn} -td ${td} -bd ${bd} -citr ${i}"
    ./scripts/run.sh bin/ctc-block -itr ${itr} -bpblk ${bpblk} -qn ${qn} -td ${td} -bd ${bd} -citr ${i} | tee log
    read sync_time async_time speedup <<< $(get_results)


    sync_speedup=$(echo "scale=2; (${sync_compute} + ${sync_load})/${sync_time}" | bc)
    async_speedup=$(echo "scale=2; (${async_compute} + ${async_load})/${async_time}" | bc)

    sync_CTC=$(echo "scale=2; ${sync_compute}/${sync_load}" | bc)
    async_CTC=$(echo "scale=2; ${async_compute}/${sync_load}" | bc)

    echo "${i}, ${sync_load}, ${sync_compute}, ${sync_time}, ${sync_CTC}, ${sync_speedup}, ${async_load}, ${async_compute}, ${async_time}, ${async_CTC}, ${async_speedup}, ${speedup}" >> ./results/ctc-block-itr-${itr}-bpblk-${bpblk}-qn-${qn}-td-${td}-bd-${bd}-start-${start}-end-${end}-step-${step}.csv
    
done
