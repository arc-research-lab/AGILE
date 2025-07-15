#!/bin/bash

for((v = 65536*4; v <= 65536*4; v = v * 2))
do
    ./experiments/run_dlrm.sh ${v} 
done


