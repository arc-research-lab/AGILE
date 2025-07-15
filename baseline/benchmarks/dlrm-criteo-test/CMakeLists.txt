cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (test-dlrm-criteo-test-module test-dlrm-criteo-test-bench "main.cu;runner.cu;")
# target_link_libraries(test-dlrm-benchmark-module PRIVATE libcublas)

make_benchmark_choice (test-dlrm-criteo-test test-dlrm-criteo-test-smartio test-dlrm-criteo-test-module)
