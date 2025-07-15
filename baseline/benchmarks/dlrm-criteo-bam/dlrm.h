

#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctrl.h>
#include <buffer.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>

#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>

#include <iostream>
#include <fstream>

// #define CPU_CACHE_IMPL DisableCPUCache
// #define SHARE_TABLE_IMPL DisableShareTable
// #define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
// // #define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

// #define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
// #define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
// #define AGILE_CACHE_HIERARCHY AgileCacheHierarchy<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
// #define AGILE_BUF_ARR AgileBufArrayShared<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

const unsigned int embedding_vec_dim = 64;

// __global__ void prefetch_embedding_kernel(AGILE_CTRL *ctrl, \
//     unsigned int *d_sparse_ptr, unsigned long table_offset, \
//     unsigned int batch_size, unsigned int table_size, unsigned int multi_hot_k, unsigned int embedding_dim)
// {

//     unsigned int b = blockIdx.x;
//     unsigned int t = blockIdx.y;
//     unsigned int k = threadIdx.x;
//     unsigned int offset = b * table_size * multi_hot_k + t * multi_hot_k;
//     unsigned int idx = d_sparse_ptr[offset + k];
//     unsigned long blk_idx = (idx * embedding_dim) * sizeof(float);
//     blk_idx = blk_idx / ctrl->buf_size;
//     blk_idx += table_offset * t; 
//     ctrl->prefetch(0, blk_idx);

// }



__global__ void sum_embedding_kernel(array_d_t<uint32_t>* da, \
    unsigned int *d_sparse_ptr, unsigned int * d_vector_sum, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vector_idx = tid / embedding_vector_dim;
    unsigned int vector_offset = tid % embedding_vector_dim;

    if(vector_idx >= batch_size){
        return;
    }

    for(unsigned int i = 0; i < sparse_input_dim; ++i){
        unsigned long idx = d_sparse_ptr[vector_idx * sparse_input_dim + i] * embedding_vector_dim;
        unsigned int val = (*da)[idx + vector_offset];
        if(i == 0) {
            d_vector_sum[vector_idx * embedding_vector_dim + vector_offset] = val;
        } else {
            d_vector_sum[vector_idx * embedding_vector_dim + vector_offset] += val;
        }

    }
}

template<int threads_per_block>
void sum_embedding(bool enable, cudaStream_t & stream, \
    array_d_t<uint32_t>* da, unsigned int * d_sparse, unsigned int * d_vector_sum, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    if(!enable){
        return;
    }
    unsigned int total_threads = batch_size * embedding_vector_dim;
    unsigned int total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
    dim3 block(threads_per_block);
    dim3 grid(total_blocks);

    sum_embedding_kernel<<<grid, block, 0, stream>>>(da, d_sparse, d_vector_sum, table_offset, batch_size, sparse_input_dim, embedding_vector_dim);
}


__global__ void mean_embedding_kernel(array_d_t<uint32_t>* da, \
    unsigned int *d_sparse_ptr, float * d_vector_mean, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vector_idx = tid / embedding_vector_dim;
    unsigned int vector_offset = tid % embedding_vector_dim;

    if(vector_idx >= batch_size){
        return;
    }

    for(unsigned int i = 0; i < sparse_input_dim; ++i){
        unsigned long idx = d_sparse_ptr[vector_idx * sparse_input_dim + i] * embedding_vector_dim;
        unsigned int val = (*da)[idx + vector_offset];
        if(i == 0) {
            d_vector_mean[vector_idx * embedding_vector_dim + vector_offset] = __uint_as_float(val);
        } else {
            d_vector_mean[vector_idx * embedding_vector_dim + vector_offset] += __uint_as_float(val);
        }

        if(i == sparse_input_dim - 1){
            d_vector_mean[vector_idx * embedding_vector_dim + vector_offset] /= sparse_input_dim;
        }

    }
}


template<int threads_per_block>
void mean_embedding(bool enable, cudaStream_t & stream, \
    array_d_t<uint32_t>* da, unsigned int * d_sparse, float * d_vector_mean, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    if(!enable){
        return;
    }
    unsigned int total_threads = batch_size * embedding_vector_dim;
    unsigned int total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
    dim3 block(threads_per_block);
    dim3 grid(total_blocks);

    mean_embedding_kernel<<<grid, block, 0, stream>>>(da, d_sparse, d_vector_mean, table_offset, batch_size, sparse_input_dim, embedding_vector_dim);
}

__global__ void concate_kernel(float * d_bottom_out, float * d_vector_mean, float * d_all_features, \
    unsigned int batch_size, unsigned int embedding_vector_dim)
{
    unsigned int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch_idx >= batch_size){
        return;
    }

    float * line = d_all_features + batch_idx * (2 * embedding_vector_dim);
    for(unsigned int i = 0; i < embedding_vector_dim; ++i){
        line[i] = d_vector_mean[batch_idx * embedding_vector_dim + i];
        line[i + embedding_vector_dim] = d_bottom_out[batch_idx * embedding_vector_dim + i];
    }

}

template<int threads_per_block>
void concate(bool enable, cudaStream_t & stream, \
    float * d_bottom_out, float * d_vector_mean, float * d_all_features, \
    unsigned int batch_size, unsigned int embedding_vector_dim)
{
    if(!enable){
        return;
    }
    dim3 block(threads_per_block);
    dim3 grid(batch_size / threads_per_block + (batch_size % threads_per_block == 0 ? 0 : 1));
    concate_kernel<<<grid, block, 0, stream>>>(d_bottom_out, d_vector_mean, d_all_features, batch_size, embedding_vector_dim);

}

void loadBinFile(std::string path, void* ptr, unsigned int bytes, unsigned long offset){
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    in.seekg(offset);
    unsigned int read_size = 0;
    while (read_size < bytes) {
        in.read((char*)ptr + read_size, bytes - read_size);
        read_size += in.gcount();
    }
    in.close();
    std::cout << "load " << bytes << " bytes from file " << path << " to " << ptr << std::endl;
}

__global__ void relu_kernel(float* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        data[idx] = fmaxf(0.0f, data[idx]);
}

void relu(bool enable, cudaStream_t & stream, \
    float * d_activation, unsigned int size)
{
    if(!enable){
        return;
    }

    relu_kernel<<<size / 1024 + (size % 1024 == 0 ? 0 : 1), 1024, 0, stream>>>(d_activation, size);

}

__global__ void sigmoid_kernel(float* data, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

void sigmoid(bool enable, cudaStream_t & stream, \
    float * d_activation, unsigned int size)
{
    if(!enable){
        return;
    }
    sigmoid_kernel<<<size / 1024 + (size % 1024 == 0 ? 0 : 1), 1024, 0, stream>>>(d_activation, size);
}