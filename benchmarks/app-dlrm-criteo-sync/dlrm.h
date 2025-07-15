#include "agile_helpper.h"

#include <iostream>
#include <fstream>

#define CPU_CACHE_IMPL DisableCPUCache
#define SHARE_TABLE_IMPL DisableShareTable
// #define GPU_CACHE_IMPL SimpleGPUCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define GPU_CACHE_IMPL GPUClockReplacementCache<CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

#define AGILE_CTRL AgileCtrl<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_HOST AgileHost<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_CACHE_HIERARCHY AgileCacheHierarchy<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>
#define AGILE_BUF_ARR AgileBufArrayShared<GPU_CACHE_IMPL, CPU_CACHE_IMPL, SHARE_TABLE_IMPL>

// const unsigned int embedding_vec_dim = 64;

__global__ void prefetch_embedding_kernel(AGILE_CTRL *ctrl, \
    unsigned int *d_sparse_ptr, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    unsigned int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int batch_offset = batch_idx * sparse_input_dim;
    if(batch_idx >= batch_size){
        return;
    }

    for(unsigned int i = 0; i < sparse_input_dim; ++i){
        unsigned long idx = d_sparse_ptr[batch_offset + i] * embedding_vector_dim;
        unsigned long blk_idx = (idx) * sizeof(float) / ctrl->buf_size;
        ctrl->prefetch(0, blk_idx);
    }

}

__global__ void mean_embedding_kernel(AGILE_CTRL *ctrl, \
    unsigned int *d_sparse_ptr, float * d_vector_mean, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vector_idx = tid / embedding_vector_dim;
    unsigned int vector_offset = tid % embedding_vector_dim;

    if(vector_idx >= batch_size){
        return;
    }

    AgileLockChain chain;
    auto agileArr = ctrl->getArrayWrap<float>(chain);
    for(unsigned int i = 0; i < sparse_input_dim; ++i){
        unsigned long idx = d_sparse_ptr[vector_idx * sparse_input_dim + i] * embedding_vector_dim;
        float val = agileArr[0][idx + vector_offset];
        if(i == 0) {
            d_vector_mean[vector_idx * embedding_vector_dim + vector_offset] = val;
        } else {
            d_vector_mean[vector_idx * embedding_vector_dim + vector_offset] += val;
        }

        if(i == sparse_input_dim - 1){
            d_vector_mean[vector_idx * embedding_vector_dim + vector_offset] /= sparse_input_dim;
        }
    }
}


template<int threads_per_block>
void prefetch_embedding(bool enable, cudaStream_t & stream, \
    AGILE_CTRL *ctrl, unsigned int * d_sparse, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    if(!enable){
        return;
    }

    dim3 block(threads_per_block);
    dim3 grid(batch_size / threads_per_block + (batch_size % threads_per_block == 0 ? 0 : 1));
    prefetch_embedding_kernel<<<grid, block, 0, stream>>>(ctrl, d_sparse, table_offset, batch_size, sparse_input_dim, embedding_vector_dim);
}

template<int threads_per_block>
void mean_embedding(bool enable, cudaStream_t & stream, \
    AGILE_CTRL *ctrl, unsigned int * d_sparse, float * d_vector_mean, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    if(!enable){
        return;
    }
    
    unsigned int total_threads = batch_size * embedding_vector_dim;
    unsigned int total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
    dim3 block(threads_per_block);
    dim3 grid(total_blocks);
    // printf("total_threads: %d total_blocks: %d\n", total_threads, total_blocks);
    mean_embedding_kernel<<<grid, block, 0, stream>>>(ctrl, d_sparse, d_vector_mean, table_offset, batch_size, sparse_input_dim, embedding_vector_dim);

}


__global__ void sum_embedding_kernel(AGILE_CTRL *ctrl, \
    unsigned int *d_sparse_ptr, unsigned int * d_vector_sum, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vector_idx = tid / embedding_vector_dim;
    unsigned int vector_offset = tid % embedding_vector_dim;

    if(vector_idx >= batch_size){
        return;
    }

    AgileLockChain chain;
    auto agileArr = ctrl->getArrayWrap<unsigned int>(chain);
    for(unsigned int i = 0; i < sparse_input_dim; ++i){
        unsigned long idx = d_sparse_ptr[vector_idx * sparse_input_dim + i] * embedding_vector_dim;
        unsigned int val = agileArr[0][idx + vector_offset];
        if(i == 0) {
            d_vector_sum[vector_idx * embedding_vector_dim + vector_offset] = val;
        } else {
            d_vector_sum[vector_idx * embedding_vector_dim + vector_offset] += val;
        }

    }

}

//sum_embedding(enable_load, stream_compute, ctrl, d_sparse_in, d_vector_sum, 0, cfg.batch_size, 1, cfg.sparse_input_dim, cfg.embedding_vector_dim); 
template<int threads_per_block>
void sum_embedding(bool enable, cudaStream_t & stream, \
    AGILE_CTRL *ctrl, unsigned int * d_sparse, unsigned int * d_vector_sum, unsigned long table_offset, \
    unsigned int batch_size, unsigned int sparse_input_dim, unsigned int embedding_vector_dim)
{
    if(!enable){
        return;
    }
    // dim3 block(threads_per_block);
    // dim3 grid(batch_size / threads_per_block + (batch_size % threads_per_block == 0 ? 0 : 1));
    // sum_embedding_kernel<<<grid, block, 0, stream>>>(ctrl, d_sparse, d_vector_sum, table_offset, batch_size, sparse_input_dim, embedding_vector_dim);

    unsigned int total_threads = batch_size * embedding_vector_dim;
    unsigned int total_blocks = total_threads / threads_per_block + (total_threads % threads_per_block == 0 ? 0 : 1);
    dim3 block(threads_per_block);
    dim3 grid(total_blocks);
    // printf("total_threads: %d total_blocks: %d\n", total_threads, total_blocks);
    sum_embedding_kernel<<<grid, block, 0, stream>>>(ctrl, d_sparse, d_vector_sum, table_offset, batch_size, sparse_input_dim, embedding_vector_dim);
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

// void loadBinFile(void * ptr, std::string path, unsigned int bytes, unsigned long offset){
//     // load bytes binary file to ptr
//     std::ifstream file(path, std::ios::binary);
//     if (!file) {
//         std::cerr << "Error opening file!" << std::endl;
//         exit(0);
//     }
//     file.seekg(offset);
//     for(unsigned int i = 0; i < bytes; ++i) {
//         ((char*)ptr)[i] = file.get();
//     }
//     file.close();
//     std::cout << "load " << bytes << " bytes from file " << path << " to " << ptr << std::endl;
//     // std::cout << "load file " << path << " to " << ((unsigned int*)ptr)[0] << std::endl;
//     // std::cout << "load file " << path << " to " << ((float*)ptr)[0] << std::endl;
// }

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


// load binary file to host pointer
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