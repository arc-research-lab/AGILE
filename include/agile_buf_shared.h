#ifndef AGILE_BUF_SHARED
#define AGILE_BUF_SHARED
#include "agile_helpper.h"
#include "agile_buf.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define AGILE_SHARED_BUF_EMPTY  0
#define AGILE_SHARED_BUF_INIT  1
#define AGILE_SHARED_BUF_READY  2

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileBufArrayShared {
public:
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl;
    AgileBuf *buf;
    unsigned int shared_status; // check if the buffer is ready to use in the block
    unsigned int size;

    __device__ void static init(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl, \
            AgileBufArrayShared<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *obj, AgileBuf *buf, unsigned int size)
    {
        cg::thread_block block = cg::this_thread_block();
        if(block.thread_rank() == 0){
            obj->buf = buf;
            obj->ctrl = ctrl;
            obj->size = size;
            atomicExch(&(obj->shared_status), AGILE_SHARED_BUF_INIT);
        }
    }

    __device__ unsigned long load(NVME_DEV_IDX_TYPE dev_id, SSDBLK_TYPE blk_idx_start, AgileLockChain &chain){
        unsigned long start, end;
        LOG_CLOCK(start);
        cg::thread_block block = cg::this_thread_block();
        unsigned int tid = threadIdx.x;
        if(tid < size){
            // printf("load %p %d %d\n", &buf[tid], buf[tid].status, tid);
            AgileBufPtr bufPtr;
            // buf[tid].resetStatus();
            bufPtr.buf = &buf[tid];

            ctrl->asyncRead(dev_id, blk_idx_start + tid, &bufPtr, &chain);
        }
        LOG_CLOCK(end);
        return end - start;
    }

    __device__ unsigned long wait(){
        unsigned long start, end;
        LOG_CLOCK(start);
        __syncthreads();
        unsigned int tid = threadIdx.x;
        if(tid < size){
            buf[tid].wait();
        }
        __syncthreads();
        LOG_CLOCK(end);
        return end - start;
    }

};

#endif