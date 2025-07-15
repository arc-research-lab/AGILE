#include "agile_ctrl.h"
#include "agile_swcache.h"
#include "agile_cache_hierarchy.h"
#include "agile_helpper.h"
#include "agile_shared_table_buf.h"

__device__ void AgileCtrlBase::issueReadNvme2GPU(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
    unsigned long phy_addr = this->cache_hierarchy->gpu_cache->physical_addr + ((unsigned long) cache_idx) * ((unsigned long) (this->buf_size));
    this->dev[dev_idx].issueRead(GPU_DEVICE, 0, ((unsigned long)ssd_blk_idx) * (unsigned long)(this->buf_size / 512), this->buf_size / 512, phy_addr, chain);
}

__device__ void AgileCtrlBase::issueWriteGPU2Nvme(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
    unsigned long phy_addr = this->cache_hierarchy->gpu_cache->physical_addr + ((unsigned long) cache_idx) * ((unsigned long) (this->buf_size));
    this->dev[dev_idx].issueWrite(GPU_DEVICE, 0, ((unsigned long)ssd_blk_idx) * (unsigned long)(this->buf_size / 512), this->buf_size / 512, phy_addr, chain);
}

__device__ void AgileCtrlBase::issueWriteCPU2Nvme(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
    // unsigned int table_idx = cache_idx * this->buf_size / this->cpu_page_size;
    // unsigned long table_offset = cache_idx * this->buf_size % this->cpu_page_size;
    // unsigned long phy_addr = this->cache_hierarchy->cpu_cache->phy_addr_table[table_idx] + table_offset;
    unsigned long phy_addr = this->cache_hierarchy->cpu_cache->physical_addr + ((unsigned long) cache_idx) * ((unsigned long) (this->buf_size));
    this->dev[dev_idx].issueWrite(CPU_DEVICE, 0, ((unsigned long)ssd_blk_idx) * ((unsigned long)this->buf_size / 512), this->buf_size / 512, phy_addr, chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::startAgile(unsigned int &AGILE_BID){
    // if(threadIdx.x == 0){
    //     AGILE_BID = atomicAdd(&(this->g_bid), 1);
    // }
    // __syncthreads();
    if(blockIdx.x < num_block_for_polling){
        this->pollingService(blockIdx.x);
        return false;
    }

    // if(threadIdx.x == 0){
    //     AGILE_BID -= num_block_for_polling;
    // }
    // __syncthreads();

    return true;
}

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ bool AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::stopAgile(unsigned int AGILE_BID){
//     // __syncwarp();
//     __syncthreads();
//     if(threadIdx.x == 0){
//         atomicAdd(&(this->finished_blocks_num), 1);
//         LOGGING(atomicAdd(&(logger->finished_block), 1));
//         // printf("block finish %d\n", AGILE_BID);
//     }
//     return true;
// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::waitCpl(unsigned int queue_idx, unsigned int AGILE_BID){
    
#if FAKE_NVME
    unsigned int pos_g = atomicAdd(&(this->list->pairs[queue_idx].sq.prev_sq_pos), 0);
    unsigned int sqpos_g = atomicAdd(&(this->list->pairs[queue_idx].sq.g_pos), 0);
    unsigned int prev_pos = pos_g % this->list->pairs[queue_idx].sq.depth;
    unsigned int gap = sqpos_g - pos_g;
    // unsigned int fake_sqdb = atomicAdd(&(this->list->pairs[queue_idx].sq.fake_sqdb), 0);
    
    // GPU_ASSERT((prev_pos == fake_sqdb) ? pos_g == sqpos_g : true, "assert fail");

    for(int i = 0; i < gap; ++i){

        volatile unsigned int * cmd_ptr = (volatile unsigned int *) this->list->pairs[queue_idx].sq.data + 16 * prev_pos;
        
        unsigned int cmd_type = cmd_ptr[0] & 0x7f;
        unsigned int cid = cmd_ptr[0] >> 16;
        unsigned long phy_addr = cmd_ptr[6];
        phy_addr |= ((unsigned long)cmd_ptr[7]) << 32;
        unsigned int ssd_blk_idx = cmd_ptr[10];
        unsigned int blocks = cmd_ptr[12] + 1;
        unsigned int device_type = (cmd_ptr[13] >> 8) & 0x1; // TODO: check if this works
        unsigned int table_idx = (cmd_ptr[13] >> 9) & 0x7FFFFF;
        
        GPU_ASSERT(blocks * 512 == this->buf_size, "buffer size mismatch");
        
        AgileLockChain chain;
        unsigned int cache_idx; // = gpu_data_offset / this->buf_size;
        
        void * data_ptr;
        if(device_type == GPU_DEVICE){
            unsigned long gpu_data_offset = phy_addr - this->cache_hierarchy->gpu_cache->physical_addr;
            cache_idx = gpu_data_offset / this->buf_size;
            data_ptr = this->cache_hierarchy->gpu_cache->getCacheDataPtr(cache_idx);
        }else{
            unsigned long data_offset = phy_addr - this->cache_hierarchy->cpu_cache->physical_addr;
            cache_idx = data_offset / this->buf_size;
            data_ptr = this->cache_hierarchy->cpu_cache->getCacheDataPtr(cache_idx);
        }

        // printf("cmdtype: %d devicetype: %d, cache_idx: %d\n", cmd_type, device_type, cache_idx);
        
        uint4 * src;
        uint4 * dst;

        // unsigned int emu_idx = ssd_blk_idx * 512 / emu_size;
        unsigned int emu_offset = (ssd_blk_idx * 512 % emu_size);
        if(device_type == GPU_DEVICE){
            this->cache_hierarchy->gpu_cache->acquireBaseLock_lockStart(cache_idx, &chain); // fake issue
        }else{
            this->cache_hierarchy->cpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
        }
        
        if(cmd_type == AGILE_NVME_READ){
            LOGGING(atomicAdd(&(logger->finished_read), 1));
            src = reinterpret_cast<uint4*>(this->emu_nvme + emu_offset);
            dst = reinterpret_cast<uint4*>(data_ptr);
            
        } else { // emulate write
            // printf("%s:%d write gpu cache %d %f\n", __FILE__, __LINE__, cache_idx, ((float*)data_ptr)[1]);
            dst = reinterpret_cast<uint4*>(this->emu_nvme + emu_offset);
            src = reinterpret_cast<uint4*>(data_ptr);
            LOGGING(atomicAdd(&(logger->finished_write), 1));
        }

        
        for(unsigned int i = 0; i < this->buf_size / sizeof(uint4); ++i){
            dst[i] = src[i];
        }

        if(cmd_type == AGILE_NVME_READ){
            if(device_type == GPU_DEVICE){
                this->cache_hierarchy->gpu_cache->finishReading_inLockArea(cache_idx); // fake issue
                this->cache_hierarchy->gpu_cache->propagateAgileBuf_inLockArea(cache_idx); // fake issue
                __threadfence_system();
                this->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain); 
            }else{
                // this->cache_hierarchy->cpu_cache->
            }
        }else{
            if(device_type == GPU_DEVICE){
                this->cache_hierarchy->gpu_cache->finishWriting_inLockArea(cache_idx); // fake issue
                __threadfence_system();
                this->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain); 
            }else{
                this->cache_hierarchy->cpu_cache->finishWriting_inLockArea(cache_idx, AGILE_CPUCACHE_READY);
                this->cache_hierarchy->cpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain);
            }
        }
        
        LOGGING(atomicAdd(&(logger->cpl_count[queue_idx]), 1));
        
        this->list->pairs[queue_idx].sq.cmd_locks[prev_pos].remoteRelease();
        prev_pos = (prev_pos + 1) % this->list->pairs[queue_idx].sq.depth;
        // printf("update prevpos, qid:, %d, pos:, %d\n", queue_idx, prev_pos);
    }

    atomicAdd(&(this->list->pairs[queue_idx].sq.prev_sq_pos), gap);
    
    return gap != 0;
#else
    unsigned int pos_g = atomicAdd(&(this->list->pairs[queue_idx].sq.prev_sq_pos), 0);
    unsigned int sqpos_g = atomicAdd(&(this->list->pairs[queue_idx].sq.g_pos), 0);
    unsigned int gap = min(sqpos_g - pos_g, this->list->pairs[queue_idx].cq.depth / 8);
    unsigned int prev_pos = pos_g % this->list->pairs[queue_idx].sq.depth;
    unsigned int new_cqdb = pos_g % this->list->pairs[queue_idx].cq.depth;
    // LOGGING(atomicAdd(&(logger->cq_running[queue_idx]), 1));
    unsigned int finished_num = 0;
    for(int i = 0; i < gap; ++i){ // make sure the gap is not too large, otherwise, the cqdb will not be updated, and the cq will be full
        volatile unsigned int * cpl = (volatile unsigned int *)(((void*)this->list->pairs[queue_idx].cq.data) + new_cqdb * 16);
        unsigned int count = 0;
        while(((cpl[3] >> 16) & 0x1) == this->list->pairs[queue_idx].cq.phase){
            count++;
            if(count == 1000000){
                count = 0;
                // LOGGING(atomicExch(&(logger->gap[queue_idx]), sqpos_g - pos_g - i));
                // LOGGING(atomicExch(&(logger->cq_waiting[queue_idx]), new_cqdb));
                // LOGGING(atomicAdd(&(logger->cq_running[queue_idx]), 1));
                // *(this->list->pairs[queue_idx].cq.cqdb) = new_cqdb;
                LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                goto update_cqdb;
            }
            // 
        } // wait for cpl

        unsigned int entry3 = cpl[3];
        unsigned int entry2 = cpl[2];
        new_cqdb += 1;
        if(new_cqdb == this->list->pairs[queue_idx].cq.depth){
            new_cqdb = 0;
            this->list->pairs[queue_idx].cq.phase = ~(this->list->pairs[queue_idx].cq.phase) & 0x1;
        }
        if(((entry3 >> 17) & 0x1) != 0){ // error happens
            printf("find nvme error: %.8x %.8x %.8x %.8x\n", cpl[0], cpl[1], cpl[2], cpl[3]);
        }
        unsigned int cid = (entry3 & 0x0000ffff);
        unsigned int sq_identifier = entry2 >> 16;
        unsigned int sq_head_pointer = (entry2 & 0xffff);
        // get command info using cid

        volatile unsigned int * cmd_ptr = (volatile unsigned int *) this->list->pairs[queue_idx].sq.data + 16 * cid;
        unsigned int cmd_type = cmd_ptr[0] & 0x7f;
        unsigned long phy_addr = cmd_ptr[6];
        phy_addr |= ((unsigned long)cmd_ptr[7]) << 32;
        unsigned int ssd_blk_idx = cmd_ptr[10];
        unsigned int blocks = cmd_ptr[12] + 1;
        unsigned int device_type = (cmd_ptr[13] >> 8) & 0x1; // TODO: check if this works

        // printf("qidx %d pos: %d phyaddrL %lx type: %d cid %d ssd_blk_idx: %d \n", queue_idx, cid, phy_addr, cmd_type, cid, ssd_blk_idx);
        // release the command slot
        wati_status(this->list->pairs[queue_idx].sq.cmd_status + cid, AGILE_CMD_STATUS_ISSUED, AGILE_CMD_STATUS_EMPTY);
        __threadfence_system();
        this->list->pairs[queue_idx].sq.cmd_locks[cid].remoteRelease();
        AgileLockChain chain;
        if(device_type == CPU_DEVICE && cmd_type == AGILE_NVME_WRITE){
            LOGGING(atomicAdd(&(logger->finished_write), 1));
            unsigned long data_offset = phy_addr - this->cache_hierarchy->cpu_cache->physical_addr;
            // unsigned int cache_idx = table_idx * this->cache_hierarchy->ctrl->cpu_page_size / this->buf_size;
            unsigned int cache_idx = data_offset / this->buf_size;
            this->cache_hierarchy->cpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
            this->cache_hierarchy->cpu_cache->finishWriting_inLockArea(cache_idx, AGILE_CPUCACHE_READY);
            this->cache_hierarchy->cpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain);
        } else if (device_type == GPU_DEVICE && cmd_type == AGILE_NVME_READ) {
            
            LOGGING(atomicAdd(&(logger->finished_read), 1));
            unsigned long gpu_data_offset = phy_addr - this->cache_hierarchy->gpu_cache->physical_addr;
            unsigned int cache_idx = gpu_data_offset / this->buf_size;
            
            this->cache_hierarchy->gpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
            this->cache_hierarchy->gpu_cache->finishReading_inLockArea(cache_idx); // fake issue
            this->cache_hierarchy->gpu_cache->propagateAgileBuf_inLockArea(cache_idx); // fake issue
            __threadfence_system();
            this->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain); 
        } else if (device_type == GPU_DEVICE && cmd_type == AGILE_NVME_WRITE){
            LOGGING(atomicAdd(&(logger->finished_write), 1));
            unsigned long gpu_data_offset = phy_addr - this->cache_hierarchy->gpu_cache->physical_addr;
            unsigned int cache_idx = gpu_data_offset / this->buf_size;
            this->cache_hierarchy->gpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
            this->cache_hierarchy->gpu_cache->finishWriting_inLockArea(cache_idx);
            this->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain); 
        }
        // LOGGING(atomicAdd(&(logger->cpl_count[queue_idx]), 1));
        finished_num++;

    }
    update_cqdb:
    if(finished_num != 0){
        atomicAdd(&(this->list->pairs[queue_idx].sq.prev_sq_pos), finished_num);
        *(this->list->pairs[queue_idx].cq.cqdb) = new_cqdb;
        // unsigned int last_cq_pos = atomicAdd(&(logger->curr_cq_pos[queue_idx]), 0);
        // atomicExch(&(logger->curr_cq_pos[queue_idx]), new_cqdb);
        // atomicExch(&(logger->last_cq_pos[queue_idx]), last_cq_pos);
    }
    return finished_num != 0;
#endif
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::warpService(unsigned int AGILE_BID, unsigned int start, unsigned int end){
    unsigned int warp_idx = threadIdx.x % 32;
    bool valid_run = false;
    for(int i = start + warp_idx; i < end; i += 32){
        valid_run = this->waitCpl(i, AGILE_BID);
        // __nanosleep(5000);
    }
    __ballot_sync(0xFFFFFFFF, 1);
    return valid_run || (start + warp_idx >= end);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::pollingServiceLast(unsigned int AGILE_BID){
    unsigned int stop_sig = 0;
    unsigned int start_idx, end_idx;
    unsigned int groups = this->list->num_pairs / 32 + (this->list->num_pairs % 32 == 0 ? 0 : 1); // 32 queues are packed and assigned to the same queue
    unsigned available_warps = this->num_block_for_polling * threads_per_block / 32;
    unsigned int groups_per_warp = groups / available_warps;
    groups_per_warp = groups_per_warp == 0 ? 1 : groups_per_warp;
    unsigned int warp_idx = AGILE_BID * threads_per_block / 32 + threadIdx.x / 32;
    start_idx = (warp_idx * groups_per_warp) * 32;
    end_idx = min(((warp_idx + 1) * groups_per_warp) * 32, this->list->num_pairs);
    unsigned int counter = 0;
    unsigned int mask = __ballot_sync(0xFFFFFFFF, 1);
    
    bool valid = this->warpService(AGILE_BID, start_idx, end_idx);
    
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::pollingService(unsigned int AGILE_BID){
    
    unsigned int start_idx, end_idx;
    unsigned int groups = this->list->num_pairs / 32 + (this->list->num_pairs % 32 == 0 ? 0 : 1); // 32 queues are packed and assigned to the same queue
    unsigned available_warps = this->num_block_for_polling * threads_per_block / 32;
    unsigned int groups_per_warp = groups / available_warps;
    groups_per_warp = groups_per_warp == 0 ? 1 : groups_per_warp;
    unsigned int warp_idx = AGILE_BID * threads_per_block / 32 + threadIdx.x / 32;
    start_idx = (warp_idx * groups_per_warp) * 32;
    end_idx = min(((warp_idx + 1) * groups_per_warp) * 32, this->list->num_pairs);
    unsigned int counter = 0;
    do {
        unsigned int mask = __ballot_sync(0xFFFFFFFF, 1);
        bool valid = this->warpService(AGILE_BID, start_idx, end_idx);

    } while (*((volatile unsigned int *)this->stop_signal) == 0);
    // if(threadIdx.x % 32 == 0){
    //     LOGGING(atomicAdd(&(logger->finished_agile_warp), 1));
    // }
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ unsigned int AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::waitCpl2(unsigned int queue_idx, unsigned int warp_idx, unsigned int cq_offset, unsigned int mask){
    volatile unsigned int * cpl = (volatile unsigned int *)(((void*)this->list->pairs[queue_idx].cq.data) + (cq_offset + warp_idx) * 16);

    unsigned int flag = (mask >> warp_idx) & 0x1;
    if(flag == 1){ // this thread finished its job in current warp
        return 1;
    }

    if(((cpl[3] >> 16) & 0x1) == this->list->pairs[queue_idx].cq.phase){ // if the cpl is not ready
        return 0;
    }


    unsigned int entry2 = cpl[2];
    unsigned int entry3 = cpl[3];
    if(((entry3 >> 17) & 0x1) != 0){ // error happens
        printf("find nvme error: %.8x %.8x %.8x %.8x\n", cpl[0], cpl[1], cpl[2], cpl[3]);
    }


    unsigned int cid = (entry3 & 0x0000ffff);
    unsigned int sq_identifier = entry2 >> 16;
    unsigned int sq_head_pointer = (entry2 & 0xffff);

    volatile unsigned int * cmd_ptr = (volatile unsigned int *) this->list->pairs[queue_idx].sq.data + 16 * cid;
    unsigned int cmd_type = cmd_ptr[0] & 0x7f;
    unsigned long phy_addr = cmd_ptr[6];
    phy_addr |= ((unsigned long)cmd_ptr[7]) << 32;
    unsigned int ssd_blk_idx = cmd_ptr[10];
    unsigned int blocks = cmd_ptr[12] + 1;
    unsigned int device_type = (cmd_ptr[13] >> 8) & 0x1; // TODO: check if this works

    wati_status(this->list->pairs[queue_idx].sq.cmd_status + cid, AGILE_CMD_STATUS_ISSUED, AGILE_CMD_STATUS_EMPTY);
    __threadfence_system();
    this->list->pairs[queue_idx].sq.cmd_locks[cid].remoteRelease();
    AgileLockChain chain;

    if(device_type == CPU_DEVICE && cmd_type == AGILE_NVME_WRITE){
        LOGGING(atomicAdd(&(logger->finished_write), 1));
        unsigned long data_offset = phy_addr - this->cache_hierarchy->cpu_cache->physical_addr;
        // unsigned int cache_idx = table_idx * this->cache_hierarchy->ctrl->cpu_page_size / this->buf_size;
        unsigned int cache_idx = data_offset / this->buf_size;
        this->cache_hierarchy->cpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
        this->cache_hierarchy->cpu_cache->finishWriting_inLockArea(cache_idx, AGILE_CPUCACHE_READY);
        this->cache_hierarchy->cpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain);
    } else if (device_type == GPU_DEVICE && cmd_type == AGILE_NVME_READ) {
        LOGGING(atomicAdd(&(logger->finished_read), 1));
        
        unsigned long gpu_data_offset = phy_addr - this->cache_hierarchy->gpu_cache->physical_addr;
        unsigned int cache_idx = gpu_data_offset / this->buf_size;
        
        this->cache_hierarchy->gpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
        this->cache_hierarchy->gpu_cache->finishReading_inLockArea(cache_idx); // fake issue
        this->cache_hierarchy->gpu_cache->propagateAgileBuf_inLockArea(cache_idx); // fake issue
        __threadfence_system();
        this->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain); 
    } else if (device_type == GPU_DEVICE && cmd_type == AGILE_NVME_WRITE){
        LOGGING(atomicAdd(&(logger->finished_write), 1));
        unsigned long gpu_data_offset = phy_addr - this->cache_hierarchy->gpu_cache->physical_addr;
        unsigned int cache_idx = gpu_data_offset / this->buf_size;
        this->cache_hierarchy->gpu_cache->acquireBaseLock_lockStart(cache_idx, &chain);
        this->cache_hierarchy->gpu_cache->finishWriting_inLockArea(cache_idx);
        this->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(cache_idx, &chain); 
    }

    return 1;
}

// each warp is responsible for one cq queue, check 32 cq entries in one iteration
template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::warpService2(unsigned int queue_idx, unsigned int warp_idx, unsigned int & offset, unsigned int & mask){
    unsigned int processed = waitCpl2(queue_idx, warp_idx, offset, mask);
    mask = __ballot_sync(0xFFFFFFFF, processed);
    if (mask == 0xFFFFFFFF) {
        mask = 0;
        offset += 32; // % depth
        if(offset == this->list->pairs[queue_idx].cq.depth){
            offset = 0;
            this->list->pairs[queue_idx].cq.phase = (~(this->list->pairs[queue_idx].cq.phase)) & 0x1;
        }
        if(warp_idx == 0){
            atomicAdd(&(this->list->pairs[queue_idx].sq.prev_sq_pos), 32);
            *(this->list->pairs[queue_idx].cq.cqdb) = offset;
        }
        
    }
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::pollingService2(){
    unsigned int stop_sig = 0;
    unsigned int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    unsigned int warp_idx = threadIdx.x % 32;
    unsigned int queue_idx = warp_id;
    do {
        unsigned int mask = __ballot_sync(0xFFFFFFFF, 1);
        bool valid = this->warpService2(warp_id, warp_idx, this->list->pairs[queue_idx].cq.pos_offset, this->list->pairs[queue_idx].cq.mask); // change queue idx in the future
        if(warp_idx == 0){
            stop_sig = *((volatile unsigned int *)this->stop_signal);
        }
        stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
    } while (stop_sig == 0);
}


// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::pollingService3(){
//     unsigned int stop_sig = 0;
//     unsigned int warp_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
//     unsigned int warp_idx = threadIdx.x % 32;
//     unsigned int queue_idx = warp_id;
//     do {
//         unsigned int mask = __ballot_sync(0xFFFFFFFF, 1);
//         bool valid = this->warpService2(warp_id, warp_idx, this->list->pairs[queue_idx].cq.pos_offset, this->list->pairs[queue_idx].cq.mask); // change queue idx in the future
//         if(warp_idx == 0){
//             stop_sig = *((volatile unsigned int *)this->stop_signal);
//         }
//         stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
//     } while (stop_sig == 0);
// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__host__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::setComputeBlocks(BID_TYPE compute_blocks) {
    this->compute_blocks = compute_blocks;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__host__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::setAgileBlocks(BID_TYPE agile_blocks) {
    this->num_block_for_polling = agile_blocks;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__host__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::setThreadsPreBlock(TID_TYPE threads_per_block) {
    this->threads_per_block = threads_per_block;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::asyncRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    
    buf_ptr->resetStatus();
    buf_ptr->setReadTag(dev_idx, ssd_blk_idx);
    LOGGING(atomicAdd(&(logger->wating_buffer), 1));
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->cacheRead(dev_idx, ssd_blk_idx, buf_ptr, chain);
}

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::asyncReadShared(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtrShared & buf_shared, AgileLockChain & chain){
//     cg::thread_block block = cg::this_thread_block();
//     if(block.thread_rank() == 0){
//         AgileBufPtr bufPtr(buf_shared.buf);
//         bufPtr.resetStatus();
//         bufPtr.setReadTag(dev_idx, ssd_blk_idx);
//         static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->cacheRead(dev_idx, ssd_blk_idx, &bufPtr, &chain);
//         buf_shared.initReady();
//     }
// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ unsigned int AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::prefetch_core(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){

    unsigned int gpu_cache_idx = -1;
     
    AgileLockChain chain;
    bool gpu_hit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &gpu_cache_idx, &chain);
    if(gpu_hit){
        LOGGING(atomicAdd(&(logger->gpu_cache_hit), 1));
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
        return gpu_cache_idx;
    }

    unsigned int cpu_cache_idx = -1;
    bool cpu_hit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &cpu_cache_idx, &chain);
    if(cpu_hit){
        LOGGING(atomicAdd(&(logger->cpu_cache_hit), 1));
        this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
        this->cache_hierarchy->moveData(this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx), this->getGPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx));
        this->getGPUCacheBasePtr()->finishReading_inLockArea(gpu_cache_idx);
        this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, &chain);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
        return gpu_cache_idx;
    }

    LOGGING(atomicAdd(&(logger->cpu_cache_miss), 1));
    LOGGING(atomicAdd(&(logger->gpu_cache_miss), 1));

    this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, &chain);
    this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);

    this->issueReadNvme2GPU(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
    LOGGING(atomicAdd(&(logger->prefetch_issue), 1));
    return gpu_cache_idx;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ unsigned int AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::prefetch(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){
    unsigned int mask = __activemask();
    unsigned int dev_mask = __match_any_sync(mask, dev_idx);
    unsigned int blk_mask = __match_any_sync(mask, ssd_blk_idx);
    unsigned int eq_mask = dev_mask & blk_mask;
    unsigned int master = __ffs(eq_mask) - 1;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int gpu_cache_idx = -1;
    if(lane_id == master){
        gpu_cache_idx = this->getGPUCacheBasePtr()->checkCacheHit_relaxed(dev_idx, ssd_blk_idx);
        if(gpu_cache_idx != -1){
            LOGGING(atomicAdd(&(logger->prefetch_relaxed_hit), 1));
        }else{
            gpu_cache_idx = this->prefetch_core(dev_idx, ssd_blk_idx);
            LOGGING(atomicAdd(&(logger->prefetch_relaxed_miss), 1));
        }
        
    }
    gpu_cache_idx = __shfl_sync(mask, gpu_cache_idx, master);
    return gpu_cache_idx;
}

__device__ void moveData(void * src, void * dst, unsigned int size){
    unsigned int * src_ptr = (unsigned int *)src;
    unsigned int * dst_ptr = (unsigned int *)dst;
    for(unsigned int i = 0; i < size / sizeof(unsigned int); ++i){
        dst_ptr[i] = src_ptr[i];
    }
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::read_cache(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int line_offset, void * dst, unsigned int size){
    read_cache_start:
    AgileLockChain chain;
    unsigned int gpu_cache_idx = -1;
    bool gpu_hit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &gpu_cache_idx, &chain);
    if(gpu_hit){

        if(this->getGPUCacheBasePtr()->checkInReadProcessing_inLockArea(gpu_cache_idx)){
            this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
            __nanosleep(1000);
            goto read_cache_start;
        }

        moveData(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx) + line_offset, dst, size);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
        return;
    }

    unsigned int cpu_cache_idx = -1;
    bool cpu_hit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &cpu_cache_idx, &chain);
    if(cpu_hit){
        this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
        this->cache_hierarchy->moveData(this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx), this->getGPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx));
        this->getGPUCacheBasePtr()->finishReading_inLockArea(gpu_cache_idx);
        moveData(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx) + line_offset, dst, size);
        this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, &chain);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
        return;
    }

    this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, &chain);
    this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);

    this->issueReadNvme2GPU(dev_idx, ssd_blk_idx, gpu_cache_idx, &chain);
    __nanosleep(1000);
    goto read_cache_start;
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::asyncWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->cacheWrite(dev_idx, ssd_blk_idx, buf_ptr, chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::asyncRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr &buf_ptr, AgileLockChain &chain){
    // this->asyncRead(dev_idx, ssd_blk_idx, &buf_ptr, &chain);
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->cacheRead(dev_idx, ssd_blk_idx, &buf_ptr, &chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::asyncWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr &buf_ptr, AgileLockChain &chain){
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->cacheWrite(dev_idx, ssd_blk_idx, &buf_ptr, &chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::writeThroughNvme(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr & buf_ptr, AgileLockChain & chain){
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->writeThroughNvme(dev_idx, ssd_blk_idx, &buf_ptr, &chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::writeThroughNvme_noRead_benchWrite(NVME_DEV_IDX_TYPE dev_idx, unsigned long idx, T val, AgileLockChain & chain){
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->template writeThroughNvme_noRead<T>(dev_idx, idx, val, &chain);
}





// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::prepareWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
//     static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->prepareWrite(dev_idx, ssd_blk_idx, buf_ptr, chain);
// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ AgileArrReadWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getArrayReadWrap(AgileBufPtr &buf, AgileLockChain &chain){
    return AgileArrReadWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(this, &buf, &chain);
}

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// template <typename T>
// __device__ AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getArrayWriteWrap(AgileBufPtr &buf, AgileLockChain &chain, bool withRead){
//     return AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(this, &buf, &chain, withRead);
// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getArraySharedWrap(AgileBufPtr &buf, AgileLockChain &chain, bool read, bool write){
    return AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(this, &buf, &chain, read, write);
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::updateWrite(AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> & array){
    array.buf->w_buf->decReference();
    if(array.buf->buf == array.buf->w_buf){
        AgileBuf * tmp_ptr;
        AgileCtrl_updateWrite_lockStart:
        static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->getShareTablePtr()->checkTableHitAcquireLock_lockStart(array.buf->buf->tag_ssd_dev, array.buf->buf->tag_ssd_blk, tmp_ptr, array.chain);
        while(atomicAdd(&(array.buf->buf->reference), 0) != 0){
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(array.buf->buf->tag_ssd_dev, array.buf->buf->tag_ssd_blk, array.chain);
            busyWait(1000);
            LOGGING(atomicAdd(&(logger->waitTooMany), 1));
            goto AgileCtrl_updateWrite_lockStart;
        }
        static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->getShareTablePtr()->removeBuf_inLockArea(array.buf->buf->tag_ssd_dev, array.buf->buf->tag_ssd_blk, array.buf->buf, array.chain);
        static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(array.buf->buf->tag_ssd_dev, array.buf->buf->tag_ssd_blk, array.chain);
        this->asyncWrite(array.buf->buf->tag_ssd_dev, array.buf->buf->tag_ssd_blk, array.buf, array.chain);
    }
    array.buf->resetStatus();
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ AgileTableBuf<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getAgileTableBuf(AgileBuf &buf, AgileLockChain & chain){
    return AgileTableBuf<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(this, &buf, chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ ShareTableBase<ShareTableImpl> * AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getTable(){
    return static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->getShareTablePtr();
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ unsigned int AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::warpCoalesceAcquireGPUCache(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int &eq_mask, unsigned int &lane_id, unsigned int &master_id){

    unsigned int mask = __activemask();
    unsigned int dev_mask = __match_any_sync(mask, dev_idx);
    unsigned int blk_mask = __match_any_sync(mask, ssd_blk_idx);
    eq_mask = dev_mask & blk_mask;
    master_id = __ffs(eq_mask) - 1;
    lane_id = threadIdx.x % 32;
    unsigned int possible_gpu_cache_idx = -1;
    unsigned int gpu_cache_idx = -1;
    if(master_id == lane_id){
        master_start:
        possible_gpu_cache_idx = this->getGPUCacheBasePtr()->getPossibleGPUCacheIdx(dev_idx, ssd_blk_idx);
        if(possible_gpu_cache_idx != -1){
            if(!this->getGPUCacheBasePtr()->checkHit(dev_idx, ssd_blk_idx, possible_gpu_cache_idx)){ // check if the gpu cache hit
                // if not hit, we need to decrease the reference count to allow eviction
                LOGGING(atomicAdd(&(logger->gpu_cache_miss), 1));
                // we need to load data to GPU cache
                gpu_cache_idx = this->prefetch_core(dev_idx, ssd_blk_idx);
                // LOGGING(atomicAdd(&(logger->runtime_issue), 1));
                this->waitCacheSlot_warpMaste(gpu_cache_idx);
                this->getGPUCacheBasePtr()->incReference(gpu_cache_idx); // inc after it is ready, this may cause repeat check, should be avoid later
                if(!this->getGPUCacheBasePtr()->checkHit(dev_idx, ssd_blk_idx, gpu_cache_idx)){
                    this->getGPUCacheBasePtr()->decReference(gpu_cache_idx);
                    goto master_start;
                    // GPU_ASSERT(false, "goto master_start;");
                }
            } else {
                LOGGING(atomicAdd(&(logger->gpu_cache_hit), __popc(eq_mask)));
                this->waitCacheSlot_warpMaste(possible_gpu_cache_idx);
                this->getGPUCacheBasePtr()->incReference(possible_gpu_cache_idx); // this will prevent gpu cache eviction
                gpu_cache_idx = possible_gpu_cache_idx;
            }
        } else {
            LOGGING(atomicAdd(&(logger->gpu_cache_miss), 1));
            gpu_cache_idx = this->prefetch_core(dev_idx, ssd_blk_idx);
            // LOGGING(atomicAdd(&(logger->runtime_issue), 1));
            this->waitCacheSlot_warpMaste(gpu_cache_idx);
            this->getGPUCacheBasePtr()->incReference(gpu_cache_idx); // inc after it is ready, this may cause repeat check, should be avoid later
            if(!this->getGPUCacheBasePtr()->checkHit(dev_idx, ssd_blk_idx, gpu_cache_idx)){
                this->getGPUCacheBasePtr()->decReference(gpu_cache_idx);
                goto master_start;
                // GPU_ASSERT(false, "goto master_start;");
            }
            LOGGING(atomicAdd(&(logger->gpu_cache_hit), __popc(eq_mask)));
        }
    }
    
    gpu_cache_idx = __shfl_sync(eq_mask, gpu_cache_idx, master_id);
    return gpu_cache_idx;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::warpCoalesceReleaseGPUCache(unsigned int gpu_cache_idx, unsigned int mask, unsigned int lane_id, unsigned int master_id){
    __syncwarp(mask);
    if(master_id == lane_id){
        this->getGPUCacheBasePtr()->decReference(gpu_cache_idx);
    }
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ T AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::readCacheElement(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int idx, AgileLockChain * chain){
    T val = -1;
    unsigned int eq_mask, lane_id, master_id;
    unsigned int gpu_cache_idx = this->warpCoalesceAcquireGPUCache(dev_idx, ssd_blk_idx, eq_mask, lane_id, master_id);
    val = this->template readCacheElement_inWarp<T>(gpu_cache_idx, idx);
    this->warpCoalesceReleaseGPUCache(gpu_cache_idx, eq_mask, lane_id, master_id);
    
    return val;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::loadArrayFromCache(NVME_DEV_IDX_TYPE dev_idx, unsigned long idx, T * data, unsigned int size){
    unsigned long ssd_blk_idx = idx * sizeof(T) / this->buf_size;
    unsigned long line_offset = idx * sizeof(T) % this->buf_size;
    GPU_ASSERT(line_offset + size * sizeof(T) <= this->buf_size, "line offset + size > buf size");
    unsigned int eq_mask, lane_id, master_id;
    unsigned int gpu_cache_idx = this->warpCoalesceAcquireGPUCache(dev_idx, ssd_blk_idx, eq_mask, lane_id, master_id);
    T* data_ptr = (T*) (this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx) + line_offset);
    for(unsigned int i = 0; i < size; ++i){
        data[i] = data_ptr[i];
    }
    this->warpCoalesceReleaseGPUCache(gpu_cache_idx, eq_mask, lane_id, master_id);
}
// TODO: no AgileChain inside for debug lock
template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::accumulateArrayFromCache(NVME_DEV_IDX_TYPE dev_idx, unsigned long idx, T * data, unsigned int size){
    unsigned long ssd_blk_idx = idx * sizeof(T) / this->buf_size;
    unsigned long line_offset = idx * sizeof(T) % this->buf_size;
    GPU_ASSERT(line_offset + size * sizeof(T) <= this->buf_size, "line offset + size > buf size");
    unsigned int eq_mask, lane_id, master_id;
    unsigned int gpu_cache_idx = this->warpCoalesceAcquireGPUCache(dev_idx, ssd_blk_idx, eq_mask, lane_id, master_id);
    T* data_ptr = (T*) (this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx) + line_offset);
    for(unsigned int i = 0; i < size; ++i){
        data[i] += data_ptr[i];
    }
    this->warpCoalesceReleaseGPUCache(gpu_cache_idx, eq_mask, lane_id, master_id);
}

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// template <typename T>
// __device__ T AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::readCacheElement(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int idx, AgileLockChain * chain){
    
    
    
//     readCacheElement_start:
//     T val = -1;
//     unsigned int element_hit = this->getGPUCacheBasePtr()->template getCacheElement<T>(dev_idx, ssd_blk_idx, idx, &val, chain);
//     if(element_hit){
//         LOGGING(atomicAdd(&(logger->gpu_cache_hit), 1));
//         return val;
//     }

//     unsigned int gpu_cache_idx = -1;
//     bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
//     if(gpuHit){
//         unsigned int stat = this->getGPUCacheBasePtr()->getStatus_inLockArea(gpu_cache_idx);
//         GPU_ASSERT(stat != AGILE_GPUCACHE_EMPTY, "GPU cache is not ready");
//         if(stat == AGILE_GPUCACHE_READ_PROCESSING){
//             this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
//             goto readCacheElement_start;
//         }
//         T * data = static_cast<T *>(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
//         this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
//         return data[idx];
//     }

//     unsigned int cpu_cache_idx = -1;
//     bool cpuHit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &cpu_cache_idx, chain);
//     if(cpuHit){
//         this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
//         this->cache_hierarchy->moveData(this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx), this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
//         this->getGPUCacheBasePtr()->finishReading_inLockArea(gpu_cache_idx);
//         T * data = static_cast<T *>(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
//         this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
//         this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
//         return data[idx];
//     }

//     this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
//     this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
//     this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
//     this->issueReadNvme2GPU(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);

//     goto readCacheElement_start;

// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ GPUCacheBase<GPUCacheImpl> * AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getGPUCacheBasePtr(){
    return static_cast<GPUCacheBase<GPUCacheImpl> * >(this->cache_hierarchy->gpu_cache);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ CPUCacheBase<CPUCacheImpl> * AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getCPUCacheBasePtr(){
    return static_cast<CPUCacheBase<CPUCacheImpl> * >(this->cache_hierarchy->cpu_cache);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template <typename T>
__device__ AgileArrayWarp<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::getArrayWrap(AgileLockChain & chain){
    return AgileArrayWarp<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(this, chain);
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::checkCacheSlot_warpMasteAcquire(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *gpu_cache_idx, AgileLockChain * chain){
    bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    if(gpuHit){
        this->getGPUCacheBasePtr()->incReference_inLockArea(*gpu_cache_idx);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, *gpu_cache_idx, chain);
        return;
    }

    unsigned int cpu_cache_idx = -1;
    bool cpuHit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &cpu_cache_idx, chain);
    if(cpuHit){
        this->getGPUCacheBasePtr()->processingReading_inLockArea(*gpu_cache_idx);
        this->cache_hierarchy->moveData(this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx), this->getGPUCacheBasePtr()->getCacheDataPtr(*gpu_cache_idx));
        this->getGPUCacheBasePtr()->finishReading_inLockArea(*gpu_cache_idx);
        this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
        this->getGPUCacheBasePtr()->incReference_inLockArea(*gpu_cache_idx);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, *gpu_cache_idx, chain);
        return;
    }

    this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
    this->getGPUCacheBasePtr()->processingReading_inLockArea(*gpu_cache_idx);
    this->getGPUCacheBasePtr()->incReference_inLockArea(*gpu_cache_idx);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, *gpu_cache_idx, chain);
    this->issueReadNvme2GPU(dev_idx, ssd_blk_idx, *gpu_cache_idx, chain);

}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::checkCacheSlot_warpMasteRelease(unsigned int gpu_cache_idx){
    this->getGPUCacheBasePtr()->decReference(gpu_cache_idx);
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::waitCacheSlot_warpMaste(unsigned int gpu_cache_idx){
    
    waitCacheSlot_warpMaste_start:
    unsigned int stat = this->getGPUCacheBasePtr()->getStatus_inLockArea(gpu_cache_idx);
    if(!(stat == AGILE_GPUCACHE_READ_PROCESSING || stat >= AGILE_GPUCACHE_READY || stat == AGILE_GPUCACHE_EMPTY)){
        printf("GPU cache stat error: %d\n", stat);
    }
    GPU_ASSERT(stat == AGILE_GPUCACHE_READ_PROCESSING || stat >= AGILE_GPUCACHE_READY || stat == AGILE_GPUCACHE_EMPTY, "GPU cache stat error");
    if(stat == AGILE_GPUCACHE_READ_PROCESSING || stat == AGILE_GPUCACHE_EMPTY){
        
        goto waitCacheSlot_warpMaste_start;
    }
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template<typename T>
__device__ T AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::readCacheElement_inWarp(unsigned int gpu_cache_idx, unsigned int idx){
    T * data = static_cast<T *>(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
    return data[idx];
}


/********* array wrap **********/

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrReadWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::AgileArrReadWrapIdx(NVME_DEV_IDX_TYPE dev_id, AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain){
    this->dev_id = dev_id;
    this->buf = buf;
    this->ctrl = ctrl;
    this->chain = chain;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ const T AgileArrReadWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::operator[](unsigned long idx) const {
    SSDBLK_TYPE blk_idx = idx / (ctrl->buf_size / sizeof(T));
    // printf("here\n");
    if(buf->readHit(dev_id, blk_idx)){ // hit local
        LOGGING(atomicAdd(&(logger->buffer_localhit), 1));
        return static_cast<T*>(this->buf->buf->data)[idx % (ctrl->buf_size / sizeof(T))];
    }else{
        // printf("read miss, dev_id: %d, blk_idx: %d\n", dev_id, blk_idx);
        this->buf->resetStatus();
        this->buf->setSelfTag(dev_id, blk_idx);
        ctrl->asyncRead(dev_id, blk_idx, this->buf, chain);
        this->buf->buf->wait();
        return static_cast<T*>(this->buf->buf->data)[idx % (ctrl->buf_size / sizeof(T))];
    }
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrReadWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::AgileArrReadWrap(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain){
    this->ctrl = ctrl;
    this->buf = buf;
    this->chain = chain;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrReadWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileArrReadWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::operator[](NVME_DEV_IDX_TYPE dev_id){
    return AgileArrReadWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(dev_id, ctrl, buf, chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrSharedWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::AgileArrSharedWrapIdx(NVME_DEV_IDX_TYPE dev_id, AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain, unsigned int read, unsigned int write){
    this->dev_id = dev_id;
    this->buf = buf;
    this->ctrl = ctrl;
    this->chain = chain;
    this->read = read;
    this->write = write;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ T& AgileArrSharedWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::operator[](unsigned long idx) {
    SSDBLK_TYPE blk_idx = idx / (ctrl->buf_size / sizeof(T));
    
    if(buf->shared_buf_ptr != nullptr){ // local shared buffer pointer to 
        if(buf->sharedBufHit(dev_id, blk_idx)){ // hit
            // GPU_ASSERT(buf->shared_buf_ptr != nullptr, "buf->shared_buf_ptr != nullptr");
            LOGGING(atomicAdd(&(logger->buffer_localhit), 1));
            return static_cast<T*>(buf->shared_buf_ptr->getDataPtr())[idx % (ctrl->buf_size / sizeof(T))];
        }else if(buf->buf == buf->shared_buf_ptr){ // if this thread holds the shared buffer, it needs to wait all reference threads finish their jobs
            // release the shared buffer
            // GPU_ASSERT(this->buf->shared_buf_ptr != nullptr, "shared buffer should not be null");
            buf->shared_buf_ptr->decReference();
            unsigned int count = 0;
            // GPU_ASSERT(atomicAdd(&(this->buf->buf->in_table), 0) == 1, "should be in table");
            AgileArrSharedWrapIdx_wait_reference:
            // GPU_ASSERT(buf->buf != nullptr, "buf->buf != nullptr");
            // GPU_ASSERT(buf->shared_buf_ptr != nullptr, "buf->buf != nullptr");
            // GPU_ASSERT(buf->buf == buf->shared_buf_ptr, "should hit");

            while(atomicAdd(&(buf->shared_buf_ptr->reference), 0) != 0){
                count++;
                if(count > 100000){
                    LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                    count = 0;
                }
                // static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, chain);
                goto AgileArrSharedWrapIdx_wait_reference;
            }

            AgileBuf * tmp_ptr = nullptr; 
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->checkTableHitAcquireLock_lockStart(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, &tmp_ptr, chain);
            if(atomicAdd(&(buf->shared_buf_ptr->reference), 0) != 0){
                static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, chain);
                goto AgileArrSharedWrapIdx_wait_reference;
            }
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->removeBuf_inLockArea(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, this->buf->buf, chain, __LINE__);
            __threadfence_system();
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, chain);

            buf->shared_buf_ptr->share_table_next = nullptr;
            buf->shared_buf_ptr->setTag(-1, -1);
            buf->shared_buf_ptr = nullptr;
            buf->self = 0;
            
            buf->buf->resetStatus();
            
        } else { // miss and this thread does not hold the shared buffer
            // release others' shared buffer
            GPU_ASSERT(buf->buf != buf->shared_buf_ptr, "should not be the same");
            buf->shared_buf_ptr->decReference(); // decrease the reference count
            buf->shared_buf_ptr = nullptr;
        }
    }
    // get new shared buffer
    buf->resetStatus();
    buf->shared_buf_ptr = nullptr;
    static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(ctrl->cache_hierarchy)->checkShareTable(dev_id, blk_idx, buf, read, chain);
    GPU_ASSERT(this->buf->shared_buf_ptr != nullptr, "shared buffer should not be null");
    this->buf->shared_buf_ptr->wait();
    if(write){
        this->buf->shared_buf_ptr->setModified();
    }
    // printf("%s:%d dev: %d blk: %d %p %p [%d]: %d\n", __FILE__, __LINE__, dev_id, blk_idx, buf->shared_buf_ptr, buf->shared_buf_ptr->getDataPtr(), idx, static_cast<T*>(buf->shared_buf_ptr->getDataPtr())[idx % (ctrl->buf_size / sizeof(T))]);
    return static_cast<T*>(buf->shared_buf_ptr->getDataPtr())[idx % (ctrl->buf_size / sizeof(T))];
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ void AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::reset(){
    if(this->buf->shared_buf_ptr != nullptr){
        this->buf->shared_buf_ptr->decReference();
        if(this->buf->buf == this->buf->shared_buf_ptr){
            unsigned int count = 0;
            AgileBuf * tmp_ptr = nullptr;
            AgileArrSharedWrap_lockStart:
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->checkTableHitAcquireLock_lockStart(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, &tmp_ptr, chain);
            if(tmp_ptr == nullptr){
                this->buf->self = 0;
                buf->shared_buf_ptr->setTag(-1, -1);
                buf->shared_buf_ptr->share_table_next = nullptr;
                return;
                printf("buf->shared_buf_ptr->tag_ssd_dev : %d, buf->shared_buf_ptr->tag_ssd_blk : %d\n", buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk);
                GPU_ASSERT(false, "tmp_ptr should not be null");
            }
            // if(tmp_ptr == nullptr && atomicAdd(&(buf->shared_buf_ptr->reference), 0) == 0){
            //     this->buf->self = 0;
            //     static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, chain);
            // }
            while(atomicAdd(&(buf->shared_buf_ptr->reference), 0) != 0){
                count++;
                if(count > 100000){
                    LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                    count = 0;
                }
                static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, chain);
                goto AgileArrSharedWrap_lockStart;
            }
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->removeBuf_inLockArea(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, this->buf->buf, chain, __LINE__);
            static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->ctrl->cache_hierarchy)->getShareTablePtr()->releaseTableLock_lockEnd(buf->shared_buf_ptr->tag_ssd_dev, buf->shared_buf_ptr->tag_ssd_blk, chain);
            // TODO: write
            this->buf->self = 0;
            buf->shared_buf_ptr->setTag(-1, -1);
            buf->shared_buf_ptr->share_table_next = nullptr;
        } 

        // this->buf->buf->table_next = nullptr;
        // this->buf->shared_buf_ptr->table_next = nullptr;
        // this->buf->buf->cache_next = nullptr;
        // this->buf->shared_buf_ptr->cache_next = nullptr;
        this->buf->shared_buf_ptr = nullptr;
        this->buf->resetStatus();
    }
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::~AgileArrSharedWrap(){
    this->buf->resetStatus();
    this->buf->shared_buf_ptr = nullptr;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::AgileArrSharedWrap(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain, unsigned int read, unsigned int write){
    this->ctrl = ctrl;
    this->buf = buf;
    this->chain = chain;
    this->read = read;
    this->write = write;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ AgileArrSharedWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::operator[](NVME_DEV_IDX_TYPE dev_id){
    return AgileArrSharedWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>(dev_id, ctrl, buf, chain, read, write);
}

