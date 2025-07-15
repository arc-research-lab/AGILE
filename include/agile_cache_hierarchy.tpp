#include "agile_ctrl.h"
#include "agile_swcache.h"
#include "agile_cache_hierarchy.h"
#include "agile_share_table.h"
/********* tool functions **********/

__device__ void AgileCacheHierarchyBase::moveData(void * src_ptr, void * dst_ptr){
    uint4 * src = reinterpret_cast<uint4*>(src_ptr);
    uint4 * dst = reinterpret_cast<uint4*>(dst_ptr);
    for(unsigned int i = 0; i < this->ctrl->buf_size / sizeof(uint4); ++i){
        dst[i] =  src[i];
    }
}

/********* GPU cache **********/

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::evictGPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, bool* inProcessing, AgileLockChain * chain){
    evictGPUCache_inLockArea_start:
    unsigned int stat = atomicAdd(&(this->getGPUCacheBasePtr()->cache_status[gpu_cache_idx]), 0);
    if(stat == AGILE_GPUCACHE_READ_PROCESSING || stat == AGILE_GPUCACHE_WRITE_PROCESSING){
        // *inProcessing = false;
        return false;
    } else if(stat == AGILE_GPUCACHE_EMPTY){
        // printf("evict empty %d\n", gpu_cache_idx);
        return true;
    } 
    
    // check reference;
    unsigned int ref = atomicAdd(&(this->getGPUCacheBasePtr()->cache_ref[gpu_cache_idx]), 0);
    if(ref > 0){
        return false;
    }

    bool succ = false;
    if(stat == AGILE_GPUCACHE_READY){ 
        LOGGING(atomicAdd(&(logger->gpu_cache_evict), 1));
        unsigned int old_stat = atomicCAS(&(this->getGPUCacheBasePtr()->cache_status[gpu_cache_idx]), AGILE_GPUCACHE_READY, AGILE_GPUCACHE_EMPTY);
        if(old_stat != AGILE_GPUCACHE_READY){
            GPU_ASSERT(false, "evictGPUCache_inLockArea error");
            // LOGGING(atomicAdd(&(logger->deadlock_check), 1));
            // goto evictGPUCache_inLockArea_start; // wait for (AGILE_GPUCACHE_READY + reference goes) down to AGILE_GPUCACHE_READY
        }
        succ = true;
        this->evictAttemptGPU2CPU_inLockArea(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
        this->getGPUCacheBasePtr()->resetSlot_inLockArea(gpu_cache_idx);
    } else if (stat == AGILE_GPUCACHE_MODIFIED){ // check modified
        unsigned int old_stat = atomicCAS(&(this->getGPUCacheBasePtr()->cache_status[gpu_cache_idx]), AGILE_GPUCACHE_MODIFIED, AGILE_GPUCACHE_WRITE_PROCESSING);
        if(old_stat != AGILE_GPUCACHE_MODIFIED){
            GPU_ASSERT(false, "evictGPUCache_inLockArea error");
            // LOGGING(atomicAdd(&(logger->deadlock_check), 1));
            // goto evictGPUCache_inLockArea_start; // wait for (AGILE_GPUCACHE_READY + reference goes) down to AGILE_GPUCACHE_READY
        }
        if(this->evictAttemptGPU2CPU_inLockArea(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain)){
            this->getGPUCacheBasePtr()->resetSlot_inLockArea(gpu_cache_idx);
        } else {
            LOGGING(atomicAdd(&(logger->gpu2nvme), 1));
            this->evictGPU2Nvme_inLockArea(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
            // this->getGPUCacheBasePtr()->processingWriting_inLockArea(gpu_cache_idx);
        }
    } else {
        // GPU_ASSERT(false, "status error");
        LOGGING(atomicAdd(&(logger->deadlock_check), 1));
        goto evictGPUCache_inLockArea_start; // wait for (AGILE_GPUCACHE_READY + reference goes) down to AGILE_GPUCACHE_READY
    }
    
    return succ;
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::evictGPU2Nvme_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, AgileLockChain * chain){
    // issue write
    // GPU_ASSERT(false, "evictGPU2Nvme_inLockArea not implemented");
    this->ctrl->issueWriteGPU2Nvme(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    this->getGPUCacheBasePtr()->processingWriting_inLockArea(gpu_cache_idx);
}

// called after cache hit of cpu cache, move data to gpu again, checking if write-through happens or other write TODO
template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::refillBuf2GPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, AgileBufPtr * buf_ptr){
    this->moveData(buf_ptr->getDataPtr(), this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
    this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->finishReading_inLockArea(gpu_cache_idx);
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::fillOrAppendAgileBuf2GPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileBufPtr * buf_ptr){
    if (this->getGPUCacheBasePtr()->readReadyCheck_inLockArea(cache_idx)) { // the data is already in gpu cache
        buf_ptr->buf->setProcessingRead();
        this->moveData(this->getGPUCacheBasePtr()->getCacheDataPtr(cache_idx), buf_ptr->getDataPtr());
        buf_ptr->ready();
    } else {
        this->getGPUCacheBasePtr()->appendAgileBuf_inLockArea(cache_idx, buf_ptr->buf);
    }
}


/********* CPU cache **********/

__device__ bool AgileCacheHierarchyBase::evictCPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, bool *inProcessing, AgileLockChain * chain){
    // GPU_ASSERT(false, "evictCPUCache_inLockArea not implemented.");
    bool succ = false;
    if(this->cpu_cache->checkEmpty_inLockArea(cache_idx)){
        succ = true;
        this->cpu_cache->resetSlot_inLockArea(cache_idx);
    } else if(this->cpu_cache->checkErasable_inLockArea(cache_idx)){
        succ = true;
        this->cpu_cache->resetSlot_inLockArea(cache_idx);
        LOGGING(atomicAdd(&(logger->cpu_cache_evict), 1));
    } else {
        succ = false;
        GPU_ASSERT(false, "evictCPUCache_inLockArea not implemented.");
        LOGGING(atomicAdd(&(logger->cpu_cache_evict), 1));
        this->ctrl->issueWriteCPU2Nvme(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
    }
    
    return succ; // TODO fix it
}

/********* GPU-CPU cache interaction **********/

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::cacheRead(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    
    // AgileBuf * table_buf = this->getShareTablePtr()->checkTableHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, chain);
    // // printf("%s:%d\n", __FILE__, __LINE__);
    // if(table_buf != nullptr){
    //     this->moveData(table_buf->data, buf_ptr->buf);
    //     this->getShareTablePtr()->releaseTableLock_lockEnd(ssd_dev_idx, ssd_blk_idx, chain);
    //     buf_ptr->buf->ready();
    //     return;
    // }
    
    unsigned int gpu_cache_idx = -1;
    bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
    if(gpuHit){
        this->fillOrAppendAgileBuf2GPUCache_inLockArea(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
        // this->getShareTablePtr()->releaseTableLock_lockEnd(ssd_dev_idx, ssd_blk_idx, chain);
        LOGGING(atomicAdd(&(logger->gpu_cache_hit), 1));
        return;
    }
    
    LOGGING(atomicAdd(&(logger->gpu_cache_miss), 1));
    unsigned int cpu_cache_idx = -1;
    bool cpuHit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &cpu_cache_idx, chain);
    if(cpuHit){
        this->moveData(this->cpu_cache->getCacheDataPtr(cpu_cache_idx), buf_ptr->buf->data);
        this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
        this->refillBuf2GPUCache_inLockArea(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
        buf_ptr->buf->ready();
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
        // this->getShareTablePtr()->releaseTableLock_lockEnd(ssd_dev_idx, ssd_blk_idx, chain);
        LOGGING(atomicAdd(&(logger->cpu_cache_hit), 1));
        LOGGING(atomicAdd(&(logger->cpu2gpu), 1));
        return;
    }
    
    LOGGING(atomicAdd(&(logger->cpu_cache_miss), 1));
    // if all fail issue read from nvme
    this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
    this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx); // mark status as read_processing so later checks knows
    this->getGPUCacheBasePtr()->appendAgileBuf_inLockArea(gpu_cache_idx, buf_ptr->buf);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    // this->getShareTablePtr()->releaseTableLock_lockEnd(ssd_dev_idx, ssd_blk_idx, chain);
    this->ctrl->issueReadNvme2GPU(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::cacheWrite(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    unsigned int gpu_cache_idx = -1;
    bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
    this->moveData(buf_ptr->buf->data, this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
    this->getGPUCacheBasePtr()->setModified_inLoadArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::writeThroughNvme(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    unsigned int gpu_cache_idx = -1;
    bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
    this->moveData(buf_ptr->buf->data, this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
    __threadfence_system();
    this->getGPUCacheBasePtr()->setModified_inLoadArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->processingWriting_inLockArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    this->ctrl->issueWriteGPU2Nvme(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
template<typename T>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::writeThroughNvme_noRead(NVME_DEV_IDX_TYPE ssd_dev_idx, unsigned long idx, T val, AgileLockChain * chain){
    unsigned long ssd_blk_idx = idx / this->ctrl->buf_size;
    unsigned long line_offset = idx % this->ctrl->buf_size;
    unsigned int gpu_cache_idx = -1;
    bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
    if(gpuHit){
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
        return;
    }
    static_cast<T *>(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx))[line_offset] = val;
    __threadfence_system();
    this->getGPUCacheBasePtr()->setModified_inLoadArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->processingWriting_inLockArea(gpu_cache_idx);
    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    this->ctrl->issueWriteGPU2Nvme(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
}


// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::prepareShared_withoutRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    // AgileBuf * table_buf;
    // // printf("%s:%d\n", __FILE__, __LINE__);
    // bool tableHit = this->getShareTablePtr()->checkTableHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, table_buf, chain);
    // if(tableHit){
    //     // printf("%s:%d\n", __FILE__, __LINE__);
    //     buf_ptr->setSharedPtr(table_buf);
    // }else{
    //     // printf("%s:%d\n", __FILE__, __LINE__);
    //     buf_ptr->setSharedPtr(buf_ptr->buf);
    //     buf_ptr->shared_buf->setTag(dev_idx, ssd_blk_idx);
    //     buf_ptr->shared_buf->setModified();
    //     this->getShareTablePtr()->appendBuf_inLockArea(dev_idx, ssd_blk_idx, buf_ptr->buf, chain);
    // }
    // // printf("%s:%d\n", __FILE__, __LINE__);
    // atomicAdd(&(buf_ptr->w_buf->reference), 1);
    // this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
// }

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::prepareShared_withRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
    // // printf("%s:%d\n", __FILE__, __LINE__);
    // AgileBuf * table_buf;
    // bool writeTableHit = this->getShareTablePtr()->checkTableHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, table_buf, chain);
    // if(writeTableHit){
    //     GPU_ASSERT(table_buf != nullptr, "table_buf is nullptr");
    //     atomicAdd(&(table_buf->reference), 1);
    //     buf_ptr->setWritePtr(table_buf);
    //     this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
    //     return;
    // }

    // buf_ptr->setWritePtr(buf_ptr->buf);
    // buf_ptr->w_buf->setTag(dev_idx, ssd_blk_idx);
    // this->getShareTablePtr()->appendBuf_inLockArea(dev_idx, ssd_blk_idx, buf_ptr->buf, chain);
    // atomicAdd(&(buf_ptr->buf->reference), 1);

    // unsigned int gpu_cache_idx = -1;
    // bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
    // if(gpuHit){
    //     this->fillOrAppendAgileBuf2GPUCache_inLockArea(dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
        
    //     this->getShareTablePtr()->appendBuf_inLockArea(dev_idx, ssd_blk_idx, buf_ptr->buf, chain);
    //     this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    //     this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
    //     return;
    // }

    // unsigned int cpu_cache_idx = -1;
    // bool cpuHit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &cpu_cache_idx, chain);
    // if(cpuHit){
    //     this->moveData(this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx), buf_ptr->buf->data);
    //     this->refillBuf2GPUCache_inLockArea(dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
    //     this->getShareTablePtr()->appendBuf_inLockArea(dev_idx, ssd_blk_idx, buf_ptr->buf, chain);
    //     this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
    //     this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    //     this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
    //     return;
    // }

    // this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
    // this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx); // mark status as read_processing so later checks knows
    // this->getGPUCacheBasePtr()->appendAgileBuf_inLockArea(gpu_cache_idx, buf_ptr->buf);
    // this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    // this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
    // this->ctrl->issueReadNvme2GPU(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);

    // issue read
// }
// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::checkTableWithoutRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain){
//     AgileBuf * table_buf;
//     bool writeTableHit = this->getShareTablePtr()->checkTableHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, table_buf, chain);
//     if(writeTableHit){
//         GPU_ASSERT(table_buf != nullptr, "table_buf is nullptr");
//         buf_ptr->setSharedPtr(table_buf);
//         atomicAdd(&(buf_ptr->shared_buf->reference), 1);
//         this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
//         return;
//     }

//     // if not hit, then this thread should hold the shared buffer
//     buf_ptr->setSharedPtr(buf_ptr->buf);
//     atomicAdd(&(buf_ptr->shared_buf->reference), 1);
//     buf_ptr->shared_buf->setTag(dev_idx, ssd_blk_idx);
//     this->getShareTablePtr()->appendBuf_inLockArea(dev_idx, ssd_blk_idx, buf_ptr->buf, chain);

//     // start to check GPU cache
//     unsigned int gpu_cache_idx = -1;
//     bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
//     if(gpuHit){
//         this->fillOrAppendAgileBuf2GPUCache_inLockArea(dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
//         this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
//         this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
//         return;
//     }

// }

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::checkShareTable(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr *& buf_ptr, unsigned int read, AgileLockChain * chain){
    AgileBuf * table_buf = nullptr;
    // 
    this->getShareTablePtr()->checkTableHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &table_buf, chain); 
    buf_ptr->shared_buf_ptr =  table_buf;
    if(buf_ptr->shared_buf_ptr != nullptr){
        atomicAdd(&(buf_ptr->shared_buf_ptr->reference), 1);
        this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
        return;
    }
    buf_ptr->self = 1;
    // if not hit, then this thread should hold the shared buffer
    buf_ptr->setSharedBufPtr(buf_ptr->buf);
    atomicAdd(&(buf_ptr->shared_buf_ptr->reference), 1);
    buf_ptr->shared_buf_ptr->setTag(dev_idx, ssd_blk_idx);
    this->getShareTablePtr()->appendBuf_inLockArea(dev_idx, ssd_blk_idx, buf_ptr->buf, chain);
    
    // start to check GPU cache
    unsigned int gpu_cache_idx = -1;
    bool gpuHit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
    if(gpuHit){
        this->fillOrAppendAgileBuf2GPUCache_inLockArea(dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
        this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
        return;
    }

    // start to check CPU cache
    unsigned int cpu_cache_idx = -1;
    bool cpuHit = this->getCPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(dev_idx, ssd_blk_idx, &cpu_cache_idx, chain);
    if(cpuHit){
        this->moveData(this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx), buf_ptr->buf->data);
        this->refillBuf2GPUCache_inLockArea(dev_idx, ssd_blk_idx, gpu_cache_idx, buf_ptr);
        buf_ptr->buf->ready();
        this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
        this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
        this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
        return;
    }

    this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
    
    if(read){
        this->getGPUCacheBasePtr()->processingReading_inLockArea(gpu_cache_idx); // mark status as read_processing so later checks knows
        this->getGPUCacheBasePtr()->appendAgileBuf_inLockArea(gpu_cache_idx, buf_ptr->buf);
    }

    this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    this->getShareTablePtr()->releaseTableLock_lockEnd(dev_idx, ssd_blk_idx, chain);
    if(read){
        this->ctrl->issueReadNvme2GPU(dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
    }
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::evictAttemptGPU2CPU_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, AgileLockChain * chain){
    
    unsigned int cpu_cache_idx = -1;
    bool hit = false;
    if(this->getCPUCacheBasePtr()->checkCacheHitAttemptAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &cpu_cache_idx, &hit, chain)){
            this->getCPUCacheBasePtr()->processingWriting_inLockArea(cpu_cache_idx);
            LOGGING(atomicAdd(&(logger->gpu2cpu), 1));
            this->moveData(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx), this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx)); // move data from GPU to CPU
            unsigned int cpu_status = (this->getGPUCacheBasePtr()->isModified_inLockArea(gpu_cache_idx)) ? AGILE_CPUCACHE_MODIFIED : AGILE_CPUCACHE_READY;
            __threadfence_system();
            this->getCPUCacheBasePtr()->finishWriting_inLockArea(cpu_cache_idx, cpu_status);
        this->getCPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, cpu_cache_idx, chain);
        return true;
    }

    return false;
}


/********* GPU-CPU cache interaction write policy **********/
/* This part should be called in CacheBase evict */

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ bool AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::cacheWriteThroughAttempt2CPU(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain){
//     unsigned int gpu_cache_idx = -1, cpu_cache_idx = -1;
//     GPU_ASSERT(false, "cacheWriteThroughAttempt2CPU write to non-writable gpu cache slot (W-W race)");
//     bool succ = false;
//     bool gpu_cache_hit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
//     if(this->getGPUCacheBasePtr()->writableCheck_inLockArea(gpu_cache_idx)){
//         this->moveData(buf_ptr->getDataPtr(), this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
//         this->getGPUCacheBasePtr()->finishWriting_inLockArea(cpu_cache_idx);
//     } else {
//         GPU_ASSERT(false, "cacheWriteThroughAttempt2CPU write to non-writable gpu cache slot (W-W race)");
//     }
    
//     if(this->getCPUCacheBasePtr()->checkCacheHitAttemptAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &cpu_cache_idx, chain)){
//         succ = true;
//         // this->moveDataGPU2CPU_inLockArea(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, cpu_cache_idx);
//         this->moveData(this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx), this->getCPUCacheBasePtr()->getCacheDataPtr(cpu_cache_idx));
//         unsigned int cpu_status = (this->getGPUCacheBasePtr()->isModified_inLockArea(gpu_cache_idx)) ? AGILE_CPUCACHE_MODIFIED : AGILE_CPUCACHE_READY;
//         this->getCPUCacheBasePtr()->finishWriting_inLockArea(cpu_cache_idx, cpu_status);
//         this->getCPUCacheBasePtr()->releaseLock_lockEnd(cpu_cache_idx, chain);
//     }

//     this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);

//     return succ;
// }



// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::cacheWriteThrough2Nvme(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain){
//     GPU_ASSERT(false, "cacheWriteThrough2Nvme write to non-writable gpu cache slot (W-W race)");
//     unsigned int gpu_cache_idx = -1;
//     bool gpu_cache_hit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
//     if(this->getGPUCacheBasePtr()->writableCheck_inLockArea(gpu_cache_idx)){
//         this->moveData(buf_ptr->getDataPtr(), this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
//         this->getGPUCacheBasePtr()->processingWriting_inLockArea(gpu_cache_idx);
//     } else {
//         GPU_ASSERT(false, "cacheWriteThrough2Nvme write to non-writable gpu cache slot (W-W race)");
//     }
//     this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);

//     this->ctrl->issueWriteGPU2Nvme(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);

// }

// template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
// __device__ void AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>::cacheWriteLateEvict(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain){
//     GPU_ASSERT(false, "cacheWriteLateEvict write to non-writable gpu cache slot (W-W race)");
//     unsigned int gpu_cache_idx = -1;
//     bool gpu_cache_hit = this->getGPUCacheBasePtr()->checkCacheHitAcquireLock_lockStart(ssd_dev_idx, ssd_blk_idx, &gpu_cache_idx, chain);
//     if(this->getGPUCacheBasePtr()->writableCheck_inLockArea(gpu_cache_idx)){
//         this->moveData(buf_ptr->getDataPtr(), this->getGPUCacheBasePtr()->getCacheDataPtr(gpu_cache_idx));
//         this->getGPUCacheBasePtr()->finishModifying_inLockArea(gpu_cache_idx);
//     } else {
//         GPU_ASSERT(false, "cacheWriteLateEvict write to non-writable gpu cache slot (W-W race)");
//     }
//     this->getGPUCacheBasePtr()->releaseSlotLock_lockEnd(ssd_dev_idx, ssd_blk_idx, gpu_cache_idx, chain);
// }

