#include "agile_ctrl.h"
#include "agile_swcache.h"

/**********************************/

/***** START AgileCacheBase ******/

__host__ AgileCacheBase::AgileCacheBase(unsigned int slot_num, unsigned int slot_size, char type[5]){

    this->slot_num = slot_num;
    this->slot_size = slot_size;
    this->cache_hierarchy = nullptr;

    cuda_err_chk(cudaMalloc(&(this->locks), sizeof(AgileLock) * slot_num));
    AgileLock * h_locks = (AgileLock *) malloc(sizeof(AgileLock) * slot_num);

    for(unsigned int i = 0; i < slot_num; ++i){
        
#if LOCK_DEBUG
        char lockName[20];
        sprintf(lockName, "%s-cache-l%d", type, i);
        h_locks[i] = AgileLock(lockName);
#else
        h_locks[i] = AgileLock();
#endif
    }

    cuda_err_chk(cudaMemcpy(this->locks, h_locks, sizeof(AgileLock) * slot_num, cudaMemcpyHostToDevice));
    free(h_locks);
}

__host__ AgileCacheBase::~AgileCacheBase(){
    // cuda_err_chk(cudaFree(this->locks));
}

__host__ void AgileCacheBase::setCacheHierarchy(AgileCacheHierarchyBase * cache_hierarchy){
    this->cache_hierarchy = cache_hierarchy;
}

__device__ void AgileCacheBase::acquireBaseLock_lockStart(unsigned int cache_idx, AgileLockChain * chain){
    this->locks[cache_idx].acquire(chain);
}

__device__ bool AgileCacheBase::acquireBaseLockAttempt_lockStart(unsigned int cache_idx, AgileLockChain * chain){
    return this->locks[cache_idx].try_acquire(chain);
}

__device__ void AgileCacheBase::releaseBaseLock_lockEnd(unsigned int cache_idx, AgileLockChain * chain){
    this->locks[cache_idx].release(chain);
}


/***** END AgileCacheBase ******/

/**********************************/

/***** START GPUCacheBase_T ******/

__host__ GPUCacheBase_T::GPUCacheBase_T(unsigned int slot_num, unsigned int slot_size) : AgileCacheBase(slot_num, slot_size, "gpu") {
    cuda_err_chk(cudaMalloc(&(this->cache_status), sizeof(unsigned int) * slot_num));
    cuda_err_chk(cudaMalloc(&(this->bufChain), sizeof(AgileBuf *) * slot_num));
    cuda_err_chk(cudaMalloc(&(this->cache_ref), sizeof(unsigned int) * slot_num));
}

__host__ GPUCacheBase_T::~GPUCacheBase_T(){
    // cuda_err_chk(cudaFree(this->cache_status));
    // cuda_err_chk(cudaFree(this->bufChain));
}

__host__ void GPUCacheBase_T::setPinedMem(void * data, unsigned long physical_addr){
    this->data = data;
    this->physical_addr = physical_addr;
}

__host__ unsigned int GPUCacheBase_T::getRequiredMemSize(){
    return this->slot_num * this->slot_size;
}

// __device__ bool GPUCacheBase_T::invokeHierarchyEvictAttemptGPU2CPU_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int tid, AgileLockChain * chain){
//     return this->getCacheHierarchyPtr()->evictAttemptGPU2CPU_inLockArea(ssd_dev_idx, ssd_blk_idx, tid, chain);
// }

__device__ void * GPUCacheBase_T::getCacheDataPtr(unsigned int cache_idx){
    return data + cache_idx * cache_hierarchy->ctrl->buf_size;
}

__device__ void GPUCacheBase_T::appendAgileBuf_inLockArea(unsigned int cache_idx, AgileBuf * buf){
    // printf("appendAgileBuf_inLockArea %p\n", buf);

    buf->setProcessingRead();
    // buf->cache_next = nullptr;
    atomicExchangePtr((void**)&(buf->cache_next), (void*)nullptr);
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat == AGILE_GPUCACHE_READ_PROCESSING, "append error");
    GPU_ASSERT(buf != nullptr, "append error");
    GPU_ASSERT(buf != buf->cache_next, "append error");
    GPU_ASSERT(buf != bufChain[cache_idx], "double append");

    // if(this->bufChain[cache_idx] == nullptr){
    if(atomicLoadPtr((void**)&(bufChain[cache_idx])) == nullptr){
        // this->bufChain[cache_idx] = buf;
        atomicExchangePtr((void**)&(this->bufChain[cache_idx]), (void*)buf);
        atomicExchangePtr((void**)&(buf->cache_next), (void*)nullptr);
        // buf->cache_next = nullptr;
    } else {
        GPU_ASSERT(bufChain[cache_idx]->cache_next != bufChain[cache_idx], "double append");
        AgileBuf * ptr = (AgileBuf *) atomicLoadPtr((void**)&(bufChain[cache_idx]));
        unsigned int counter = 0;
        GPU_ASSERT(ptr != buf, "double append");
        // while(ptr->cache_next!= nullptr){
        while(atomicLoadPtr((void**)&(ptr->cache_next)) != nullptr){
            counter++;
            
            GPU_ASSERT(ptr != ptr->cache_next, "double append");
            
            ptr = (AgileBuf *) atomicLoadPtr((void**)&(ptr->cache_next)); // ptr->cache_next;

            if(ptr == buf){
                printf("double append %d %d \n", cache_idx, counter);
            }
            GPU_ASSERT(ptr != buf, "double append");
            
            if(counter == 1000){
                counter = 0;
                LOGGING(atomicAdd(&(logger->deadlock_check), 1));
                // printf("warning buffer chain too long in appendAgileBuf_inLockArea\n");
            }
            GPU_ASSERT(bufChain[cache_idx] != ptr, "appendAgileBuf_inLockArea circle");
        }
        // ptr->cache_next = buf;
        atomicExchangePtr((void**)&(ptr->cache_next), (void*)buf);
    }
    LOGGING(atomicAdd(&(logger->appendbuf_count), 1));
}

__device__ void GPUCacheBase_T::propagateAgileBuf_inLockArea(unsigned int cache_idx){
    // AgileBuf * ptr = bufChain[cache_idx];
    // unsigned int counter = 0;
    // propagateAgileBuf_inLockArea_start:
    // if(ptr != nullptr){
    //     counter++;
    //     this->cache_hierarchy->moveData(this->getCacheDataPtr(cache_idx), ptr->data);
    //     AgileBuf * tmp = ptr->cache_next;
    //     ptr->ready();
    //     if(tmp != nullptr){
    //         if(atomicCAS(&(ptr->status), AGILE_BUF_READY, AGILE_BUF_PROPAGATED) == AGILE_BUF_READY){
    //             ptr = tmp;
    //             LOGGING(atomicAdd(&(logger->deadlock_check), 1));
    //             goto propagateAgileBuf_inLockArea_start;
    //         }else{
    //             ptr == nullptr; // let other threads to propagate the raset of the chain
    //         }
    //     }
    // }

    // LOGGING(atomicAdd(&(logger->propogate_time), 1));
    // LOGGING(atomicAdd(&(logger->propogate_count), counter));

    AgileBuf * ptr = (AgileBuf *) atomicLoadPtr((void**)&(bufChain[cache_idx])); //bufChain[cache_idx];
    AgileBuf * tmp = nullptr;
    while(ptr != nullptr){
        this->cache_hierarchy->moveData(this->getCacheDataPtr(cache_idx), ptr->data);
        LOGGING(atomicAdd(&(logger->propogate_count), 1));
        // tmp = ptr->cache_next;
        tmp = (AgileBuf *) atomicLoadPtr((void**)&(ptr->cache_next));
        // ptr->cache_next = nullptr;
        atomicExchangePtr((void**)&(ptr->cache_next), (void*)nullptr);
        ptr->ready();
        ptr = tmp;
        tmp = nullptr;
    }
    
    // bufChain[cache_idx] = nullptr;
    atomicExchangePtr((void**)&(bufChain[cache_idx]), (void*)nullptr);
}

__device__ bool GPUCacheBase_T::checkEmpty_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return (stat == AGILE_GPUCACHE_EMPTY);
}

__device__ bool GPUCacheBase_T::readReadyCheck_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return (stat == AGILE_GPUCACHE_READY || stat == AGILE_GPUCACHE_WRITE_PROCESSING || stat == AGILE_GPUCACHE_MODIFIED);
}

__device__ void GPUCacheBase_T::setModified_inLoadArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat != AGILE_GPUCACHE_READ_PROCESSING, "setModified_inLoadArea error gpu cache");
    GPU_ASSERT(stat != AGILE_GPUCACHE_WRITE_PROCESSING, "setModified_inLoadArea error gpu cache");
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_MODIFIED);
}

__device__ bool GPUCacheBase_T::writableCheck_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return (stat == AGILE_GPUCACHE_EMPTY || stat == AGILE_GPUCACHE_READY || stat == AGILE_GPUCACHE_MODIFIED);
}

__device__ bool GPUCacheBase_T::checkErasable_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat != AGILE_GPUCACHE_READ_PROCESSING && stat != AGILE_GPUCACHE_WRITE_PROCESSING;
}

__device__ bool GPUCacheBase_T::checkInReadProcessing_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat == AGILE_GPUCACHE_READ_PROCESSING;
}

__device__ bool GPUCacheBase_T::checkInWriteProcessing_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat == AGILE_GPUCACHE_WRITE_PROCESSING;
}

__device__ bool GPUCacheBase_T::processingReading_inLockArea(unsigned int cache_idx) {
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat != AGILE_GPUCACHE_READ_PROCESSING, "processingReading_inLockArea error gpu cache");
    GPU_ASSERT(stat != AGILE_GPUCACHE_WRITE_PROCESSING, "processingReading_inLockArea error gpu cache");
    GPU_ASSERT(stat != AGILE_GPUCACHE_READY, "processingReading_inLockArea error gpu cache");
    GPU_ASSERT(stat != AGILE_GPUCACHE_MODIFIED, "processingReading_inLockArea error gpu cache");
    if(stat != AGILE_GPUCACHE_EMPTY){
        printf("processing, %d (%d) stat: %d\n", cache_idx, this->slot_num, stat);
    }
    GPU_ASSERT(stat == AGILE_GPUCACHE_EMPTY, "processingReading_inLockArea error gpu cache");
    // printf("set reading, %d\n", cache_idx);
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_READ_PROCESSING);
    // __threadfence_system();
}

__device__ bool GPUCacheBase_T::processingWriting_inLockArea(unsigned int cache_idx) {
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat == AGILE_GPUCACHE_MODIFIED || stat == AGILE_GPUCACHE_BUSY, "processingWriting_inLockArea error gpu cache");
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_WRITE_PROCESSING);
}

__device__ void GPUCacheBase_T::finishReading_inLockArea(unsigned int cache_idx) {
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    if(stat != AGILE_GPUCACHE_READ_PROCESSING){
        printf("stat error: %d %d %p\n", cache_idx, stat, &(this->cache_status[cache_idx]));
    }else{
        // printf("set finish, %d\n", cache_idx);
    }
    GPU_ASSERT(stat == AGILE_GPUCACHE_READ_PROCESSING, "finishReading_inLockArea error gpu cache");
    // this->cache_status[cache_idx] = AGILE_GPUCACHE_READY;
    // printf("finish %d\n", cache_idx);
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_READY);
}

__device__ void GPUCacheBase_T::finishWriting_inLockArea(unsigned int cache_idx) {
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat == AGILE_GPUCACHE_WRITE_PROCESSING, "finishWriting_inLockArea error gpu cache");
    // this->cache_status[cache_idx] = AGILE_GPUCACHE_READY;
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_READY);
}

__device__ void GPUCacheBase_T::finishModifying_inLockArea(unsigned int cache_idx) {
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat == AGILE_GPUCACHE_EMPTY || stat == AGILE_GPUCACHE_READY || stat == AGILE_GPUCACHE_MODIFIED, "modifying error gpu cache");
    // this->cache_status[cache_idx] = AGILE_GPUCACHE_MODIFIED;
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_MODIFIED);
}

__device__ bool GPUCacheBase_T::isModified_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat == AGILE_GPUCACHE_MODIFIED;
}

__device__ bool GPUCacheBase_T::hitStatus_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat != AGILE_GPUCACHE_EMPTY;
}

__device__ void GPUCacheBase_T::resetSlot_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat != AGILE_GPUCACHE_READ_PROCESSING && stat != AGILE_GPUCACHE_WRITE_PROCESSING, "resetSlot_inLockArea");
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_EMPTY);
    // printf("reset, %d\n", cache_idx);
}

__device__ unsigned int GPUCacheBase_T::getStatus_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat;
}

// __device__ bool GPUCacheBase_T::incReference(unsigned int cache_idx){
//     unsigned int thresh = AGILE_GPUCACHE_READY;
//     incReference_Start_l0:
//     unsigned int old_val = AGILE_GPUCACHE_READY;
//     unsigned int expected = AGILE_GPUCACHE_READY;
//     incReference_Start_l1:
//     unsigned int target = expected + 1;
//     old_val = atomicCAS(&(this->cache_status[cache_idx]), expected, target);
//     if(old_val == expected){
//         return true;
//     }else if(old_val >= AGILE_GPUCACHE_READY){
//         expected = old_val;
//         goto incReference_Start_l1;
//     }else if(old_val == AGILE_GPUCACHE_READ_PROCESSING){ // wait for read processing
//         goto incReference_Start_l0;
//     }
//     return false;
// }

// __device__ bool GPUCacheBase_T::decReference(unsigned int cache_idx){
//     unsigned int stat = atomicSub(&(this->cache_status[cache_idx]), 1);
//     if(stat < AGILE_GPUCACHE_READY){
//         printf("decReference error: %d %d\n", cache_idx, stat);
//     }
//     GPU_ASSERT(stat >= AGILE_GPUCACHE_READY, "decReference error");
// }

__device__ bool GPUCacheBase_T::incReference(unsigned int cache_idx){
    atomicAdd(&(this->cache_ref[cache_idx]), 1);
    return true;
}

__device__ bool GPUCacheBase_T::decReference(unsigned int cache_idx){
    unsigned int stat = atomicSub(&(this->cache_ref[cache_idx]), 1);
    GPU_ASSERT(stat != -1, "decReference error");
    return true;
}


// __device__ bool GPUCacheBase_T::incReference_inLockArea(unsigned int cache_idx){
//     incReference_Start_l0:
//     unsigned int old_val = 0;
//     unsigned int expected = 0;
//     incReference_Start_l1:
//     unsigned int target = expected + 1;
//     old_val = atomicCAS(&(this->cache_ref[cache_idx]), expected, target);
//     if(old_val == expected){
//         return true;
//     } else {
//         expected = old_val;
//         goto incReference_Start_l1;
//     }
//     return false;
// }

// __device__ bool GPUCacheBase_T::decReference(unsigned int cache_idx){
//     unsigned int stat = atomicSub(&(this->cache_ref[cache_idx]), 1);
//     GPU_ASSERT(stat != -1, "decReference error");
// }

// __device__ bool GPUCacheBase_T::notifyEvict_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
//     unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
//     GPU_ASSERT(stat != AGILE_GPUCACHE_EMPTY, "evict empty cache slot");
//     // GPU_ASSERT(stat != AGILE_GPUCACHE_READ_PROCESSING, "evict cache slot in reading processing");
//     // GPU_ASSERT(stat != AGILE_GPUCACHE_WRITE_PROCESSING, "evict cache slot in writing processing");
//     // try to evict to cpu first and then nvme, if all fail then discard
//     if(stat == AGILE_GPUCACHE_READ_PROCESSING || stat == AGILE_GPUCACHE_WRITE_PROCESSING){
//         return false;
//     }
   
//     bool succ = false;
//     if(stat == AGILE_GPUCACHE_READY){ 
//         succ = evictAttemptGPU2CPU();
//     } else if (stat == AGILE_GPUCACHE_MODIFIED){ // check modified
//         succ = true;
//         if(evictAttemptGPU2CPU()){
            
//         } else {
//             evictGPU2Nvme_inLockArea();
//         }
//     } else {
//         GPU_ASSERT(false, "status error");
//     }

//     return succ;
// }


/***** END GPUCacheBase_T ******/

/**********************************/

/***** START CPUCacheBase_T ******/

__host__ CPUCacheBase_T::CPUCacheBase_T(unsigned int slot_num, unsigned int slot_size) : AgileCacheBase(slot_num, slot_size, "cpu") {
    cuda_err_chk(cudaMalloc(&(this->cache_status), sizeof(unsigned int) * slot_num));
}

__host__ void CPUCacheBase_T::setPinedMem(void * data, unsigned long physical_addr){
    this->data = data;
    this->physical_addr = physical_addr;
}

__host__ CPUCacheBase_T::~CPUCacheBase_T(){
    // TODO free the following buffer
    // cuda_err_chk(cudaFree(this->phy_addr_table));
    // cuda_err_chk(cudaFree(this->data));
    // cuda_err_chk(cudaFree(this->cache_status));
}

__host__ unsigned int CPUCacheBase_T::getRequiredMemSize(){
    return this->slot_num * this->slot_size;
}

__device__ void CPUCacheBase_T::finishWriting_inLockArea(unsigned int cache_idx, unsigned int status) {
    // this->cache_status[cache_idx] = status;
    // GPU_ASSERT(atomicAdd(&(this->cache_status[cache_idx]), 0) == );
    // atomicCAS();
    atomicExch(&(this->cache_status[cache_idx]), status);
}

__device__ void * CPUCacheBase_T::getCacheDataPtr(unsigned int cache_idx) {
    // unsigned int ptr_idx = ((unsigned long)cache_idx) * cache_hierarchy->ctrl->buf_size / (1024 * 1024 * 4);
    // unsigned int ptr_offset = ((unsigned long)cache_idx) * cache_hierarchy->ctrl->buf_size % (1024 * 1024 * 4);
    return (this->data + cache_idx * cache_hierarchy->ctrl->buf_size);
}

__device__ bool CPUCacheBase_T::processingWriting_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat != AGILE_CPUCACHE_EVICTING, "cannot write a slot in evicting"); // TODO check here 
    return stat != AGILE_CPUCACHE_EVICTING;
}

__device__ void CPUCacheBase_T::resetSlot_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    GPU_ASSERT(stat == AGILE_CPUCACHE_READY || stat == AGILE_CPUCACHE_EMPTY, "resetSlot_inLockArea");
    atomicExch(&(this->cache_status[cache_idx]), AGILE_GPUCACHE_EMPTY);
}


__device__ bool CPUCacheBase_T::checkErasable_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat != AGILE_CPUCACHE_MODIFIED && stat != AGILE_CPUCACHE_EVICTING;
}

__device__ bool CPUCacheBase_T::checkEmpty_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat == AGILE_CPUCACHE_EMPTY;
}

__device__ bool CPUCacheBase_T::hitStatus_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat != AGILE_CPUCACHE_EMPTY;
}

__device__ unsigned int CPUCacheBase_T::getStatus_inLockArea(unsigned int cache_idx){
    unsigned int stat = atomicAdd(&(this->cache_status[cache_idx]), 0);
    return stat;
}

/***** END GPUCacheBase ******/

/**********************************/

/***** START GPUCacheBase ******/

template <typename GPUCacheImpl>
__host__ GPUCacheBase<GPUCacheImpl>::GPUCacheBase(unsigned int slot_num, unsigned int slot_size) : GPUCacheBase_T(slot_num, slot_size){

}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__device__ bool GPUCacheBase_T::notifyEvict_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, bool *inProcessing, AgileLockChain * chain){
    // printf("try to evict %d\n", cache_idx);
    return static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(this->cache_hierarchy)->evictGPUCache_inLockArea(ssd_dev_idx, ssd_blk_idx, cache_idx, inProcessing, chain);
}

/***** END GPUCacheBase ******/

/**********************************/

/***** START CPUCacheBase ******/

template <typename CPUCacheImpl>
__host__ CPUCacheBase<CPUCacheImpl>::CPUCacheBase(unsigned int slot_num, unsigned int slot_size) : CPUCacheBase_T(slot_num, slot_size){

}


template <typename CPUCacheImpl>
__device__ bool CPUCacheBase<CPUCacheImpl>::notifyEvict_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, bool *inProcessing, AgileLockChain * chain){
    return this->cache_hierarchy->evictCPUCache_inLockArea(ssd_dev_idx, ssd_blk_idx, cache_idx, inProcessing, chain); // check if modified, if yes then issue write cpu memory
}

/***** END CPUCacheBase ******/
