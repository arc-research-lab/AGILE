#include "agile_swcache.h"
#include "agile_ctrl.h"
/**********************************/

class DisableCPUCache : public CPUCacheBase<DisableCPUCache> {
public:
    __host__ DisableCPUCache(unsigned int slot_num, unsigned int slot_size) : CPUCacheBase<DisableCPUCache>(slot_num, slot_size) {
    }

    __device__ bool checkCacheHitAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, AgileLockChain * chain){
        *cache_idx = -1;
        return false;
    }
    
    __device__ bool checkCacheHitAttemptAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, bool *hit, AgileLockChain * chain){
        *cache_idx = -1;
        return false;
    }
    
    __device__ void getTaginfoImpl_inLockArea(NVME_DEV_IDX_TYPE *ssd_dev_idx, SSDBLK_TYPE *ssd_blk_idx, unsigned int cache_idx){
        *ssd_dev_idx = -1;
        *ssd_blk_idx = -1;
    }
    
    __device__ void releaseSlotLockImpl_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
    }
};

class SimpleCPUCache : public CPUCacheBase<SimpleCPUCache> {
    NVME_DEV_IDX_TYPE * tag_dev_id;
    SSDBLK_TYPE * tag_blk_id;
    
public:
    SimpleCPUCache(unsigned int slot_num, unsigned int slot_size) : CPUCacheBase<SimpleCPUCache>(slot_num, slot_size) {
        cuda_err_chk(cudaMalloc(&(this->tag_dev_id), sizeof(NVME_DEV_IDX_TYPE) * slot_num));
        cuda_err_chk(cudaMalloc(&(this->tag_blk_id), sizeof(SSDBLK_TYPE) * slot_num));
    }

    __device__ bool checkCacheHitAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, AgileLockChain * chain){
        unsigned int counter = 0;
        CPUCache_checkCacheHitAcquireLockImpl_lockStart:
        bool hit = false;
        *cache_idx = ssd_blk_idx % this->slot_num;
        this->acquireBaseLock_lockStart(*cache_idx, chain);
        if(tag_dev_id[*cache_idx] == ssd_dev_idx && tag_blk_id[*cache_idx] == ssd_blk_idx){
            hit = true;
        }else{
            bool inprocessing;
            if(this->notifyEvict_inLockArea(tag_dev_id[*cache_idx], tag_blk_id[*cache_idx], *cache_idx, &inprocessing, chain)){
                tag_dev_id[*cache_idx] = ssd_dev_idx;
                tag_blk_id[*cache_idx] = ssd_blk_idx;
            }else{
                this->releaseBaseLock_lockEnd(*cache_idx, chain);
                __threadfence();
                busyWait(1000);
                counter++;
                if(counter == 1000){
                    counter = 0;
                    printf("try to evict CPU cache failed too many times\n");
                    LOGGING(atomicAdd(&(logger->deadlock_check), 1));
                }
                goto CPUCache_checkCacheHitAcquireLockImpl_lockStart;
            }
        }
        return hit;
    }
    
    __device__ bool checkCacheHitAttemptAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, bool *hit, AgileLockChain * chain){
        *hit = false;
        bool acquired = false;
        *cache_idx = ssd_blk_idx % this->slot_num;
        if(this->acquireBaseLockAttempt_lockStart(*cache_idx, chain)){
            acquired = true;
            if(tag_dev_id[*cache_idx] == ssd_dev_idx && tag_blk_id[*cache_idx] == ssd_blk_idx){
                *hit = true;
            }else{
                bool inprocessing;
                if(this->notifyEvict_inLockArea(tag_dev_id[*cache_idx], tag_blk_id[*cache_idx], *cache_idx, &inprocessing, chain)){
                    tag_dev_id[*cache_idx] = ssd_dev_idx;
                    tag_blk_id[*cache_idx] = ssd_blk_idx;
                }else{
                    this->releaseBaseLock_lockEnd(*cache_idx, chain);
                    acquired = false;
                }
            }
        }
        return acquired;
    }
    
    __device__ void getTaginfoImpl_inLockArea(NVME_DEV_IDX_TYPE *ssd_dev_idx, SSDBLK_TYPE *ssd_blk_idx, unsigned int cache_idx){
        *ssd_dev_idx = tag_dev_id[cache_idx];
        *ssd_blk_idx = tag_blk_id[cache_idx];
    }
    
    __device__ void releaseSlotLockImpl_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
        this->releaseBaseLock_lockEnd(cache_idx, chain);
    }
};

template <typename CPUCacheImpl, typename WriteTableImpl>
class SimpleGPUCache : public GPUCacheBase<SimpleGPUCache<CPUCacheImpl, WriteTableImpl> > {
public:
    NVME_DEV_IDX_TYPE * tag_dev_id;
    SSDBLK_TYPE * tag_blk_id;

    SimpleGPUCache(unsigned int slot_num, unsigned int slot_size) : GPUCacheBase<SimpleGPUCache>(slot_num, slot_size) {
        cuda_err_chk(cudaMalloc(&(this->tag_dev_id), sizeof(NVME_DEV_IDX_TYPE) * slot_num));
        cuda_err_chk(cudaMalloc(&(this->tag_blk_id), sizeof(SSDBLK_TYPE) * slot_num));
    }

    __device__ bool checkCacheHitAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, AgileLockChain * chain){
        unsigned int counter = 0;
        GPUCache_checkCacheHitAcquireLockImpl_lockStart:
        bool hit = false;
        *cache_idx = ssd_blk_idx % this->slot_num;
        this->acquireBaseLock_lockStart(*cache_idx, chain);
        if(tag_dev_id[*cache_idx] == ssd_dev_idx && tag_blk_id[*cache_idx] == ssd_blk_idx){
            hit = true;
        }else{
            bool inprocessing;
            if(static_cast<GPUCacheBase_T*>(this)->notifyEvict_inLockArea<SimpleGPUCache, CPUCacheImpl, WriteTableImpl>(tag_dev_id[*cache_idx], tag_blk_id[*cache_idx], *cache_idx, &inprocessing, chain)){
                tag_dev_id[*cache_idx] = ssd_dev_idx;
                tag_blk_id[*cache_idx] = ssd_blk_idx;
            }else{
                this->releaseBaseLock_lockEnd(*cache_idx, chain);
                __threadfence();
                busyWait(1000);
                counter++;
                if(counter == 100000){
                    counter = 0;
                    LOGGING(atomicAdd(&(logger->deadlock_check), 1));
                    // printf("try to evict GPU cache failed too many times\n");
                }
                goto GPUCache_checkCacheHitAcquireLockImpl_lockStart;
            }
        }
        return hit;
    }

    __device__ void getTaginfoImpl_inLockArea(NVME_DEV_IDX_TYPE *ssd_dev_idx, SSDBLK_TYPE *ssd_blk_idx, unsigned int cache_idx){
        *ssd_dev_idx = tag_dev_id[cache_idx];
        *ssd_blk_idx = tag_blk_id[cache_idx];
    }

    __device__ void releaseSlotLockImpl_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
        this->releaseBaseLock_lockEnd(cache_idx, chain);
    }

    __device__ unsigned int getPossibleGPUCacheIdxImpl(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return ssd_blk_idx % this->slot_num;
    }

    template<typename T>
    __device__ bool getCacheElementImpl(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int idx, T * val, AgileLockChain * chain){
        
        unsigned int c_idx = ssd_blk_idx % this->slot_num;
        
        bool possible_in_cache = this->incReference(c_idx); // increase reference count only when the cache line is ready, other wise it may fail
        bool find = false;

        if(possible_in_cache){
            if(atomicAdd(&(tag_dev_id[c_idx]), 0) == ssd_dev_idx && atomicAdd(&(tag_blk_id[c_idx]), 0) == ssd_blk_idx){ // when the cache line is ready, check the tag
                *val = static_cast<T*>(this->getCacheDataPtr(c_idx))[idx];
                find = true;
            }
            this->decReference(c_idx);
        }
        return find;
    }

};


template <typename CPUCacheImpl, typename WriteTableImpl>
class GPUClockReplacementCache : public GPUCacheBase<GPUClockReplacementCache<CPUCacheImpl, WriteTableImpl> >{
    // NVME_DEV_IDX_TYPE * tag_dev_id;
    SSDBLK_TYPE * tag_blk_id; // store the ssd block index for cache line idx
    NVME_DEV_IDX_TYPE * tag_dev_id;
    SSDBLK_TYPE * ssd_blk_record; // store the cache line idx for ssd block
    AgileLock * ssd_blk_lock;
    unsigned int * ref; // reference bit for eviction: 0 (empty) -> 2 (just used) -> 1 (possible victim) -> 0 (empty)
    unsigned int clock; // find next victim
    unsigned int ssd_block_num;
public:

    __host__ GPUClockReplacementCache(unsigned int slot_num, unsigned int slot_size, unsigned int ssd_block_num) : GPUCacheBase<GPUClockReplacementCache>(slot_num, slot_size) {
        cuda_err_chk(cudaMalloc(&(this->tag_dev_id), sizeof(NVME_DEV_IDX_TYPE) * slot_num));
        this->ssd_block_num = ssd_block_num;
        // printf("GPUClockReplacementCache: slot_num %d, ssd_block_num %d\n", slot_num, ssd_block_num);
        this->clock = 0;
        cuda_err_chk(cudaMalloc(&(this->ssd_blk_record), sizeof(SSDBLK_TYPE) * ssd_block_num));
        cuda_err_chk(cudaMalloc(&(this->tag_blk_id), sizeof(SSDBLK_TYPE) * slot_num));
        cuda_err_chk(cudaMalloc(&(this->ref), sizeof(unsigned int) * slot_num));
        cuda_err_chk(cudaMemset(this->ref, 0, sizeof(unsigned int) * slot_num));
        cuda_err_chk(cudaMalloc(&(this->ssd_blk_lock), sizeof(AgileLock) * ssd_block_num));
        AgileLock * h_lock = (AgileLock *)malloc(sizeof(AgileLock) * ssd_block_num);
        for(unsigned int i = 0; i < ssd_block_num; i++){
#if LOCK_DEBUG
            char name[20];
            sprintf(name, "g-cr-cache-%d", i);
            h_lock[i] = AgileLock(name);
#else
            h_lock[i] = AgileLock();
#endif
        }
        cuda_err_chk(cudaMemcpy(this->ssd_blk_lock, h_lock, sizeof(AgileLock) * ssd_block_num, cudaMemcpyHostToDevice));
        free(h_lock);
        SSDBLK_TYPE * h_records = (SSDBLK_TYPE *)malloc(sizeof(SSDBLK_TYPE) * ssd_block_num);
        SSDBLK_TYPE * h_tag = (SSDBLK_TYPE *)malloc(sizeof(SSDBLK_TYPE) * slot_num);
        for(SSDBLK_TYPE i = 0; i < slot_num; i++){
            h_tag[i] = -1;
        }
        for(SSDBLK_TYPE i = 0; i < ssd_block_num; i++){
            h_records[i] = -1;
        }
        cuda_err_chk(cudaMemcpy(this->tag_blk_id, h_tag, sizeof(SSDBLK_TYPE) * slot_num, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->ssd_blk_record, h_records, sizeof(SSDBLK_TYPE) * ssd_block_num, cudaMemcpyHostToDevice));
        free(h_tag);
        free(h_records);
    }

    // this will ensure to find a empty slot
    __device__ unsigned int clockOrderFindNext_lockStart(AgileLockChain * chain){
        
        clockOrderCheck_getNext:
        LOGGING(atomicAdd(&(logger->deadlock_check), 1));
        unsigned int c_idx = atomicAdd(&clock, 1) % this->slot_num;
        bool acquired = this->acquireBaseLockAttempt_lockStart(c_idx, chain);
        if(acquired){
            unsigned int ref_count = atomicAdd(&(ref[c_idx]), 0);
            if(ref_count == 0){
                return c_idx;
            }else if(ref_count == 1){
                bool inprocessing;
                bool evicted = static_cast<GPUCacheBase_T*>(this)->notifyEvict_inLockArea<GPUClockReplacementCache, CPUCacheImpl, WriteTableImpl>(tag_dev_id[c_idx], tag_blk_id[c_idx], c_idx, &inprocessing, chain);
                if(evicted){
                    // ref[c_idx] = 0;
                    atomicExch(&(ref[c_idx]), 0);
                    return c_idx;
                }else{
                    this->releaseBaseLock_lockEnd(c_idx, chain);
                    goto clockOrderCheck_getNext;
                }
                return c_idx;
            }else{
                // ref[c_idx] = 1;
                atomicExch(&(ref[c_idx]), 1);
                this->releaseBaseLock_lockEnd(c_idx, chain);
                goto clockOrderCheck_getNext;
            }
        }else{
            goto clockOrderCheck_getNext;
        }
    }

    __device__ bool checkCacheHitAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int * cache_idx, AgileLockChain * chain){
        
        if(ssd_blk_idx > ssd_block_num){
            printf("GPUClockReplacementCache: ssd_blk_idx out of range %ld %d\n", ssd_blk_idx, ssd_block_num);
        }
        GPU_ASSERT(ssd_blk_idx < ssd_block_num, "GPUClockReplacementCache: ssd_blk_idx out of range");
        // GPU_ASSERT(ssd_dev_idx == 0, "GPUClockReplacementCache: only support one device");

        bool hit = false;
        this->ssd_blk_lock[ssd_blk_idx].acquire(chain);
        unsigned int idx = atomicAdd(&(ssd_blk_record[ssd_blk_idx]), 0);
        if(idx == -1){ // empty
            LOGGING(atomicAdd(&(logger->find_new_cacheline), 1));
            *cache_idx = clockOrderFindNext_lockStart(chain); // find next empty slot and lock
            atomicExch(&(tag_blk_id[*cache_idx]), ssd_blk_idx);
            atomicExch(&(tag_dev_id[*cache_idx]), ssd_dev_idx);
            atomicExch(&(ssd_blk_record[ssd_blk_idx]), *cache_idx);
            atomicExch(&(ref[*cache_idx]), 2);
        } else {
            this->acquireBaseLock_lockStart(idx, chain);
            if(atomicAdd(&(tag_blk_id[idx]), 0) == ssd_blk_idx && atomicAdd(&(tag_dev_id[idx]), 0) == ssd_dev_idx){ // check tag again
                hit = true;
                atomicExch(&(ref[idx]), 2);
                *cache_idx = idx;
            }else{ // if not equal, other ssd block is using this cache line
                this->releaseBaseLock_lockEnd(idx, chain); 
                *cache_idx = clockOrderFindNext_lockStart(chain);
                atomicExch(&(tag_blk_id[*cache_idx]), ssd_blk_idx);
                atomicExch(&(tag_dev_id[*cache_idx]), ssd_dev_idx);
                atomicExch(&(ssd_blk_record[ssd_blk_idx]), *cache_idx);
                atomicExch(&(ref[*cache_idx]), 2);
            }
        }
        this->ssd_blk_lock[ssd_blk_idx].release(chain);
        return hit;
    }

    __device__ void releaseSlotLockImpl_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
        // __threadfence_system();
        this->releaseBaseLock_lockEnd(cache_idx, chain);
    }

    __device__ void getTaginfoImpl_inLockArea(NVME_DEV_IDX_TYPE *ssd_dev_idx, SSDBLK_TYPE *ssd_blk_idx, unsigned int cache_idx){
        *ssd_dev_idx = atomicAdd(&(tag_dev_id[cache_idx]), 0); // tag_dev_id[cache_idx];
        *ssd_blk_idx = atomicAdd(&(tag_blk_id[cache_idx]), 0);

    }

    __device__ unsigned int getPossibleGPUCacheIdxImpl(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return atomicAdd(&(ssd_blk_record[ssd_blk_idx]), 0);
    }

    __device__ bool checkHitImpl(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx){
        bool hit = false;
        // GPU_ASSERT(ssd_dev_idx == 0, "GPUClockReplacementCache: only support one device");
        hit = (atomicAdd(&(tag_blk_id[cache_idx]), 0) == ssd_blk_idx && atomicAdd(&(tag_dev_id[cache_idx]), 0) == ssd_dev_idx);
        return hit;
    }

    __device__ unsigned int checkCacheHitImpl_relaxed(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        unsigned int rtn_idx = -1;
        // GPU_ASSERT(ssd_dev_idx == 0, "GPUClockReplacementCache: only support one device");
        unsigned int c_idx = atomicAdd(&(ssd_blk_record[ssd_blk_idx]), 0);
        if(c_idx == -1){
            return -1;
        }
        if(atomicAdd(&(tag_blk_id[c_idx]), 0) == ssd_blk_idx && atomicAdd(&(tag_dev_id[c_idx]), 0) == ssd_dev_idx){ // check tag again
            rtn_idx = c_idx;
        }
        return rtn_idx;
    } 

    template<typename T>
    __device__ bool getCacheElementImpl(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int idx, T * val, AgileLockChain * chain){
        // GPU_ASSERT(ssd_dev_idx == 0, "GPUClockReplacementCache: only support one device");
        unsigned int c_idx = atomicAdd(&(ssd_blk_record[ssd_blk_idx]), 0);
        if(c_idx == -1){
            return false;
        }
        bool possible_in_cache = this->incReference(c_idx); // increase reference count only when the cache line is ready, other wise it may fail
        bool find = false;

        if(possible_in_cache){
            if(atomicAdd(&(tag_blk_id[c_idx]), 0) == ssd_blk_idx && atomicAdd(&(tag_dev_id[c_idx]), 0) == ssd_dev_idx){ // when the cache line is ready, check the tag
                *val = static_cast<T*>(this->getCacheDataPtr(c_idx))[idx];
                find = true;
            }
            this->decReference(c_idx);
        }

        return find;
    }

};

