#ifndef AGILE_SWCACHE
#define AGILE_SWCACHE

#include <stdio.h>

#include "agile_lock.h"
#include "agile_nvme.h"
#include "agile_buf.h"


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileCtrl;

class AgileCacheHierarchyBase;

/**
 * AgileCache is used for readonly. 
 */
class AgileCacheBase {
    AgileLock * locks;
public:
    AgileCacheHierarchyBase * cache_hierarchy;

    // CacheStatus * status; // processing in GPUCacheBase_T and CPUCacheBase_T due to different behavior
    
    unsigned int slot_num;
    unsigned int slot_size; // how many uint32

    __host__ AgileCacheBase(unsigned int slot_num, unsigned int slot_size, char type[5]);

    __host__ ~AgileCacheBase();

    __host__ void setCacheHierarchy(AgileCacheHierarchyBase * cache_hierarchy);

    __device__ void acquireBaseLock_lockStart(unsigned int cache_idx, AgileLockChain * chain);

    __device__ bool acquireBaseLockAttempt_lockStart(unsigned int cache_idx, AgileLockChain * chain);

    __device__ void releaseBaseLock_lockEnd(unsigned int cache_idx, AgileLockChain * chain);


};

#define AGILE_GPUCACHE_EMPTY 0
#define AGILE_GPUCACHE_READ_PROCESSING 1
#define AGILE_GPUCACHE_WRITE_PROCESSING 2
#define AGILE_GPUCACHE_MODIFIED 3
#define AGILE_GPUCACHE_BUSY 4
#define AGILE_GPUCACHE_READY 5


// have functions for hierarachy
class GPUCacheBase_T : public AgileCacheBase {

    // TODO: enmu cannot use atomicAdd, which is unsafe
    // enum class GPUCacheStatus : unsigned int {
    //     EMPTY = 0,
    //     READ_PROCESSING = 1,
    //     WRITE_PROCESSING = 2, 
    //     READY = 3,
    //     MODIFIED = 4
    //     // one refer = 3 ...
    // };

public: 
    
    unsigned int * cache_status; // processing in GPUCacheBase_T and CPUCacheBase_T due to different behavior
    unsigned int * cache_ref; // reference count, used for CPUCacheBase_T
    AgileBuf ** bufChain; // a chain of buffers that need to be filled with data in cache slots
    

    void * data; // exposed GPU memory, NVME can write to it directly, used as sw cache
    unsigned long physical_addr;

    __host__ GPUCacheBase_T(unsigned int slot_num, unsigned int slot_size);

    __host__ ~GPUCacheBase_T();

    __host__ void setPinedMem(void * data, unsigned long physical_addr);

    __host__ unsigned int getRequiredMemSize();


    /**** start gpu cache data operation ****/

    // __device__ bool invokeHierarchyEvictAttemptGPU2CPU_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int tid, AgileLockChain * chain){
    //     return this->getCacheHierarchyPtr()->evictAttemptGPU2CPU_inLockArea(ssd_dev_idx, ssd_blk_idx, tid, chain);
    // }
    template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
    __device__ bool notifyEvict_inLockArea(NVME_DEV_IDX_TYPE old_ssd_dev_idx, SSDBLK_TYPE old_ssd_blk_idx, unsigned int cache_idx, bool *inProcessing, AgileLockChain * chain);



    __device__ void * getCacheDataPtr(unsigned int cache_idx);

    

    /**** end gpu cache data operation ****/

    /***************************************/

    /**** start buffer chain operation ****/

    __device__ void appendAgileBuf_inLockArea(unsigned int cache_idx, AgileBuf * buf);

    __device__ void propagateAgileBuf_inLockArea(unsigned int cache_idx);
    /**** end buffer chain operation ****/

    /***************************************/

    /**** start gpu cache status operation ****/

    __device__ unsigned int getStatus_inLockArea(unsigned int cache_idx);

    __device__ void resetSlot_inLockArea(unsigned int cache_idx);

    __device__ void setModified_inLoadArea(unsigned int cache_idx);

    __device__ bool checkErasable_inLockArea(unsigned int cache_idx);

    __device__ bool readReadyCheck_inLockArea(unsigned int cache_idx);

    __device__ bool checkEmpty_inLockArea(unsigned int cache_idx);

    __device__ bool writableCheck_inLockArea(unsigned int cache_idx);

    __device__ bool checkInReadProcessing_inLockArea(unsigned int cache_idx);

    __device__ bool checkInWriteProcessing_inLockArea(unsigned int cache_idx);

    __device__ bool processingReading_inLockArea(unsigned int cache_idx);

    __device__ bool processingWriting_inLockArea(unsigned int cache_idx);

    __device__ void finishReading_inLockArea(unsigned int cache_idx);

    __device__ void finishWriting_inLockArea(unsigned int cache_idx);

    __device__ void finishModifying_inLockArea(unsigned int cache_idx);

    __device__ bool isModified_inLockArea(unsigned int cache_idx);

    __device__ bool hitStatus_inLockArea(unsigned int cache_idx);

     /**** end gpu cache status operation ****/

    __device__ bool incReference(unsigned int cache_idx);
    __device__ bool decReference(unsigned int cache_idx);

    // __device__ bool notifyEvictAndCheckAvailability_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain);



};


#define AGILE_CPUCACHE_EMPTY 0
#define AGILE_CPUCACHE_READY 1
#define AGILE_CPUCACHE_MODIFIED 2
#define AGILE_CPUCACHE_EVICTING 3

class CPUCacheBase_T : public AgileCacheBase {
public:

    // unsigned int table_size;
    // unsigned long * phy_addr_table; // now reserve a large space for cpu cache in /etc/defaul/grub, no need for this
    void * data; // point to CPU host pinned memory not continuous
    unsigned long physical_addr;

    // enum class CPUCacheStatus : unsigned int {
    //     EMPTY = 0,
    //     READY = 1,
    //     MODIFIED = 2
    //     // one refer = 3 ...
    // };

    unsigned int * cache_status;

    __host__ CPUCacheBase_T(unsigned int slot_num, unsigned int slot_size);

    __host__ ~CPUCacheBase_T();

    __host__ void setPinedMem(void * data, unsigned long physical_addr);
    
    __host__ unsigned int getRequiredMemSize();
    /**** start cpu cache data operation ****/

    // __device__ void moveDataCPU2Buf_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr);

    /**** end cpu cache data operation ****/

    /***************************************/

    /**** start cpu cache status operation ****/

    __device__ void resetSlot_inLockArea(unsigned int cache_idx);

    __device__ bool checkErasable_inLockArea(unsigned int cache_idx);

    __device__ bool checkEmpty_inLockArea(unsigned int cache_idx);

    __device__ bool processingWriting_inLockArea(unsigned int cache_idx);

    __device__ void finishWriting_inLockArea(unsigned int cache_idx, unsigned int status);

    __device__ bool hitStatus_inLockArea(unsigned int cache_idx);

    __device__ unsigned int getStatus_inLockArea(unsigned int cache_idx);

    /**** end cpu cache status operation ****/

    __device__ void * getCacheDataPtr(unsigned int cache_idx);


};


template <typename GPUCacheImpl>
class GPUCacheBase : public GPUCacheBase_T {
public:

    __host__ GPUCacheBase(unsigned int slot_num, unsigned int slot_size);


    // __device__ void markWrite(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * bufPtr, AgileLockChain * chain){
    //     static_cast<GPUCacheImpl *>(this)->markWriteImpl(ssd_dev_idx, ssd_blk_idx, bufPtr, chain);
    // }
    /** 
    * the following functions' *Impl version need to be implemented in the child class.
    */
    __device__ bool checkCacheHitAcquireLock_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int * cache_idx, AgileLockChain * chain){
        bool hit = static_cast<GPUCacheImpl *>(this)->checkCacheHitAcquireLockImpl_lockStart(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
        return hit && !this->checkEmpty_inLockArea(*cache_idx);
    }

    __device__ void releaseSlotLock_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
        static_cast<GPUCacheImpl *>(this)->releaseSlotLockImpl_lockEnd(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
    }

    __device__ void getTaginfo_inLockArea(NVME_DEV_IDX_TYPE *ssd_dev_idx, SSDBLK_TYPE *ssd_blk_idx, unsigned int cache_idx){
        static_cast<GPUCacheImpl *>(this)->getTaginfoImpl_inLockArea(ssd_dev_idx, ssd_blk_idx, cache_idx);
    }

    template<typename T>
    __device__ bool getCacheElement(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int idx, T * val, AgileLockChain * chain){
        return static_cast<GPUCacheImpl *>(this)->getCacheElementImpl(ssd_dev_idx, ssd_blk_idx, idx, val, chain);
    }

    __device__ unsigned int getPossibleGPUCacheIdx(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return static_cast<GPUCacheImpl *>(this)->getPossibleGPUCacheIdxImpl(ssd_dev_idx, ssd_blk_idx);
    }

    __device__ bool checkHit(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx){
        return static_cast<GPUCacheImpl *>(this)->checkHitImpl(ssd_dev_idx, ssd_blk_idx, cache_idx);
    }

    __device__ unsigned int checkCacheHit_relaxed(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return static_cast<GPUCacheImpl *>(this)->checkCacheHitImpl_relaxed(ssd_dev_idx, ssd_blk_idx);
    } 
    // __device__ bool checkCacheHitAttemptAcquireLock_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, AgileBuf * buf, bool * hit, AgileLockChain * chain){
    //     bool aquired = static_cast<GPUCacheImpl *>(this)->checkCacheHitAttemptAcquireLockImpl_lockStart(ssd_dev_idx, ssd_blk_idx, cache_idx, buf, hit, chain);
    //     return aquired;
    // }

    // __device__ void updateCacheTag_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
    //     static_cast<GPUCacheImpl *>(this)->updateCacheTagImpl_inLockArea(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
    // }


    

};


template <typename CPUCacheImpl>
class CPUCacheBase : public CPUCacheBase_T {
public:
    __host__ CPUCacheBase(unsigned int slot_num, unsigned int slot_size);

    __device__ bool notifyEvict_inLockArea(NVME_DEV_IDX_TYPE old_ssd_dev_idx, SSDBLK_TYPE old_ssd_blk_idx, unsigned int cache_idx, bool *inProcessing, AgileLockChain * chain);
    
    /** 
    * the following functions' *Impl version need to be implemented in the child class.
    */

    __device__ bool checkCacheHitAcquireLock_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, AgileLockChain * chain){
        bool hit = static_cast<CPUCacheImpl *>(this)->checkCacheHitAcquireLockImpl_lockStart(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
        return hit && !this->checkEmpty_inLockArea(*cache_idx); 
    }

    __device__ bool checkCacheHitAttemptAcquireLock_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *cache_idx, bool *hit, AgileLockChain * chain){
        bool acquired = static_cast<CPUCacheImpl *>(this)->checkCacheHitAttemptAcquireLockImpl_lockStart(ssd_dev_idx, ssd_blk_idx, cache_idx, hit, chain);
        return acquired;
    }

    // __device__ void updateCacheTag_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
    //     static_cast<CPUCacheImpl *>(this)->updateCacheTagImpl_inLockArea(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
    // }

    __device__ void getTaginfo_inLockArea(NVME_DEV_IDX_TYPE *ssd_dev_idx, SSDBLK_TYPE *ssd_blk_idx, unsigned int cache_idx){
        static_cast<CPUCacheImpl *>(this)->getTaginfoImpl_inLockArea(ssd_dev_idx, ssd_blk_idx, cache_idx);
    }

    __device__ void releaseSlotLock_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain){
        static_cast<CPUCacheImpl *>(this)->releaseSlotLockImpl_lockEnd(ssd_dev_idx, ssd_blk_idx, cache_idx, chain);
    }

};


#include "agile_swcache.tpp"

#endif