#ifndef AGILE_CACHE_HIERARCHY
#define AGILE_CACHE_HIERARCHY

#include "agile_ctrl.h"
// #include "agile_swcache.h"
class CPUCacheBase_T;
class GPUCacheBase_T;
class AgileCtrlBase;
class ShareTableBase_T;

class AgileCacheHierarchyBase {
public:
    GPUCacheBase_T * gpu_cache;
    CPUCacheBase_T * cpu_cache;
    ShareTableBase_T * share_table;
    AgileCtrlBase  * ctrl;

    __device__ bool evictCPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, bool* inProcessing, AgileLockChain * chain);

    __device__ void moveData(void * src, void * dst);
    
};

template <typename GPUCacheImpl>
class GPUCacheBase;

template <typename CPUCacheImpl>
class CPUCacheBase;

template<typename ShareTableImpl>
class ShareTableBase;

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileCacheHierarchy : public AgileCacheHierarchyBase {
public:
    __device__ GPUCacheBase<GPUCacheImpl> * getGPUCacheBasePtr(){
        return static_cast<GPUCacheBase<GPUCacheImpl> *>(this->gpu_cache);
    }

    __device__ CPUCacheBase<CPUCacheImpl> * getCPUCacheBasePtr(){
        return static_cast<CPUCacheBase<CPUCacheImpl> *>(this->cpu_cache);
    }

    __device__ AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * getAgileCtrlPtr(){
        return static_cast<AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(ctrl);
    }

    __device__ ShareTableBase<ShareTableImpl> * getShareTablePtr(){
        return static_cast<ShareTableBase<ShareTableImpl> *>(this->share_table);
    }

    /**
     * return if thie slot can be used immediately
     */
    __device__ bool evictGPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, bool *inProcessing, AgileLockChain * chain);

    /**
    * return true if the buf_ptr is appended to dst or gets data
    */
    // __device__ bool checkGPUCache(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);

    /**
    * Called after checkCPUCache. Check GPUCache agian and issue load from SSD to GPUCache if not exists
    */
    // __device__ void checkGPUCacheIssue(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);

    /**
    * Users should be allowed to choose if copy immediately or wait for agile service to copy.
    * return true if the buf_ptr is appended to dst. The CPU cache will be filled in Agile service to avoid page fault overhead
    */
    // __device__ bool checkCPUCache(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);
    
    // __device__ void writeGPUCache(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);

    /**
    * Move data and set cpu cache flags
    */
    // __device__ void moveDataGPU2CPU_in2LockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, unsigned int cpu_cache_idx);

    
    /**
    * This function will find a gpu cache slot for refilled data from bufptr
    */

    __device__ void fillOrAppendAgileBuf2GPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileBufPtr * buf);
     
    /** 
    * refillBuf2GPUCache() is invoked after reading data from CPU to a thread's buffer. This function will update GPU cache according to this buffer.
    */
    __device__ void refillBuf2GPUCache_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, AgileBufPtr * buf_ptr);

    __device__ void cacheRead(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    __device__ void cacheWrite(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    __device__ void writeThroughNvme(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    template<typename T>
    __device__ void writeThroughNvme_noRead(NVME_DEV_IDX_TYPE ssd_dev_idx, unsigned long idx, T val, AgileLockChain * chain);
    // __device__ void prepareShared_withoutRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);
    
    // __device__ void prepareShared_withRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    __device__ void checkShareTable(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr *& buf_ptr, unsigned int read, AgileLockChain * chain);

    // __device__ void checkTableWithoutRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    /**
    * evictAttemptGPU2CPU() need to be invoked in GPUCacheImpl's evict function
    * this function tries to find a valid cache slot in CPU cache, and move data to. 
    * Deadlock may happen if this function only return after write to CPU cache successfully 
    */
    __device__ bool evictAttemptGPU2CPU_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, AgileLockChain * chain);

    __device__ bool evictGPU2Nvme_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int gpu_cache_idx, AgileLockChain * chain);
    


    // /**
    // * cacheWriteThroughAttempt2CPU() cannot do hard write to CPU, otherwise W-W race or deadlock may happen: gpu-cache evict & this || locks from gpu-cache and cpu-cache at the same time
    // */
    // __device__ bool cacheWriteThroughAttempt2CPU(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);

    // /**
    // * cacheWriteThrough2Nvme() will write to NVME but not cpu cache
    // */
    // __device__ void cacheWriteThrough2Nvme(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);

    // __device__ void cacheWriteLateEvict(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf_ptr, AgileLockChain * chain);

};

#include "agile_cache_hierarchy.tpp"
#endif