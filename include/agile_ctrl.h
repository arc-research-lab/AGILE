#ifndef AGILE_CTRL
#define AGILE_CTRL

#include "agile_lock.h"
#include "agile_swcache.h"
#include "agile_nvme.h"
#include "agile_buf_shared.h"
#include "agile_array_warp.h"
// #include "agile_logger.h"


// template <typename T>
// class AgileCacheBase;

// __device__ unsigned int *agile_stop_signal;
class AgileCacheHierarchyBase;

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileTableBuf;

class AgileCtrlBase {
    
public:
    AgileNvmeDev * dev;
    AgilePollingList * list;
    AgileCacheHierarchyBase * cache_hierarchy;
    unsigned long cpu_page_size;
    unsigned int buf_size; // how many bytes in each AgileBuf
    

#if FAKE_NVME
    void * emu_nvme;
    unsigned int emu_num_total;
    unsigned int emu_num_per_dev;
    unsigned int emu_size;
#endif

    __device__ void issueReadNvme2GPU(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain);
    __device__ void issueWriteGPU2Nvme(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain);
    __device__ void issueWriteCPU2Nvme(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain);
};

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileCtrl;

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileArrReadWrapIdx {
    NVME_DEV_IDX_TYPE dev_id;
    
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl;
    
public:
    AgileLockChain * chain;
    AgileBufPtr * buf;

    __device__ AgileArrReadWrapIdx(NVME_DEV_IDX_TYPE dev_id, AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain);

    __device__ const T operator[](unsigned long idx) const;

};

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileArrReadWrap {
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl;
public:
    AgileLockChain * chain;
    AgileBufPtr * buf;
    __device__ AgileArrReadWrap(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain);
    __device__ AgileArrReadWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> operator[](NVME_DEV_IDX_TYPE dev_id);

};


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileArrSharedWrapIdx {
public:
    NVME_DEV_IDX_TYPE dev_id;
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl;
    AgileLockChain * chain;
    AgileBufPtr * buf;

    unsigned int read;
    unsigned int write;

    __device__ AgileArrSharedWrapIdx(NVME_DEV_IDX_TYPE dev_id, AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain, unsigned int read, unsigned int write);

    __device__ T& operator[](unsigned long idx);

};

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileArrSharedWrap {
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl;
    unsigned int read;
    unsigned int write;
public:
    AgileLockChain * chain;
    AgileBufPtr * buf;

    __device__ void reset();

    __device__ ~AgileArrSharedWrap();

    __device__ AgileArrSharedWrap(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileBufPtr * buf, AgileLockChain * chain, unsigned int read, unsigned int write);

    __device__ AgileArrSharedWrapIdx<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> operator[](NVME_DEV_IDX_TYPE dev_id);

};

template<typename ShareTableImpl>
class ShareTableBase;

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileCtrl : public AgileCtrlBase{
public:
    BID_TYPE g_bid;
    BID_TYPE num_block_for_polling;
    BID_TYPE total_thread_block;

    TID_TYPE threads_per_block;

    // unsigned int finished_warps_num;
    // unsigned int finished_blocks_num;
    unsigned int compute_blocks;

    // unsigned int finished_polling;
    unsigned int *stop_signal;



    
    SSDBLK_TYPE * ssd_page_offset; 
    unsigned int block_per_cmd; // 1 if 512 bytes each cmd

    // __host__ AgileCtrl(BID_TYPE total_thread_block, TID_TYPE threads_per_block, BID_TYPE num_block_for_polling);

    // __host__ void setNvmeCache(GPUCacheImpl & gpu_cache);

    __host__ void setComputeBlocks(BID_TYPE compute_blocks);

    __host__ void setAgileBlocks(BID_TYPE agile_blocks);

    __host__ void setThreadsPreBlock(TID_TYPE threads_per_block);

    // __host__ AgileCtrl * getDevicePtr();

    __device__ bool startAgile(unsigned int & AGILE_BID);

    // __device__ bool stopAgile(unsigned int AGILE_BID);

    // __device__ void issueNvme2GPUCache(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int cache_idx, AgileLockChain * chain);

    __device__ void asyncRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    __device__ void asyncWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    __device__ void asyncRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr & buf_ptr, AgileLockChain & chain);

    // __device__ void asyncReadShared(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtrShared & buf_ptr, AgileLockChain & chain);

    __device__ void asyncWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr & buf_ptr, AgileLockChain & chain);

    __device__ void writeThroughNvme(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr & buf_ptr, AgileLockChain & chain);

    template <typename T>
    __device__ void writeThroughNvme_noRead_benchWrite(NVME_DEV_IDX_TYPE dev_idx, unsigned long idx, T val, AgileLockChain & chain);

    // __device__ void prepareWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr * buf_ptr, AgileLockChain * chain);

    __device__ unsigned int prefetch(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx);

    __device__ unsigned int prefetch_core(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx);

    __device__ void read_cache(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int line_offset, void * dst, unsigned int size);

    __device__ bool waitCpl(unsigned int queue_idx, unsigned int AGILE_BID);

    __device__ bool warpService(unsigned int AGILE_BID, unsigned int start, unsigned int end);

    __device__ void pollingService(unsigned int AGILE_BID);

    __device__ unsigned int waitCpl2(unsigned int queue_idx, unsigned int warp_idx, unsigned int cq_offset, unsigned int mask);

    __device__ bool warpService2(unsigned int queue_idx, unsigned int warp_idx, unsigned int & offset, unsigned int & mask);

    __device__ void pollingService2();

    __device__ void pollingServiceLast(unsigned int AGILE_BID);

    __device__ void checkCacheSlot_warpMasteAcquire(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int *gpu_cache_idx, AgileLockChain * chain);
    __device__ void checkCacheSlot_warpMasteRelease(unsigned int gpu_cache_idx);
    __device__ void waitCacheSlot_warpMaste(unsigned int gpu_cache_idx);
    template<typename T>
    __device__ T readCacheElement_inWarp(unsigned int gpu_cache_idx, unsigned int idx);
    

    // __device__ BID_TYPE BID();

    // __device__ TID_TYPE TID();

    // __device__ unsigned int syncRead(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr & src_ptr);

    // __device__ unsigned int syncWrite(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBufPtr & src_ptr);

    template <typename T>
    __device__ AgileArrReadWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> getArrayReadWrap(AgileBufPtr &buf, AgileLockChain &chain);

    // template <typename T>
    // __device__ AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> getArrayWriteWrap(AgileBufPtr &buf, AgileLockChain &chain, bool withRead);

    template <typename T>
    __device__ AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> getArraySharedWrap(AgileBufPtr &buf, AgileLockChain &chain, bool read, bool write);

    template <typename T>
    __device__ void updateWrite(AgileArrSharedWrap<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> & array);

    template <typename T>
    __device__ AgileTableBuf<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> getAgileTableBuf(AgileBuf & buf, AgileLockChain & chain);

    template <typename T>
    __device__ AgileArrayWarp<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> getArrayWrap(AgileLockChain & chain);

    __device__ ShareTableBase<ShareTableImpl> * getTable();

    __device__ unsigned int warpCoalesceAcquireGPUCache(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int &mask, unsigned int &lane_id, unsigned int &master_id);

    __device__ void warpCoalesceReleaseGPUCache(unsigned int gpu_cache_idx, unsigned int mask, unsigned int lane_id, unsigned int master_id);

    template <typename T>
    __device__ T readCacheElement(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int idx, AgileLockChain * chain);

    template <typename T>
    __device__ void loadArrayFromCache(NVME_DEV_IDX_TYPE dev_idx, unsigned long start_idx, T * data, unsigned int size);

    template <typename T>
    __device__ void accumulateArrayFromCache(NVME_DEV_IDX_TYPE dev_idx, unsigned long start_idx, T * data, unsigned int size);

    __device__ GPUCacheBase<GPUCacheImpl> * getGPUCacheBasePtr();

    __device__ CPUCacheBase<CPUCacheImpl> * getCPUCacheBasePtr();

};




// #define START_AIGLE(ctrl) __shared__ unsigned int AGILE_BID; if (ctrl->startAgile(AGILE_BID)) {
// #define END_AIGLE(ctrl) ctrl->stopAgile(AGILE_BID);}  



#include "agile_ctrl.tpp"

#endif