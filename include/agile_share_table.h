#ifndef AGILE_WRITE_TABLE
#define AGILE_WRITE_TABLE
#include "malloc.h"
#include "agile_buf.h"
#include "agile_lock.h"

class ShareTableBase_T {

public:
    unsigned int table_size;
    AgileBuf ** table;
    AgileLock * table_locks;

    __host__ ShareTableBase_T(unsigned int table_size){
        this->table_size = table_size;
        if(table_size != 0){
            AgileBuf ** h_table = (AgileBuf **) malloc(sizeof(AgileBuf *) * table_size);
            cuda_err_chk(cudaMalloc(&table, sizeof(AgileBuf *) * table_size));
            cuda_err_chk(cudaMalloc(&table_locks, sizeof(AgileLock) * table_size));
            AgileLock * h_locks = (AgileLock *) malloc(sizeof(AgileLock) * table_size);
#if LOCK_DEBUG
            char lockName[20];
#endif
            for(unsigned int i = 0; i < table_size; ++i){
#if LOCK_DEBUG
                sprintf(lockName, "write table l%d", i);
                h_locks[i] = AgileLock(lockName);
#else   
                h_locks[i] = AgileLock();
#endif
                h_table[i] = nullptr;
            }
            cuda_err_chk(cudaMemcpy(table, h_table, sizeof(AgileBuf *) * table_size, cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->table_locks, h_locks, sizeof(AgileLock) * table_size, cudaMemcpyHostToDevice));
            free(h_locks);
        }
    }

    __device__ void acquireBaseLock_loadStart(unsigned int idx, AgileLockChain * chain){
        table_locks[idx].acquire(chain);
    }

    __device__ void releaseBaseLock_lockEnd(unsigned int idx, AgileLockChain * chain){
        table_locks[idx].release(chain);
    }
    
};

template<typename ShareTableImpl>
class ShareTableBase : public ShareTableBase_T {
public:
    __host__ ShareTableBase(unsigned int table_size) : ShareTableBase_T(table_size) {
        // printf("ShareTableBase %d\n", this->table_size);
    }

    // __device__ bool checkTableHitAcquireLock_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf** buf, AgileLockChain * chain){
    //     return static_cast<ShareTableImpl *>(this)->checkTableHitAcquireLockImpl_lockStart(ssd_dev_idx, ssd_blk_idx, buf, chain);
    // }

    // __device__ unsigned int appendBuf_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf, AgileLockChain * chain){
    //     return static_cast<ShareTableImpl *>(this)->appendBufImpl_inLockArea(ssd_dev_idx, ssd_blk_idx, buf, chain);
    // }

    // __device__ void removeBuf_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf, AgileLockChain * chain, unsigned int line){
    //     static_cast<ShareTableImpl *>(this)->removeBufImpl_inLockArea(ssd_dev_idx, ssd_blk_idx, buf, chain, line);
    // }

    // __device__ void releaseTableLock_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileLockChain * chain){
    //     __threadfence_system();
    //     static_cast<ShareTableImpl *>(this)->releaseTableLockImpl_lockEnd(ssd_dev_idx, ssd_blk_idx, chain);
    // }


    __device__ unsigned int checkTableAppend(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf** res, AgileBuf * candidate, AgileLockChain * chain){
        return static_cast<ShareTableImpl *>(this)->checkTableAppendImpl(ssd_dev_idx, ssd_blk_idx, res, candidate, chain);
    }

    __device__ bool removeBuf(unsigned int table_id, AgileBuf * buf, AgileLockChain * chain){
        return static_cast<ShareTableImpl *>(this)->removeBufImpl(table_id, buf, chain);
    }
};

#endif