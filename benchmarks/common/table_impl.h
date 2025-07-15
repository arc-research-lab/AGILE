#include "agile_share_table.h"
#include "agile_cache_hierarchy.h"
#include "agile_ctrl.h"

class DisableShareTable : public ShareTableBase<DisableShareTable> {
public:
    
    __host__ DisableShareTable(unsigned int table_size) : ShareTableBase <DisableShareTable> (table_size) {
    }

    __device__ unsigned int hashing(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return 0;
    }

    __device__ AgileBuf * checkTableHitAcquireLockImpl_lockStart(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileLockChain * chain){
        
        return nullptr;
    }

    __device__ void appendBufImpl_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf, AgileLockChain * chain){
    }

    __device__ void removeBufImpl_inLockArea(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf * buf, AgileLockChain * chain){
    }

    __device__ void releaseTableLockImpl_lockEnd(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileLockChain * chain){
    }
};



class SimpleShareTable : public ShareTableBase<SimpleShareTable> {
public:

    __host__ SimpleShareTable(unsigned int table_size) : ShareTableBase <SimpleShareTable> (table_size) {
        printf("table_size: %d\n", table_size);
    }

    __device__ unsigned int checkTableAppendImpl(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx, AgileBuf** res, AgileBuf * candidate, AgileLockChain * chain){
        unsigned int table_id = this->hashing(ssd_dev_idx, ssd_blk_idx);
        this->acquireBaseLock_loadStart(table_id, chain);
        bool found = false;
        if(this->table[table_id] == nullptr){ // if the table is null
            *res = candidate;
            this->table[table_id] = candidate;
            candidate->setTag(ssd_dev_idx, ssd_blk_idx);
            found = true;
            __threadfence_system();
            this->releaseBaseLock_lockEnd(table_id, chain);
            return table_id;
        } 
        // if the table is not null
        AgileBuf * bufptr = this->table[table_id];
        AgileBuf * prev = nullptr;
        while(bufptr != nullptr){ // find the buf
            GPU_ASSERT(bufptr != candidate, "append buf error");
            if(bufptr->tag_ssd_dev == ssd_dev_idx && bufptr->tag_ssd_blk == ssd_blk_idx){
                *res = bufptr;
                found = true;
                break;
            }
            LOGGING(atomicAdd(&(logger->deadlock_check), 1));
            prev = bufptr;
            bufptr = bufptr->share_table_next;
        }
        if(!found){
            GPU_ASSERT (prev != nullptr, "append buf error");
            prev->share_table_next = candidate;
            candidate->setTag(ssd_dev_idx, ssd_blk_idx);
            *res = candidate;
            found = true;
            __threadfence_system();
            this->releaseBaseLock_lockEnd(table_id, chain);
            return table_id;
        }
        __threadfence_system();
        this->releaseBaseLock_lockEnd(table_id, chain);
        GPU_ASSERT(found, "append buf error");
        return -1;
    }

    __device__ bool removeBufImpl(unsigned int table_id, AgileBuf * buf, AgileLockChain * chain){
        removeBufImpl_lock_start:
        this->acquireBaseLock_loadStart(table_id, chain);
        unsigned int count = 0;
        while(atomicAdd(&(buf->reference), 0) != 0){ // wait for the reference count to be 0
            this->releaseBaseLock_lockEnd(table_id, chain);
            busyWait(1000);
            count++;
            if(count > 10000){
                LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                count = 0;
            }
            goto removeBufImpl_lock_start;
        }
        GPU_ASSERT(this->table[table_id] != nullptr, "write buffer table nullptr");
        bool found = false;
        AgileBuf * buf_ptr = this->table[table_id];
        if(buf_ptr == buf){
            this->table[table_id] = this->table[table_id]->share_table_next;
            found = true;
        }else{
            while(buf_ptr->share_table_next != nullptr){
                if(buf_ptr->share_table_next == buf){
                    found = true;
                    break;
                }
                LOGGING(atomicAdd(&(logger->deadlock_check), 1));
                buf_ptr = buf_ptr->share_table_next;
            }
            if (found) {
                GPU_ASSERT(buf_ptr->share_table_next == buf, "remove buf error");
                buf_ptr->share_table_next = buf_ptr->share_table_next->share_table_next;
                buf->share_table_next = nullptr;
                buf->tag_ssd_dev = -1;
                buf->tag_ssd_blk = -1;
                buf->status = AGILE_BUF_EMPTY;
                buf->reference = 0;
            }
        }
        GPU_ASSERT(found, "remove buf error here");
        __threadfence_system();
        this->releaseBaseLock_lockEnd(table_id, chain);
        return found;
    }

    __device__ unsigned int hashing(NVME_DEV_IDX_TYPE ssd_dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return ssd_blk_idx % table_size;
    }

};