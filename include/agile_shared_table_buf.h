#ifndef AGILE_TABLE_BUF
#define AGILE_TABLE_BUF
#include "agile_helpper.h"
#include "agile_buf.h"
// #include "agile_ctrl.h"

// #define AGILE_BUF_EMPTY  0
// #define AGILE_BUF_READY  1
// #define AGILE_BUF_MODIFIED 2
// #define AGILE_BUF_AWEAK 3
// #define AGILE_BUF_PROPAGATED 4
// #define AGILE_BUF_PROCESSING_READ 5

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileTableBuf{

    AgileBuf * self_buf;
    AgileBuf * shared_buf_ptr;
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl;
    AgileLockChain * chain;
    unsigned int table_idx;

public:
    __device__ ~AgileTableBuf(){
        shared_buf_ptr = nullptr;
        self_buf = nullptr;
    }

    __device__ AgileTableBuf(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl, AgileBuf * buf, AgileLockChain & chain){
        this->self_buf = buf;
        this->shared_buf_ptr = nullptr;
        this->self_buf->reference = 0;
        this->chain = &chain;
        this->ctrl = ctrl;
        this->table_idx = -1;
        this->self_buf->tag_ssd_dev = -1;
        this->self_buf->tag_ssd_blk = -1;
        this->self_buf->status = AGILE_BUF_EMPTY;
        this->self_buf->cache_next = nullptr;
        this->self_buf->share_table_next = nullptr;
    }

    __device__ void setSharedBufPtr(AgileBuf * shared_buf){
        GPU_ASSERT(this->shared_buf_ptr == nullptr, "shared_buf_ptr is not nullptr");
        this->shared_buf_ptr = shared_buf;
        this->incReference();
    }

    __device__ void incReference(){
        atomicAdd(&(this->shared_buf_ptr->reference), 1);
    }

    __device__ void decReference(){
        atomicSub(&(this->shared_buf_ptr->reference), 1);
    }

    __device__ void setTag(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){
        GPU_ASSERT(this->shared_buf_ptr->tag_ssd_dev == -1, "shared_buf_ptr is nullptr");
        GPU_ASSERT(this->shared_buf_ptr->tag_ssd_blk == -1, "shared_buf_ptr is nullptr");
        this->shared_buf_ptr->tag_ssd_dev = dev_idx;
        this->shared_buf_ptr->tag_ssd_blk = ssd_blk_idx;
    }

    __device__ bool hit(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return this->shared_buf_ptr->tag_ssd_dev == dev_idx && this->shared_buf_ptr->tag_ssd_blk == ssd_blk_idx;
    }

    __device__ void checkTable(unsigned int req_dev, unsigned long req_idx){
        // GPU_ASSERT(this->shared_buf_ptr == nullptr, "shared_buf_ptr is nullptr");
        GPU_ASSERT(this->table_idx == -1, "table_idx is not -1");
        this->shared_buf_ptr = nullptr;
        AgileBuf * temp_ptr = nullptr;
        this->table_idx = this->ctrl->getTable()->checkTableAppend(req_dev, req_idx, &temp_ptr, this->self_buf, chain);
        this->setSharedBufPtr(temp_ptr);
        if(this->self_buf == temp_ptr){
            LOGGING(atomicAdd(&(logger->push_in_table), 1));
            GPU_ASSERT(this->table_idx != -1, "table_idx is -1");
            AgileBufPtr bufPtr(self_buf);
            this->ctrl->asyncRead(req_dev, req_idx, &bufPtr, chain);
        }
        this->shared_buf_ptr->wait();
        // if(temp_ptr != nullptr){ // find others 
        //     this->setSharedBufPtr(temp_ptr);
        // }else{ // set self
        //     this->setSharedBufPtr(self_buf);
        //     AgileBufPtr bufPtr(self_buf);
        //     this->ctrl->getTable()->appendBuf_inLockArea(req_dev, req_idx, self_buf, chain);
        //     this->ctrl->asyncRead(req_dev, req_idx, &bufPtr, chain);
        // }
        // this->ctrl->getTable()->releaseTableLock_lockEnd(req_dev, req_idx, chain);
        
    }

    __device__ void reset(){
        if(this->shared_buf_ptr == nullptr){
            self_buf->tag_ssd_dev = -1;
            self_buf->tag_ssd_blk = -1;
            self_buf->status = AGILE_BUF_EMPTY;
            self_buf->cache_next = nullptr;
            self_buf->share_table_next = nullptr;
            this->self_buf->reference = 0;
        } else {
            this->decReference();
            if(this->table_idx != -1){
                GPU_ASSERT(this->self_buf == this->shared_buf_ptr, "self_buf is not shared_buf_ptr");
                bool found = this->ctrl->getTable()->removeBuf(this->table_idx, this->self_buf, chain);
                LOGGING(atomicAdd(&(logger->pop_in_table), 1));
                GPU_ASSERT(found, "remove buf error");
                this->table_idx = -1;
                this->self_buf->share_table_next = nullptr;
            }else{
                this->shared_buf_ptr = nullptr;
                this->self_buf->tag_ssd_dev = -1;
                this->self_buf->tag_ssd_blk = -1;
                this->self_buf->status = AGILE_BUF_EMPTY;
                this->self_buf->cache_next = nullptr;
                this->self_buf->share_table_next = nullptr;
                this->self_buf->reference = 0;
            }
        }
        GPU_ASSERT(this->table_idx == -1, "table_idx is not -1");
    }

    __device__ void wait(){
        this->shared_buf_ptr->wait();
    }

    __device__ void stop(){
        this->reset();
    }

    __device__ T get(unsigned int req_dev, unsigned long req_idx){
        unsigned int req_blk = req_idx / (this->ctrl->buf_size / sizeof(T));
        if(this->shared_buf_ptr != nullptr){
            if(this->hit(req_dev, req_blk)){
                return static_cast<T*>(this->shared_buf_ptr->getDataPtr())[req_idx % (this->ctrl->buf_size / sizeof(T))];
            } 
        } 
        this->reset();
        this->checkTable(req_dev, req_blk);
        return static_cast<T*>(this->shared_buf_ptr->getDataPtr())[req_idx % (this->ctrl->buf_size / sizeof(T))];

    }
};
#endif