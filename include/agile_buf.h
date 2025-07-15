#ifndef AGILE_BUF
#define AGILE_BUF
#include "agile_helpper.h"
// #include "agile_ctrl.h"

#define AGILE_BUF_EMPTY  0
#define AGILE_BUF_READY  1
#define AGILE_BUF_MODIFIED 2
#define AGILE_BUF_AWEAK 3
#define AGILE_BUF_PROPAGATED 4
#define AGILE_BUF_PROCESSING_READ 5
// #define AGILE_BUF_PROCESSING_READ 3
// #define AGILE_BUF_PROCESSING_WRITE 4

class AgileBuf {

private:
    
    
public:

// __device__ AgileBuf(){
//     status = AGILE_BUF_EMPTY;
// }
    void * data;
    AgileBuf * cache_next;
    AgileBuf * share_table_next;
    unsigned int buf_size;
    unsigned int status; // TODO add reference number to status;
    TID_TYPE reference;
    NVME_DEV_IDX_TYPE tag_ssd_dev;
    SSDBLK_TYPE tag_ssd_blk;

    // unsigned int in_table;

    // __host__ AgileBuf(void * data){
    //     this->data = data;
    //     this->reference = 0;
    //     this->status = AGILE_BUF_EMPTY;
    //     this->cache_next = nullptr;
    //     this->table_next = nullptr;
    //     this->tag_ssd_dev = -1;
    //     this->tag_ssd_blk = -1;
    // }

    __host__ AgileBuf(unsigned int buf_size){

        cuda_err_chk(cudaMalloc(&data, buf_size));
        this->buf_size = buf_size;
        this->reference = 0;
        this->status = AGILE_BUF_EMPTY;
        this->cache_next = nullptr;
        this->share_table_next = nullptr;
        this->tag_ssd_dev = -1;
        this->tag_ssd_blk = -1;
        // this->in_table = 0;
    }

    __device__ ~AgileBuf(){}

    __device__ void * getDataPtr(){
        return this->data;
    }

    __device__ void incReference(){
        atomicAdd(&(this->reference), 1);
    }

    __device__ void decReference(){
        atomicSub(&(this->reference), 1);
    }

    // __device__ void waitReference(){
    //     unsigned int counter = 0;
    //     while(atomicAdd(&(this->reference), 0) != 0){
    //         busyWait(1000);
    //         counter++;
    //         if(counter == 10000){
    //             counter = 0;
    //             printf("warnning: wait other threads release buffer for too many times %p\n", data);
    //         }
    //     }
    // }

    __device__ bool checkHit(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE blk_idx){
        return dev_idx == tag_ssd_dev && blk_idx == tag_ssd_blk;
    }

    __device__ void setTag(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE blk_idx){
        // this->tag_ssd_dev = dev_idx;
        // this->tag_ssd_blk = blk_idx;
        atomicExch(&(this->tag_ssd_dev), dev_idx);
        atomicExch(&(this->tag_ssd_blk), blk_idx);

    }

    __device__ void resetStatus(){
        // this->table_next = nullptr;
        
        atomicExch(&(this->status), AGILE_BUF_EMPTY);

        // printf("reset buf %p %d\n", this, this->status);
        atomicExch(&(this->tag_ssd_dev), -1);
        atomicExch(&(this->tag_ssd_blk), (unsigned long) -1);
    }

    __device__ void setModified() {
        atomicExch(&(this->status), AGILE_BUF_MODIFIED);
    }

    __device__ void ready(){
        unsigned int stat = atomicAdd(&(this->status), 0);
        if(stat != AGILE_BUF_PROCESSING_READ){
            printf("ready: %d %p %d\n", this->status, this, threadIdx.x);
        }
        // GPU_ASSERT(stat == AGILE_BUF_PROCESSING_READ, "set ready error");
        // unsigned int count = 0;
        // while(atomicCAS(&(this->status), AGILE_BUF_PROCESSING_READ, AGILE_BUF_READY)){
        //     count++;
        //     if(count > 10000){
        //         count = 0;
        //         printf("status error ready: %p %d bid: %d tid: %d\n", this, this->status, blockIdx.x, threadIdx.x);
        //     }
        // }

        atomicExch(&(this->status), AGILE_BUF_READY);
        LOGGING(atomicAdd(&(logger->finish_buffer), 1));
    }

    __device__ void setProcessingRead(){
        unsigned int stat = atomicAdd(&(this->status), 0);
        // if(stat != AGILE_BUF_EMPTY){
            // printf("setProcessingRead: %p %d\n", this, stat);
        // }
        // printf("set processing %p %d\n", this, threadIdx.x);
        GPU_ASSERT(stat == AGILE_BUF_EMPTY, "setProcessingRead error");
        
        atomicExch(&(this->status), AGILE_BUF_PROCESSING_READ);
    }


    __device__ void moveData(void * src_ptr, void * dst_ptr){
        uint4 * src = reinterpret_cast<uint4*>(src_ptr);
        uint4 * dst = reinterpret_cast<uint4*>(dst_ptr);
        for(unsigned int i = 0; i < this->buf_size / sizeof(uint4); ++i){
            dst[i] =  src[i];
        }
    }


    __device__ void propagateData(){
        AgileBuf * next = this->cache_next;
        AgileBuf * tmp = nullptr;
        propagateData_start:
        if(next != nullptr){
            LOGGING(atomicAdd(&(logger->self_propagate), 1));
            this->moveData(this->data, next->data);
            tmp = next->cache_next;
            next->ready();
            if(tmp != nullptr){
                if(atomicCAS(&(next->status), AGILE_BUF_READY, AGILE_BUF_PROPAGATED) == AGILE_BUF_READY){
                    // next->cache_next = nullptr;
                    next = tmp;
                    goto propagateData_start;
                }else{
                    next == nullptr; // let other threads to propagate the raset of the chain
                }
            }
        }
    }

    __device__ void wait(){
        unsigned int counter = 0;
        while(atomicAdd(&(this->status), 0) != AGILE_BUF_READY){
            counter++;
            if(counter == 10000){
                counter = 0;
                LOGGING(atomicAdd(&(logger->waitTooMany), 1));
            }
        }
        // printf("wait: %p %d\n", this, this->status);
    }

};
  
// used in local threads, do not share
class AgileBufPtr {
public:
    AgileBuf * buf; // point to the buffer belong to the local thread
    // AgileBuf * shared_buf; // point to the shared buffer in other threads or this thread. writing to the same place to avoid granularity missmatch
    AgileBuf * shared_buf_ptr;
    unsigned int self;
    __host__ __device__ inline AgileBufPtr(){
        this->buf = nullptr;
        this->shared_buf_ptr = nullptr;
        self = 0;
    }


    __device__ AgileBufPtr(AgileBuf & buf){
        this->buf = &buf;
        this->shared_buf_ptr = nullptr;
    }

    __device__ AgileBufPtr(AgileBuf * buf){
        this->buf = buf;
        this->shared_buf_ptr = nullptr;
    }

    __device__ void wait(){
        this->buf->wait();
    }

    __device__ ~AgileBufPtr(){
    }

    __device__ void setProcessingRead(){
        this->buf->setProcessingRead();
    }

    __device__ void setSharedBufPtr(AgileBuf * shared_buf){
        this->shared_buf_ptr = shared_buf;
    }

    // __device__ void setSharedPtr(AgileBuf * table_buf){
    //     // this->shared_buf = table_buf;
    // }

    // __device__ void setModified() {
    //     atomicExch(&(this->shared_buf->status), AGILE_BUF_MODIFIED);
    // }

    __device__ void setReadTag(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE blk_idx){
        this->buf->setTag(dev_idx, blk_idx);
    }

    __device__ void resetStatus(){
        this->buf->resetStatus();
        this->shared_buf_ptr = nullptr;
        // this->w_buf == nullptr;
    }

    // __device__ void reset(){
    //     this->buf->resetStatus();
    //     this->shared_buf_ptr = nullptr;
    //     // this->shared_buf == nullptr;
    // }

    __device__ void ready(){
        this->buf->ready();
    }

    // __device__ bool checkHit(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE blk_idx){
    //     return buf->checkHit(dev_idx, blk_idx);
    // }

    __device__ void* getDataPtr(){
        return this->buf->data;
    }

    __device__ void waitRead(){
        this->buf->wait();
    }

    // __device__ void waitShared(){
    //     this->shared_buf->wait();
    // }

    __device__ bool readEmpty(){
        return atomicAdd(&(this->buf->status), 0) == AGILE_BUF_EMPTY;
    }

    // __device__ bool sharedEmpty(){
    //     return atomicAdd(&(this->shared_buf->status), 0) == AGILE_BUF_EMPTY;
    // }

    __device__ bool readHit(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return this->buf->tag_ssd_dev == dev_idx && this->buf->tag_ssd_blk == ssd_blk_idx;
    }

    __device__ bool sharedBufHit(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){
        return (this->shared_buf_ptr != nullptr) && (atomicAdd(&(this->shared_buf_ptr->tag_ssd_dev), 0) == dev_idx && atomicAdd(&(this->shared_buf_ptr->tag_ssd_blk), 0) == ssd_blk_idx);
    }

    __device__ void setSelfTag(NVME_DEV_IDX_TYPE dev_idx, SSDBLK_TYPE ssd_blk_idx){
        // if(!(this->buf->tag_ssd_dev == -1 && this->buf->tag_ssd_blk == -1)){
        //     printf("error: %d %d\n", this->buf->tag_ssd_dev, this->buf->tag_ssd_blk);
        // }
        GPU_ASSERT(this->buf->tag_ssd_dev == -1 && this->buf->tag_ssd_blk == -1, "setSelfTag error");
        this->buf->tag_ssd_dev = dev_idx;
        this->buf->tag_ssd_blk = ssd_blk_idx;
    }
};

#endif