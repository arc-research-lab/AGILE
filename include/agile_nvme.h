#ifndef AGILE_NVME
#define AGILE_NVME

#include "agile_helpper.h"


#define AGILE_NVME_READ 0x2
#define AGILE_NVME_WRITE 0x1

#define CPU_DEVICE 0X1
#define GPU_DEVICE 0X0


typedef struct {
    unsigned int dword[16];
} nvm_cmd_t;

class AgileCQ {

public:

    volatile unsigned int * data;
    volatile unsigned int * cqdb;
    unsigned int pos;
    unsigned int phase;
    unsigned int depth;

    // used in a warp
    unsigned int pos_offset;
    unsigned int mask;

#if DEBUG_NVME
    unsigned int qid;

__host__ AgileCQ(void * data, volatile unsigned int * cqdb, unsigned int depth, unsigned int qid){
    this->pos = 0;
    this->phase = 0;
    this->depth = depth;
    this->data = (volatile unsigned int *) data;
    this->cqdb = cqdb;
    this->qid = qid;
    this->pos_offset = 0;
    this->mask = 0;
}

#else

__host__ AgileCQ(void * data, void * cqdb, unsigned int depth){
    this->pos = 0;
    this->phase = 0;
    this->depth = depth;
    this->data = (volatile unsigned int *) data;
    this->cqdb = (volatile unsigned int *) cqdb;
}

#endif

};

#define AGILE_CMD_STATUS_EMPTY 0
#define AGILE_CMD_STATUS_PROCESSING 1
#define AGILE_CMD_STATUS_READY 2
#define AGILE_CMD_STATUS_ISSUSING 3
#define AGILE_CMD_STATUS_ISSUED 4

__inline__
__device__ void wati_status(unsigned int * cmd_status, unsigned int expected, unsigned int target){
    while(atomicCAS(cmd_status, expected, target) != expected){
        LOGGING(atomicAdd(&(logger->waitTooMany), 1));
    }
}

class AgileSQ  {

public:
    volatile unsigned int * sqdb;
    volatile unsigned int * data;
    unsigned int * cmd_status; // 0 empty or issued, 1 not issued
    unsigned int ssd_blk_offset; // calculate by host, multiple of 512
    
    AgileLock * cmd_locks;
    AgileLock * sqdb_lock;

    // unsigned int issued_count;

    unsigned int pos;
    unsigned int depth;

#if DEBUG_NVME
    unsigned int g_pos;
    unsigned int fake_sqdb;
    unsigned int prev_sq_pos;
    unsigned int qid;
    
__host__ AgileSQ(void * data, volatile unsigned int * sqdb, unsigned int depth, unsigned int qid, unsigned int ssd_blk_offset){
    this->pos = 0;
    this->g_pos = 0;
    this->fake_sqdb = 0;
    this->prev_sq_pos = 0;
    // this->issued_count = 0;
    this->depth = depth;
    this->qid = qid;
    this->data = (volatile unsigned int *) data;
    this->sqdb = sqdb;
    this->ssd_blk_offset = ssd_blk_offset;

    cuda_err_chk(cudaMalloc(&(this->cmd_status), sizeof(unsigned int) * depth));

    cuda_err_chk(cudaMalloc(&(this->cmd_locks), sizeof(AgileLock) * depth));
    cuda_err_chk(cudaMalloc(&(this->sqdb_lock), sizeof(AgileLock) * 1));

    AgileLock * h_cmd_locks = (AgileLock *) malloc(sizeof(AgileLock) * depth);
    AgileLock * h_sqdb_lock = (AgileLock *) malloc(sizeof(AgileLock) * 1);

    
#if LOCK_DEBUG
    char lockName[20];
    sprintf(lockName, "sqdb(%d)", qid);
    h_sqdb_lock[0] = AgileLock(lockName);
#else
    h_sqdb_lock[0] = AgileLock();
#endif
    
    for(int i = 0; i < depth; ++i){
#if LOCK_DEBUG
        sprintf(lockName, "q(%d)cmdlock(%d)", qid, i);
        h_cmd_locks[i] = AgileLock(lockName);
#else
        h_cmd_locks[i] = AgileLock();
#endif

    }

    cuda_err_chk(cudaMemcpy(this->cmd_locks, h_cmd_locks, sizeof(AgileLock) * depth, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(this->sqdb_lock, h_sqdb_lock, sizeof(AgileLock) * 1, cudaMemcpyHostToDevice));
}
#else

    __host__ AgileSQ(void * data, volatile unsigned int * sqdb, unsigned int depth){
        this->pos = 0;
        this->issued_count = 0;
        this->depth = depth;
        this->data = (volatile unsigned int *) data;
        this->sqdb = sqdb;

        cuda_err_chk(cudaMalloc(&(this->cmd_status), sizeof(unsigned int) * depth));
        cuda_err_chk(cudaMemset(this->cmd_status, 0, sizeof(unsigned int) * depth));

        cuda_err_chk(cudaMalloc(&(this->cmd_locks), sizeof(AgileLock) * depth));
        cuda_err_chk(cudaMalloc(&(this->sqdb_lock), sizeof(AgileLock) * 1));

        AgileLock * h_cmd_locks = (AgileLock *) malloc(sizeof(AgileLock) * depth);
        AgileLock * h_sqdb_lock = (AgileLock *) malloc(sizeof(AgileLock) * 1);

        h_sqdb_lock[0] = AgileLock();
        for(int i = 0; i < depth; ++i){
            h_cmd_locks[i] = AgileLock();
        }

        cuda_err_chk(cudaMemcpy(this->cmd_locks, h_cmd_locks, sizeof(AgileLock) * depth, cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->sqdb_lock, h_sqdb_lock, sizeof(AgileLock) * 1, cudaMemcpyHostToDevice));
    }

#endif

    __host__ ~AgileSQ(){
        // printf("~AgileSQ()\n");
        // cuda_err_chk(cudaFree(this->cmd_locks));
        // cuda_err_chk(cudaFree(this->sqdb_lock));
        // cuda_err_chk(cudaFree(this->cmd_status));
    }

    __device__ void attemptSQDB(AgileLockChain * chain){
        // printf("attemptSQDB\n");
        
        if(this->sqdb_lock->try_acquire(chain)){
            
            // printf("issueRead\n");
            unsigned int count = 0;
            unsigned int fake_sqdb0 = atomicAdd(&(this->fake_sqdb), 0);
            // unsigned int issued = atomicAdd(&(this->issued_count), 0);
            unsigned int old_pos = fake_sqdb0 % this->depth;
            unsigned int status = 0;
            do{
                status = atomicCAS(&(this->cmd_status[old_pos]), AGILE_CMD_STATUS_READY, AGILE_CMD_STATUS_ISSUSING);
                if(status == AGILE_CMD_STATUS_READY){
                    old_pos = (old_pos + 1) % this->depth;
                    count++;
                }
                // LOGGING(atomicAdd(&(logger->deadlock_check), 1));
            }while(status == AGILE_CMD_STATUS_READY && count < max(this->depth / 16, 1));
            
            if(count != 0){
#if FAKE_NVME
#else
                *sqdb = old_pos;
                __threadfence_system();
                // unsigned int last_sq_pos = atomicAdd(&(logger->curr_sq_pos[qid]), 0);
                // atomicExch(&(logger->curr_sq_pos[qid]), old_pos);
                // atomicExch(&(logger->last_sq_pos[qid]), last_sq_pos);
#endif

                atomicAdd(&(this->fake_sqdb), count);
            }else{
                // printf("sq no update %d %d\n", qid, issued);
            }
            __threadfence_system();
            this->sqdb_lock->release(chain);
        }else{
            
        }
    }

    __device__ bool attemptEnqueue(unsigned int cmd_type, unsigned int table_idx, unsigned int device_type, SSDBLK_TYPE ssd_blk_idx, unsigned int blocks, unsigned long phy_addr, AgileLockChain * chain){
        unsigned int cmd_pos_g = atomicAdd(&(this->pos), 0);
        unsigned int pref_pos_g = atomicAdd(&(this->prev_sq_pos), 0);
        if(cmd_pos_g - pref_pos_g > this->depth - 32){
            return false;
        }
        unsigned int cmd_pos = cmd_pos_g % this->depth;
        bool succ = false;
        unsigned int val = 0;
        
        if(cmd_locks[cmd_pos].try_acquire(chain, false)){ // release lock remotely

            unsigned int pos_g = atomicAdd(&(this->prev_sq_pos), 0);
            unsigned int sqpos_g = atomicAdd(&(this->g_pos), 0);
            unsigned int gap = sqpos_g - pos_g;
            if(gap > this->depth - 32){
                // printf("gap too large %d %d %d\n", qid, pos_g, sqpos_g);
                cmd_locks[cmd_pos].remoteRelease();
                return false;
            }

            atomicAdd(&(this->pos), 1);

            wati_status(&(this->cmd_status[cmd_pos]), AGILE_CMD_STATUS_EMPTY, AGILE_CMD_STATUS_PROCESSING);
            // while(atomicAdd(&(this->cmd_status[cmd_pos]), 0) != 0){} // ensure this command is notified since cmd_locks[cmd_pos] can be release very quickly in the emulation
            
            if(cmd_type == AGILE_NVME_READ){
                LOGGING(atomicAdd(&(logger->issued_read), 1)); // TODO add write
            }else if(cmd_type == AGILE_NVME_WRITE) {
                LOGGING(atomicAdd(&(logger->issued_write), 1)); // TODO add write
            }

            succ = true;
            volatile unsigned int * cmd_ptr = this->data + cmd_pos * 16; 
            // TODO add CPU or GPU info in reserved area
            cmd_ptr[0] = (cmd_type & 0x7f) | (cmd_pos << 16); // TODO check if cid can be same with cmd_pos
            cmd_ptr[1] = 1; // namespace
            cmd_ptr[2] = 0; 
            cmd_ptr[3] = 0; // store cpu table info to reserved place for both read and write
            cmd_ptr[4] = 0; 
            cmd_ptr[5] = 0; 
            cmd_ptr[6] = (phy_addr & 0xffffffff); 
            cmd_ptr[7] = ((phy_addr >> 32) & 0xffffffff); 
            cmd_ptr[8] = 0; 
            cmd_ptr[9] = 0; 
            cmd_ptr[10] = ssd_blk_idx + this->ssd_blk_offset; 
            cmd_ptr[11] = 0;
            cmd_ptr[12] = blocks - 1;
            cmd_ptr[13] = 0; // ((device_type & 0x1) << 8); // store device info in the reserved place [15:8] reserved in write [31:8] reserved in read
            cmd_ptr[14] = 0;
            cmd_ptr[15] = 0;
            // printf("cmd_type, %d blk, %ld \n", cmd_type, ssd_blk_idx);
            __threadfence_system(); // wait for the command write to global mem success.
            
            // atomicExch(&(this->cmd_status[cmd_pos]), 1); // mark this slot ready
            wati_status(&(this->cmd_status[cmd_pos]), AGILE_CMD_STATUS_PROCESSING, AGILE_CMD_STATUS_READY);
            
            // __threadfence_system(); // TODO remove this,
            unsigned int counter = 0;
            // printf("enqueue, qid:, %d, pos:, %d, id:, %d\n", qid, cmd_pos, cmd_pos_g);
            AgileSQ_attemptEnqueue_attemptSQDB:
            this->attemptSQDB(chain); // TODO check here
            // unsigned int issued_new = atomicAdd(&(this->issued_count), 0);
            // if((((int)issued_new) - ((int)cmd_pos_g)) < 0){ // check if this command has been notified to ssd
            // if(issued_new < cmd_pos_g){
            if(atomicAdd(&(this->cmd_status[cmd_pos]), 0) != AGILE_CMD_STATUS_ISSUSING){
                // printf("here q %d cmd_pos: %d status: %d\n", qid, cmd_pos, temp);
                // GPU_ASSERT(temp != -1 && temp != -2, "status err");
                busyWait(1000);
                counter++;
                if(counter == 100000){
                    counter = 0;
                    
                    // printf("AgileSQ_attemptEnqueue_attemptSQDB too many times %d %d\n", qid, cmd_pos_g);
                    LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                    
                }
                goto AgileSQ_attemptEnqueue_attemptSQDB;
            }else{
                // printf("success qid %d pos %d\n", qid, cmd_pos);
            }
            // atomicExch(&(this->cmd_status[cmd_pos]), AGILE_CMD_STATUS_ISSUED);
            
            atomicAdd(&(this->g_pos), 1);
            
            __threadfence_system();
            wati_status(&(this->cmd_status[cmd_pos]), AGILE_CMD_STATUS_ISSUSING, AGILE_CMD_STATUS_ISSUED);

        }

        return succ;
    }

};

/** 
* used in Agile service, no device information needed
*/
class AgileQueuePair {
    

    // __host__ AgileQueuePair(){

    // }
public:
    AgileSQ sq;
    AgileCQ cq;

#if DEBUG_NVME
    unsigned int qid;
    __host__ AgileQueuePair(void * sq_data, void * cq_data, volatile unsigned int * sq_db, volatile unsigned int * cq_db, unsigned int depth, unsigned int qid, unsigned int ssd_blk_offset) :
    sq(sq_data, sq_db, depth, qid, ssd_blk_offset), cq(cq_data, cq_db, depth, qid), qid(qid){}
#else
    __host__ AgileQueuePair(void * sq_data, void * cq_data, volatile unsigned int * sq_db, volatile unsigned int * cq_db, unsigned int depth, unsigned int ssd_blk_offset) :
    sq(sq_data, sq_db, depth, ssd_blk_offset), cq(cq_data, cq_db, depth){}
#endif
};

class AgilePollingList {
public:
    AgileQueuePair * pairs;
    unsigned int num_pairs;
};

class AgileNvmeDev : public AGILE_Kernel_TEMP<AgileNvmeDev> {
    
public:

    AgileQueuePair * pairs;
    unsigned int queue_num;
    unsigned int queue_depth;

    // enum class CMD_TYPE : unsigned int {
    //     READ = 0,
    //     WRITE = 1
    // };

    __host__ AgileNvmeDev(unsigned int queue_num, unsigned int queue_depth) : queue_num(queue_num), queue_depth(queue_depth) {
        // pairs = (AgileQueuePair *) malloc(sizeof(AgileQueuePair) * queue_num);
    }

    // __host__ void addQueuePair(unsigned int idx, unsigned int * sq_data, unsigned int * cq_data, unsigned int * sq_db, unsigned int * cq_db){
    //     pairs[idx] = AgileQueuePair(sq_data, cq_data, sq_db, cq_db, queue_depth);
    // }

    __device__ void issueRead(unsigned int device_type, unsigned int table_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int blocks, unsigned long phy_addr, AgileLockChain * chain){
        // TODO fix multiple q
        unsigned int q_idx =  (threadIdx.x) % queue_num;
        unsigned int counter = 0;
        while(!this->pairs[q_idx].sq.attemptEnqueue(AGILE_NVME_READ, table_idx, device_type, ssd_blk_idx, blocks, phy_addr, chain)){
            q_idx = (q_idx + 1) % queue_num;
            counter++;
            if(counter == 100000){
                counter = 0;
                LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                // printf("attemptEnqueue too many times %d\n", q_idx);
            }
            LOGGING(atomicAdd(&(logger->attempt_fail), 1));
            // __nanosleep(10000);
            // busyWait(10000);
            // printf("attemptEnqueue %d\n", q_idx);
        }
    }

    __device__ void issueWrite(unsigned int device_type, unsigned int table_idx, SSDBLK_TYPE ssd_blk_idx, unsigned int blocks, unsigned long phy_addr, AgileLockChain * chain){
        // TODO fix multiple q
        unsigned int q_idx =  threadIdx.x % queue_num;
        while(!this->pairs[q_idx].sq.attemptEnqueue(AGILE_NVME_WRITE, table_idx, device_type, ssd_blk_idx, blocks, phy_addr, chain)){
            q_idx = (q_idx + 1) % queue_num;
            // __nanosleep(10000);
            LOGGING(atomicAdd(&(logger->waitTooMany), 1));
        }
    }


};


#endif