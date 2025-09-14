#ifndef AIGLE_LOCK
#define AIGLE_LOCK
#include "agile_helper.h"

#define DEBUG LOCK_DEBUG
class AgileLock;
/*
* Must point to a lock that will be locally released
*/
class AgileLockChain {
public:
    AgileLock * lock_ptr;
    __device__ AgileLockChain(){
        this->lock_ptr = nullptr;
    }
};

class AgileLock : public AGILE_Kernel_TEMP<AgileLock> {

public:
#if DEBUG

    /**
    * Parameter: link_lock
    * Description: link_lock will be used if this lock is the head lock
    */
    // unsigned int link_lock;

    /**
    * Parameter: link_lock_ptr
    * Description: point to the first lock's link_lock; It should point to self's link_lock by default;
    */
    // unsigned int *link_lock_ptr;

    // __device__ void lockLink(){
    //     // TODO: when release, this->link_lock_ptr will be set to nullptr, can cause illegal memory access
    //     if (link_lock_ptr != nullptr){
    //         while(atomicCAS(this->link_lock_ptr, 0, 1) != 0){ 
                
    //         }
    //     } else {
    //         while(atomicCAS(&(this->link_lock), 0, 1) != 0){
                
    //         }
    //     }
        
    // }

    // __device__ void releaseLink(){
    //     while(atomicCAS(this->link_lock_ptr, 1, 0) != 1);
    // }

#endif

    /**
    * Parameter: lock
    * Description: record the idx of the thread that has acquired the lock
    * Val:
        MAXVAL(-1): not acquired
        thread's idx: acquired
    */
    TID_TYPE lock;

#if DEBUG
    /**
    * Parameter: lock_dep
    * Description: if a thread has acquired this lock, and currently is waiting for another lock, this variable stores the thread idx that acquires the target lock;
    * Val:
        MAXVAL(-1): not in wating for another lock
        thread's idx: waiting for the thread
    */
    char lockname[20];
    TID_TYPE tid_dep;
    
    /**
    * Parameter: next
    * Description: enable listed link to update the status of all acquired locks
    */
    AgileLock * next;

    /**
    * Parameter: lock_dep
    * Description: enable listed link to update the status of all acquired locks
    */
    AgileLock * lock_dep;

    bool release_local;

#endif

public:
    
#if DEBUG
    __host__  AgileLock(const char name[20]){
        this->lock = TID_NONE;
        this->next = nullptr;
        this->lock_dep = nullptr;
        this->tid_dep = TID_NONE;
        // this->lockname = name;
        memcpy(this->lockname, name, 20);
    }   
#else
    __host__  AgileLock(){
        this->lock = TID_NONE;
    }
#endif
    

    // __host__ AgileLock* getDeviceLockPtr(){ // TODO: how to free this pointer
    //     AgileLock* d_lock;
    //     cuda_err_chk(cudaMalloc(&d_lock, sizeof(AgileLock)));
    //     cuda_err_chk(cudaMemcpy(d_lock, this, sizeof(AgileLock), cudaMemcpyHostToDevice));
    //     return d_lock;
    // }

    __host__ ~AgileLock(){
#if DEBUG
#endif
    }

    __device__ bool try_acquire(AgileLockChain * chain){
        return this->acquire(chain, true, true);
    }

    __device__ bool try_acquire(AgileLockChain * chain, bool local_release){
        bool res =  this->acquire(chain, local_release, true);
        // printf("try_acquire: %p\n", this);
        return res;
    }

    __device__ bool acquire(AgileLockChain * chain){
        return this->acquire(chain, true, false);
    }

    __device__ bool acquire(AgileLockChain * chain, bool local_release){
        return this->acquire(chain, local_release, false);
    }

    __device__ bool acquire(AgileLockChain * chain, bool local_release, bool non_block){

        // extern __shared__ unsigned int AGILE_BID;
        TID_TYPE tid = blockIdx.x * blockDim.x + threadIdx.x;

#if DEBUG
        unsigned int counter = 0;
        // GPU_ASSERT((chain->lock_ptr != nullptr) || (chain->lock_ptr == nullptr) == (local_release), "First lock must be a locally-release lock\n (Maybe Change)");
        // if(chain->lock_ptr != nullptr){
        //     chain->lock_ptr->lockLink();
        // } else {
        //     this->lockLink();
        // }
#endif
        AgileLock_acquire:
        TID_TYPE lock_val = TID_NONE;
        
        if ((lock_val = atomicCAS(&(this->lock), TID_NONE, tid)) == TID_NONE){ // acquired this lock
            
#if DEBUG
            this->release_local = local_release;
            this->tid_dep = TID_NONE;
            this->lock_dep = nullptr;


            if(local_release){ // add to the chain
                if(chain->lock_ptr == nullptr){
                    chain->lock_ptr = this;
                    this->next = this;
                    // this->link_lock_ptr = &(this->link_lock);
                    
                } else {

                    this->next = chain->lock_ptr->next;
                    chain->lock_ptr->next = this;
                    // this->link_lock_ptr = &(chain->lock_ptr->link_lock);
    
                    // remove previous dep
                    AgileLock * lock_ptr = this->next;
                    while(this != lock_ptr){
                        LOGGING(atomicAdd(&(logger->deadlock_check), 1));
                        lock_ptr->tid_dep = TID_NONE;
                        lock_ptr->lock_dep = nullptr;
                        lock_ptr = lock_ptr->next;
                    }
                }
            }
            
            __threadfence_system();
            
            // chain->lock_ptr->releaseLink();
#endif
            // printf("152 lock %p tid %d lock_val: %d\n", this, threadIdx.x, lock_val);
            return true;


        } else { // fail to acquire the lock
            
#if DEBUG
            if (chain->lock_ptr != nullptr){ // ignore if *chain is nullptr, because this means the thraed does not acquire any locks
                // chain->lock_ptr->releaseLink();
                // mark dependence
                chain->lock_ptr->tid_dep = lock_val;
                chain->lock_ptr->lock_dep = this;

                AgileLock * ptr_acquired_list = chain->lock_ptr->next;

                while(ptr_acquired_list != chain->lock_ptr){
                    
                    ptr_acquired_list->tid_dep = lock_val;
                    ptr_acquired_list->lock_dep = this;
                    ptr_acquired_list = ptr_acquired_list->next;
                
                }

                // retrive lock
                AgileLock * ptr_deps = this;
                bool find_deadlock = false;
                find_deadlock = ptr_deps->tid_dep == tid;
                while(ptr_deps->lock_dep != nullptr && !find_deadlock){
                    find_deadlock = ptr_deps->lock_dep->tid_dep == tid;
                    ptr_deps = ptr_deps->lock_dep;
                }
                if(find_deadlock){
                    ptr_deps = this;
                    find_deadlock = false;
                    while(ptr_deps->lock_dep != nullptr && !find_deadlock){
                        printf("lock circle: %s => %s\n", this->lockname, this->lock_dep->lockname);
                        find_deadlock = ptr_deps->lock_dep->tid_dep == tid;
                        ptr_deps = ptr_deps->lock_dep;
                    }
                    // AgileLock * chainptr = chain->lock_ptr;
                    // while(chainptr->next != nullptr){
                    //     printf("lock chain: %s => %s\n", chainptr->lockname, chainptr->next->lockname);
                    // }
                }
                GPU_ASSERT(!find_deadlock, "Find Deadlock");

                if(ptr_deps->release_local == false){
                
                    // printf("warning: lock %s depensed on a non-local lock %s\n", this->lockname, ptr_deps->lockname);
                
                }
                
            } else {
                // this->releaseLink();
            }
            
#endif
            if(!non_block){
#if LOCK_DEBUG
                counter++;
                if(counter > 100000){
                    printf("try acquire %s failed too many times %d\n", this->lockname, lock_val);
                    counter = 0;
                    LOGGING(atomicAdd(&(logger->waitTooMany), 1));
                }
#endif
                // LOGGING(atomicAdd());
                goto AgileLock_acquire;
            }
        }

        return false;
    }

    __device__ bool remoteRelease(){

#if DEBUG 

        GPU_ASSERT(this->release_local == false, "Use remote release to a local lock"); // local lock cannot be release using this API
        // this->lockLink();
        this->next = nullptr;
        this->lock_dep = nullptr;
        this->tid_dep = TID_NONE;

#else
#endif

        
        atomicExch(&this->lock, TID_NONE);
        // __threadfence_system();

// #if DEBUG
        // while(atomicCAS(&(this->link_lock), 1, 0) != 1);
// #endif

        return true;
    
    }

    __device__ bool release(AgileLockChain * chain){

#if DEBUG 
        extern __shared__ unsigned int AGILE_BID;
        TID_TYPE tid = AGILE_BID * blockDim.x + threadIdx.x;
        // this->lockLink();
        GPU_ASSERT((tid == this->lock) && this->release_local, "Use local release on remote lock"); // release a local
        if(this->next == this){

            chain->lock_ptr = nullptr;

        }else{
            
            AgileLock * prev_ptr = this->next;
            
            while(prev_ptr->next != this){
                // LOGGING(atomicAdd(&(logger->deadlock_check), 1));
                prev_ptr = prev_ptr->next;
            }

            prev_ptr->next = this->next;
            chain->lock_ptr = prev_ptr;
        }
        
        this->next = nullptr;
        this->lock_dep = nullptr;
        this->tid_dep = TID_NONE;

#endif

        __threadfence_system();
        atomicExch(&(this->lock), TID_NONE);
        
// #if DEBUG
        // __threadfence_system();
        // unsigned int * lock_ptr = this->link_lock_ptr;
        // this->link_lock_ptr = nullptr;
        // while(atomicCAS(lock_ptr, 1, 0) != 1);
        
// #endif
        return true;
    }

};


class AgileLockLocalChecker {

    AgileLock *lock;

    bool acquired;
    bool released;
    bool local_release;

public:
    
    __device__ AgileLockLocalChecker(AgileLock *lock, bool local_release) : lock(lock), local_release(local_release), acquired(false), released(false) {
    }

    __device__ ~AgileLockLocalChecker(){
        GPU_ASSERT((acquired && local_release) == released, "Local lock acquired but not released");
    }

    __device__ bool try_acquire(AgileLockChain * chain){
        if (this->acquire(chain, true, true)){
            acquired = true;
            return true;
        } 
        return false;
    }

    __device__ bool try_acquire(AgileLockChain * chain, bool local_release){
        if(this->acquire(chain, local_release, true)){
            acquired = true;
            return true;
        }
        return false;
    }

    __device__ bool acquire(AgileLockChain * chain){
        acquired = true;
        return this->acquire(chain, true, false);
    }

    __device__ bool acquire(AgileLockChain * chain, bool local_release){
        GPU_ASSERT(local_release == this->local_release, "AgileLockLocalChecker lock type mismatch");
        acquired = true;
        return this->acquire(chain, local_release, false);
    }

    __device__ bool acquire(AgileLockChain * chain, bool local_release, bool attempt) {
        GPU_ASSERT(local_release == this->local_release, "AgileLockLocalChecker lock type mismatch");
        if(lock->acquire(chain, local_release, attempt)){
            acquired = true;
            return true;
        }
        return false;

    }

};


#endif