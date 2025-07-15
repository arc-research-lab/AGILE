#ifndef AGILE_ARRAY_WARP_H
#define AGILE_ARRAY_WARP_H

#include "agile_helpper.h"

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileCtrl;


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename D_Type>
class AgileArrayWarpDev{
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl;
    unsigned int dev_idx;
    AgileLockChain * chain;
public:
    __device__ AgileArrayWarpDev(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, unsigned int dev_idx, AgileLockChain * chain){
        this->ctrl = ctrl;
        this->dev_idx = dev_idx;
        this->chain = chain;
    }

    __device__ const D_Type operator[](unsigned long idx) const;

};

/**
 * the array warp access the data in synchronous way, therefore, can directly modify GPU cache without deadlock problem
 */
template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
class AgileArrayWarp {
    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl;
    AgileLockChain * chain;
public:
    __device__ AgileArrayWarp(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * ctrl, AgileLockChain & chain){
        this->ctrl = ctrl;
        this->chain = &chain;
    }

    __device__ AgileArrayWarpDev<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T> operator[](NVME_DEV_IDX_TYPE dev_id);

    __device__ void operator() (NVME_DEV_IDX_TYPE dev_id, unsigned long idx, T val);
};

#include "agile_array_warp.tpp"
#endif