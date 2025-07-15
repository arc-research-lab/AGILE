#include "agile_ctrl.h"

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename D_Type>
__device__ AgileArrayWarpDev<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, D_Type> AgileArrayWarp<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, D_Type>::operator[](NVME_DEV_IDX_TYPE dev_id){
    return AgileArrayWarpDev<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, D_Type>(this->ctrl, dev_id, this->chain);
}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename D_Type>
__device__ const D_Type AgileArrayWarpDev<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, D_Type>::operator[](unsigned long idx) const{
    SSDBLK_TYPE blk_idx = idx / (this->ctrl->buf_size / sizeof(D_Type));
    return this->ctrl->template readCacheElement<D_Type>(this->dev_idx, blk_idx, idx % (this->ctrl->buf_size / sizeof(D_Type)), this->chain);
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl, typename T>
__device__ void AgileArrayWarp<GPUCacheImpl, CPUCacheImpl, ShareTableImpl, T>::operator()(NVME_DEV_IDX_TYPE dev_id, unsigned long idx, T val){
    SSDBLK_TYPE blk_idx = idx / (this->ctrl->buf_size / sizeof(T));
    this->ctrl->template writeCacheElement<T>(dev_id, blk_idx, idx % (this->ctrl->buf_size / sizeof(T)), val, this->chain);
}