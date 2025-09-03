#pragma once

#include "agile_helper.h"
#include "agile_dma.hpp"
__device__
uint32_t g_cmd_idx = 0;
__device__
uint32_t waitCpl(AgileDmaQueuePairDevice * queue_pair, uint32_t warp_idx, uint32_t mask){
    AgileDmaCQDevice * cq = queue_pair->cq;
    AgileDmaSQDevice * sq = queue_pair->sq;
    uint32_t flag = (mask >> warp_idx) & 0x1;
    if(flag == 1){
        return 1;
    }
    volatile dma_cpl_t *cpl = &(cq->cpl[cq->head_offset + warp_idx]);
    if(cpl->status == DMA_CPL_READY){ // indicate a new completion is arrived
        flag = 1;
        uint32_t identifier = cpl->identifier;
        cpl->phase = 0;
        cpl->identifier = 0;
        sq_entry_finish_upd(&sq->cmd_locks[identifier]);
        cpl->status = DMA_CPL_EMPTY;
    }
    // printf("flag: %d\n", flag);
    return flag;
}

__device__
void warpService(int32_t warp_idx, AgileDmaQueuePairDevice * queue_pair) {
    AgileDmaCQDevice * cq = queue_pair->cq;
    uint32_t mask = cq->mask;
    if(warp_idx == 0){
        // printf("mask: %x cq->head_offset: %d\n", mask, cq->head_offset);
    }
    uint32_t processed = waitCpl(queue_pair, warp_idx, mask);
    mask = __ballot_sync(0xFFFFFFFF, processed);

    if(warp_idx == 0){
        if(mask == 0xFFFFFFFF){
            mask = 0;
            cq->head_offset = (cq->head_offset + 32) % cq->depth;
        }
        cq->mask = mask;
    }
}

__global__
void pollingService(AgileDmaQueuePairDevice * queue_pairs, uint32_t queue_num, uint32_t *g_stop_sig) {
    uint32_t stop_sig = 0;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = tid / 32;
    uint32_t warp_idx = tid % 32;

    if(warp_id >= queue_num){
        return;
    }

    do{
        __ballot_sync(0xFFFFFFFF, 1);
        warpService(warp_idx, &(queue_pairs[warp_id]));
        if(warp_idx == 0){
            stop_sig = *((volatile uint32_t *)g_stop_sig);
        }
        stop_sig = __shfl_sync(0xFFFFFFFF, stop_sig, 0);
    } while (stop_sig == 0);

}

void stop_service(uint32_t *g_stop_sig){
    *g_stop_sig = 1;
    printf("stop_sig: %d\n", *g_stop_sig);
}
