#pragma once

#ifdef __KERNEL__
    #include <linux/ioctl.h>
    #include <linux/types.h> 
#else
    #include <sys/ioctl.h>
    #include <stdint.h>
#endif

#define DMA_CPU2GPU 0
#define DMA_GPU2CPU 1

#define DMA_CPL_EMPTY 0
#define DMA_CPL_READY 1

#define SQ_CMD_EMPTY 0
#define SQ_CMD_OCCUPY 1
#define SQ_CMD_ISSUED 2
#define SQ_CMD_FINISHED 3

// the agile_dma_cmd_t will be located on CPU's memory and the GPU threads will update it directly
struct agile_dma_cmd_t {
    uint64_t src_offset;
    uint64_t dst_offset;
    uint32_t size;
    uint16_t dma_engine_id; // ignore if GPU DMA engine is used; by default, CPU monitor threads sepecify the dma engine id;
    uint8_t direction; // DMA_CPU2GPU; DMA_GPU2CPU
    uint8_t trigger; // SQ_CMD_EMPTY; SQ_CMD_OCCUPY; SQ_CMD_FINISHED
};


// the agile_dma_cpl_t will be located on GPU's memory and the userspace threads and the kernel callback function will update it directly (using GPU's DMA and CPU's DMA)
struct agile_dma_cpl_t {
    uint32_t identifier; // coresponding to which command in the SQ
    uint16_t status; // DMA_CPL_EMPTY; DMA_CPL_READY
    uint16_t reserve;
};