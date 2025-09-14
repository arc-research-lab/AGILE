#ifndef _AGILE_HOST_DRIVER_H_
#define _AGILE_HOST_DRIVER_H_

#ifdef __KERNEL__
    #include <linux/ioctl.h>
    #include <linux/types.h> 
#else
    #include <sys/ioctl.h>
    #include <stdint.h>
#endif


struct dma_buffer {
    uint64_t addr;
    uint64_t size;
    void *vaddr_krnl;
    void *vaddr_user;
    void *vaddr_cuda;
};

struct dma_queue_pair_data{
    uint32_t queue_depth;
    void *cmds; // located in CPU memory, written by GPU, read by CPU
    void *cpls; // located in GPU memory, written by CPU, read by GPU
};

struct dma_command {
    uint32_t queue_id;
    uint32_t count;
};

struct base_addr_offsets{
    uint64_t dram_offsets;
    uint64_t hbm_offsets;
};


#define AGILE_MAGIC 'a'

#define IOCTL_ALLOCATE_CACHE_BUFFER         _IOWR   (AGILE_MAGIC, 0x01, struct dma_buffer *)
#define IOCTL_SET_CACHE_BUFFER              _IOW    (AGILE_MAGIC, 0x02, struct dma_buffer *)
#define IOCTL_FREE_CACHE_BUFFER             _IOWR   (AGILE_MAGIC, 0x03, struct dma_buffer *)
#define IOCTL_GET_TOTAL_DMA_CHANNELS        _IOR    (AGILE_MAGIC, 0x04, uint32_t *)

#define IOCTL_SUBMIT_DMA_CMD                _IOWR   (AGILE_MAGIC, 0x05, struct dma_command *)
#define IOCTL_SET_CPU_DMA_QUEUE_NUM         _IOW    (AGILE_MAGIC, 0x06, uint32_t *)
#define IOCTL_FREE_CPU_DMA_QUEUES           _IOW    (AGILE_MAGIC, 0x07, uint32_t *)
#define IOCTL_REGISTER_DMA_QUEUE_PAIRS      _IOW    (AGILE_MAGIC, 0x08, struct dma_queue_pair_data *)
#define IOCTL_SET_BASE_ADDR_OFFSETS         _IOW    (AGILE_MAGIC, 0x09, struct base_addr_offsets *)

// #define IOCTL_ALLOC_CPL_QUEUE_ARRAY       _IOWR   (AGILE_MAGIC, 0x05, uint32_t *)
// #define IOCTL_FREE_CPL_QUEUE_ARRAY        _IOWR   (AGILE_MAGIC, 0x06, void *)
// #define IOCTL_SET_CPL_QUEUE               _IOW    (AGILE_MAGIC, 0x07, struct cpl_queue_config *)
// #define IOCTL_DEL_CPL_QUEUE               _IOW    (AGILE_MAGIC, 0x08, struct cpl_queue_config *)



// test ioctl functions
#define IOCTL_FORCE_RELEASE_DMA_CHANNELS    _IOW    (AGILE_MAGIC, 0xFFFF0001, uint32_t *)
#endif