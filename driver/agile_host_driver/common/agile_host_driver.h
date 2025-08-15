#ifndef _AGILE_HOST_DRIVER_H_
#define _AGILE_HOST_DRIVER_H_

#ifdef __KERNEL__
    #include <linux/ioctl.h>
    #include <linux/types.h> 
#else
    #include <sys/ioctl.h>
    #include <stdint.h>
#endif


struct cache_buffer {
    uint64_t addr;
    uint64_t size;
    void *vaddr_krnl;
    void *vaddr_user;
};

struct dma_command {
    uint64_t src_addr;
    uint64_t dst_addr;
    void *src_vaddr_krnl;
    void *dst_vaddr_krnl;
    volatile uint32_t *cpl_ptr;
    uint32_t size; // in bytes
    uint32_t direction; // 0: CPU to GPU, 1: GPU to CPU
    uint32_t identifier; // command identifier
    uint32_t pid;       // process identifier

    // uint32_t * cpl;
    // uint32_t offset;
};



#define AGILE_MAGIC 'a'
#define IOCTL_ALLOCATE_CACHE_BUFFER       _IOWR   (AGILE_MAGIC, 0x02, struct cache_buffer *)
#define IOCTL_SET_CACHE_BUFFER            _IOW    (AGILE_MAGIC, 0x03, struct cache_buffer *)
#define IOCTL_FREE_CACHE_BUFFER           _IOWR   (AGILE_MAGIC, 0x04, struct cache_buffer *)
#define IOCTL_SUBMIT_DMA_CMD              _IOW    (AGILE_MAGIC, 0x01, struct dma_command *)

#endif