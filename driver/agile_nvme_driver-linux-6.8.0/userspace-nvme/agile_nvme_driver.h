#ifndef _AGILE_IOCTL_H_
#define _AGILE_IOCTL_H_

#ifdef __KERNEL__
    #include <linux/ioctl.h>
    #include <linux/types.h> 
#else
    #include <sys/ioctl.h>
    #include <stdint.h>
#endif

struct bar_info {
    uint64_t phys_addr;
    uint64_t size;
};

struct dma_buffer {
    uint64_t addr;
    uint32_t size;
    void *vaddr;
};

#define AGILE_MAGIC 'a'
#define IOCTL_GET_BAR                   _IOR    (AGILE_MAGIC, 0x01, struct bar_info *)
#define IOCTL_ALLOCATE_DMA_BUFFER       _IOWR   (AGILE_MAGIC, 0x02, struct dma_buffer *)
#define IOCTL_SET_DMA_BUFFER            _IOW    (AGILE_MAGIC, 0x03, struct dma_buffer *)
#define IOCTL_FREE_DMA_BUFFER           _IOWR   (AGILE_MAGIC, 0x04, struct dma_buffer *)
#define IOCTL_SET_MMAP_TO_BAR           _IOW    (AGILE_MAGIC, 0x05, void *)
#define IOCTL_SET_MMAP_TO_DMA           _IOW    (AGILE_MAGIC, 0x06, void *)

// Test IOCTL commands
#define IOCTL_QUERY_MMAP_FLAG           _IOR    (AGILE_MAGIC, 0xFFFF0005, uint32_t *)
// #define INIT_GLOBAL_BUFFER           _IOW    (AGILE_MAGIC, 0x04, uint32_t *)
// #define READ_GLOBAL_BUFFER           _IOW    (AGILE_MAGIC, 0x05, uint32_t *)

#endif