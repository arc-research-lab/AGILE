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
};


struct dma_command {
    uint64_t src_addr;
    uint64_t dst_addr;
    void *src_vaddr_krnl;
    void *dst_vaddr_krnl;
    uint32_t size;
    
    uint64_t coroutine_idx;

    uint32_t cpl_queue_idx;
    uint32_t dma_channel_idx;
    
    // 0: CPU to GPU, 1: GPU to CPU
    // command identifier
    // uint32_t pid;       // process identifier

    // int eventfd; // for epoll

};

struct cpl_queue_config {
    uint32_t depth;
    uint32_t size;
    uint32_t idx;
    void *vaddr_krnl;
    int efd;
};


struct dma_cpl_entry {
    // void * coroutine_handle; // not safe, user space should track the handles all the time
    uint64_t coroutine_idx;
};

union dma_cpl_queue_data {
    struct dma_cpl_entry * entry;
    void * ptr;
};




struct cpl_queue_upd {
    uint32_t idx;
    uint32_t head;
};

#define AGILE_MAGIC 'a'


#define IOCTL_ALLOCATE_CACHE_BUFFER       _IOWR   (AGILE_MAGIC, 0x01, struct dma_buffer *)
#define IOCTL_SET_CACHE_BUFFER            _IOW    (AGILE_MAGIC, 0x02, struct dma_buffer *)
#define IOCTL_FREE_CACHE_BUFFER           _IOWR   (AGILE_MAGIC, 0x03, struct dma_buffer *)
#define IOCTL_SUBMIT_DMA_CMD              _IOW    (AGILE_MAGIC, 0x04, struct dma_command *)

#define IOCTL_ALLOC_CPL_QUEUE_ARRAY       _IOWR   (AGILE_MAGIC, 0x05, uint32_t *)
#define IOCTL_FREE_CPL_QUEUE_ARRAY        _IOWR   (AGILE_MAGIC, 0x06, void *)
#define IOCTL_SET_CPL_QUEUE               _IOW    (AGILE_MAGIC, 0x07, struct cpl_queue_config *)
#define IOCTL_DEL_CPL_QUEUE               _IOW    (AGILE_MAGIC, 0x08, struct cpl_queue_config *)
#define IOCTL_GET_TOTAL_DMA_CHANNELS      _IOR    (AGILE_MAGIC, 0x09, uint32_t *)
#define IOCTL_UPDATE_CPL_QUEUE_HEAD       _IOW    (AGILE_MAGIC, 0x0A, struct cpl_queue_upd *)


// test ioctl functions
#define IOCTL_TEST_REGISTER_EVENTFD       _IOW    (AGILE_MAGIC, 0xFFFF0001, int *)
#define IOCTL_TEST_SEND_SIG               _IOW    (AGILE_MAGIC, 0xFFFF0002, int *)
#define IOCTL_TEST_DMA_EMU                _IOW    (AGILE_MAGIC, 0xFFFF0003, struct dma_command *)

#endif