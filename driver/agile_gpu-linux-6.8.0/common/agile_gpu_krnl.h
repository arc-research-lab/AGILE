#ifndef AGILE_GPU_H
#define AGILE_GPU_H

#ifdef __KERNEL__
    #include <linux/ioctl.h>
    #include <linux/types.h> 
#else
    #include <sys/ioctl.h>
    #include <stdint.h>
#endif

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct pin_buffer_params
{
    // in
    uint64_t vaddr; // aligned GPU virtual address
    uint64_t size; // size in bytes, multiple of 64KB
    uint64_t p2p_token;
    uint32_t va_space;
    // out
    uint64_t phy_addr; // the physical address of the start of the pinned GPU buffer
    struct nvidia_p2p_page_table *page_table; // kernel space pointer, not used in user space
    void * krnl_ptr; // kernel space pointer, not used in user space
};


struct cpu_dram_buf {
    uint64_t phy_addr;
    uint64_t size;
    void * vaddr_krnl;
    void * vaddr_user;
};


#define AGILE_MAGIC 'a'
#define IOCTL_PIN_GPU_BUFFER        _IOWR   (AGILE_MAGIC, 0x01, struct pin_buffer_params *)
#define IOCTL_UNPIN_GPU_BUFFER      _IOW    (AGILE_MAGIC, 0x02, struct pin_buffer_params *)
#define IOCTL_ALLOCATE_DRAM_BUFFER  _IOWR   (AGILE_MAGIC, 0x03, struct dram_buf *)
#define IOCTL_FREE_DRAM_BUFFER      _IOW    (AGILE_MAGIC, 0x04, struct dram_buf *)

#endif // AGILE_GPU_H