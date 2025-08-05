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
    
};

#define AGILE_MAGIC 'a'
#define IOCTL_ALLOCATE_CACHE_BUFFER       _IOWR   (AGILE_MAGIC, 0x02, struct cache_buffer *)
#define IOCTL_SET_CACHE_BUFFER            _IOW    (AGILE_MAGIC, 0x03, struct cache_buffer *)
#define IOCTL_FREE_CACHE_BUFFER           _IOWR   (AGILE_MAGIC, 0x04, struct cache_buffer *)

#endif