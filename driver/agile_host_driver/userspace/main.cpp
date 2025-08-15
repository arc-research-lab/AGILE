#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#include "../common/agile_host_driver.h"


int allocate_dma_buffer(int fd, cache_buffer &buf, uint32_t size){
    buf.size = size;
    if (ioctl(fd, IOCTL_ALLOCATE_CACHE_BUFFER, &buf) < 0) {
        perror("ioctl: allocate cache buffer");
        close(fd);
        return 1;
    }
    buf.vaddr_user = mmap(NULL, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (buf.vaddr_user == MAP_FAILED) {
        perror("mmap");
        ioctl(fd, IOCTL_FREE_CACHE_BUFFER, &buf);
        close(fd);
        return 1;
    }
    return 0;
}

int free_dma_buffer(int fd, cache_buffer &buf){
    if (buf.vaddr_user && buf.vaddr_user != MAP_FAILED) {
        munmap(buf.vaddr_user, buf.size);
    }
    ioctl(fd, IOCTL_FREE_CACHE_BUFFER, &buf);
    return 0;
}

int main(){
    int fd = open("/dev/AGILE-host", O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }


    cache_buffer dma_src, dma_dst, dma_cpl;
    allocate_dma_buffer(fd, dma_src, 4096); // Allocate 4KB for source
    allocate_dma_buffer(fd, dma_dst, 4096);
    allocate_dma_buffer(fd, dma_cpl, 4096);

    
    for(uint32_t i = 0; i < 10; ++i){
        ((int*)dma_src.vaddr_user)[i] = i;
        ((int*)dma_dst.vaddr_user)[i] = 0;
    }

    printf("Initializing DMA buffers\n");


    // Submit DMA command
    dma_command cmd;
    cmd.src_addr = dma_src.addr;
    cmd.dst_addr = dma_dst.addr;
    cmd.size = 4096;
    cmd.direction = 0; // CPU to GPU
    cmd.identifier = 1;
    cmd.src_vaddr_krnl = dma_src.vaddr_krnl;
    cmd.dst_vaddr_krnl = dma_dst.vaddr_krnl;
    cmd.pid = getpid();
    cmd.cpl_ptr = (volatile uint32_t *)dma_cpl.vaddr_krnl;

    // Initialize the completion flag to 0

    volatile uint32_t *completion_flag = (volatile uint32_t *)dma_cpl.vaddr_user;
    *completion_flag = 0;

    if (ioctl(fd, IOCTL_SUBMIT_DMA_CMD, &cmd) < 0) {
        perror("ioctl: submit DMA command");
    }

    while (*completion_flag == 0) {
        usleep(1);
    }

    printf("DMA transfer completed, checking %d\n", *completion_flag);

    for(uint32_t i = 0; i < 10; ++i){
        printf("src[%d] = %d, dst[%d] = %d\n", i, ((int*)dma_src.vaddr_user)[i], i, ((int*)dma_dst.vaddr_user)[i]);
    }

    free_dma_buffer(fd, dma_src);
    free_dma_buffer(fd, dma_dst);
    free_dma_buffer(fd, dma_cpl);

    close(fd);
    return 0;
}