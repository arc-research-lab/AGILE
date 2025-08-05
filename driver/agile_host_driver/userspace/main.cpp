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

int main(){
    int fd = open("/dev/AGILE-host", O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }

    struct cache_buffer cb;
    cb.size = 1024l * 1024l * 4; // 4MB buffer
    if (ioctl(fd, IOCTL_ALLOCATE_CACHE_BUFFER, &cb) < 0) {
        perror("ioctl: allocate cache buffer");
        close(fd);
        return 1;
    }

    printf("Allocated cache buffer: addr=0x%llx size=%lu\n",
           (unsigned long long)cb.addr, cb.size);

    unsigned int *mapped = (unsigned int *) mmap(NULL, cb.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        ioctl(fd, IOCTL_FREE_CACHE_BUFFER, &cb);
        close(fd);
        return 1;
    }

    // Use the mapped memory...

    for (unsigned int i = 0; i < cb.size / sizeof(unsigned int); ++i) {
        mapped[i] = i;
    }

    printf("First 10 values in mapped memory:\n");

    for (unsigned int i = 0; i < 10; ++i) {
        printf("%u ", mapped[i]);
    }
    printf("\n");

    munmap(mapped, cb.size);
    close(fd);
    return 0;
}