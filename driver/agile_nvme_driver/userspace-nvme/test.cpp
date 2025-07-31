# include "uname.h"

int main(int argc, char *argv[]) {

    NVMeDevice nvme_device("/dev/AGILE-NVMe-0000:98:00.0", 256);

    nvme_device.resetController();
    nvme_device.registerAdminQueues();
    NVMeIOPair *io_pair = nvme_device.registerIOQueue(256);

    struct dma_buffer buf;
    buf.size = 4096; // 4KB buffer

    if(ioctl(nvme_device.getFD(), IOCTL_ALLOCATE_DMA_BUFFER, &buf) < 0) {
        perror("ioctl: allocate dma buffer");
        return 1;
    }
    unsigned int *buf_ptr = (unsigned int *) mmap(NULL, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, nvme_device.getFD(), 0);
    if (buf_ptr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    printf("Allocated DMA buffer: addr=0x%llx size=%u\n", (unsigned long long)buf.addr, buf.size);

    for (unsigned int i = 0; i < buf.size / sizeof(unsigned int); ++i) {
        buf_ptr[i] = i*2; // Fill buffer with some data
    }

    io_pair->submitIO(AGILE_NVME_WRITE, &buf); // Example command type 0x01

    
    for (unsigned int i = 0; i < buf.size / sizeof(unsigned int); ++i) {
        buf_ptr[i] = 0x0; // Fill buffer with some data
    }

    io_pair->submitIO(AGILE_NVME_READ, &buf); // Example command type 0x02

    for (unsigned int i = 0; i < buf.size / sizeof(unsigned int); i += 20) {
        for(int j = 0; j < 20 && i + j < buf.size / sizeof(unsigned int); ++j) {
            printf("%.8x ", buf_ptr[i + j]); // Print the data read
        }
        printf("\n"); // Print the data read
    }

    return 0;
}