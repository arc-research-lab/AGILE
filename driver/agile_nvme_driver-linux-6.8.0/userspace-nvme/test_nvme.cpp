# include "uname.h"

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <NVMe device path>" << std::endl;
        return 1;
    }

    // 1. Open the NVMe device
    NVMeDevice nvme_device(argv[1], 256);
    nvme_device.resetController();
    nvme_device.registerAdminQueues();
    NVMeIOPair *io_pair = nvme_device.registerIOQueue(256);

    // 2. Allocate DMA buffer
    struct dma_buffer buf = dma_allocate(nvme_device.getFD(), 4096); // 4KB buffer
    if (buf.vaddr == NULL) {
        return 1;
    }
    // 3. Map the DMA buffer
    volatile unsigned int *buf_ptr = dma_mmap(nvme_device.getFD(), &buf);
    if (buf_ptr == NULL) {
        dma_munmap_free(nvme_device.getFD(), buf_ptr, &buf);
        return 1;
    }

    printf("Allocated DMA buffer: addr=0x%llx size=%u\n", (unsigned long long)buf.addr, buf.size);

    // 4. Initialize the buffer with some data
    for (unsigned int i = 0; i < buf.size / sizeof(unsigned int); ++i) {
        buf_ptr[i] = i*2; // Fill buffer with some data
    }

    // 5. Write data to the NVMe device
    io_pair->submitIO(AGILE_NVME_WRITE, &buf); // Example command type 0x01

    
    // 6. Clear the buffer before reading
    for (unsigned int i = 0; i < buf.size / sizeof(unsigned int); ++i) {
        buf_ptr[i] = 0x0;
    }

    // 7. Read data from the NVMe device
    io_pair->submitIO(AGILE_NVME_READ, &buf); // Example command type 0x02

    // 8. Print the data read from the NVMe device
    for (unsigned int i = 0; i < 128; i += 16) {
        for(int j = 0; j < 16 && i + j < 128; ++j) {
            printf("%.8x ", buf_ptr[i + j]); // Print the data read
        }
        printf("\n"); // Print the data read
    }

    dma_munmap_free(nvme_device.getFD(), buf_ptr, &buf);
    return 0;
}