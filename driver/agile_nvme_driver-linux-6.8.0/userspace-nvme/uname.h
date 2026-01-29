#ifndef UNAME_H
#define UNAME_H

#include <iostream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <memory.h>

#include "agile_nvme_driver.h"
#include "nvme_reg_help.h"


#define AGILE_NVME_READ 0x2
#define AGILE_NVME_WRITE 0x1

struct dma_buffer dma_allocate(int fd, uint32_t size);

volatile unsigned int * dma_mmap(int fd, struct dma_buffer *buf);

void dma_munmap_free(int fd, volatile unsigned int *buf_ptr, struct dma_buffer *buf);

class NVMeIOPair {
public:
    volatile unsigned int *sq_db;
    volatile unsigned int *cq_db;

    struct dma_buffer sq_buf;
    struct dma_buffer cq_buf;

    volatile unsigned int *sq_ptr;
    volatile unsigned int *cq_ptr;

    unsigned int queue_depth;

    unsigned int sq_idx;
    unsigned int cq_idx;
    unsigned int cq_phase = 1;


    void submitIO(unsigned int CMD_TYPE, struct dma_buffer *buf) {
        // Submit I/O request using sq_ptr and sq_buf
        volatile unsigned int *cmd = sq_ptr + sq_idx * 16;
        memset((void *)cmd, 0, 16 * sizeof(uint32_t));
        cmd[0] = CMD_TYPE | (sq_idx << 16); // Set Command Identifier
        cmd[1] = 0x01; // LBA format
        cmd[2] = 0; // LBA low
        cmd[3] = 0; // LBA mid
        cmd[4] = 0; // LBA high
        cmd[5] = 0; // PRP1
        cmd[6] = buf->addr & 0xFFFFFFFF; // Data buffer low
        cmd[7] = (buf->addr >> 32) & 0xFFFFFFFF; // Data buffer high
        cmd[8] = 0; // Metadata buffer
        cmd[9] = 0; // Command Identifier
        cmd[10] = 0; // SSD target block
        cmd[11] = 0;
        cmd[12] = 0; // how many blocks to read/write - 1
        cmd[13] = 0;
        cmd[14] = 0;
        cmd[15] = 0;

        volatile unsigned int *cpl = cq_ptr + cq_idx * 4;
        cpl[3] = 0xFFFFFFFF;
        __sync_synchronize();
        sq_idx = (sq_idx + 1) % queue_depth;
        *sq_db = sq_idx; // Update SQ doorbell
        std::cout << "Submitted I/O command: ";
        for (int i = 0; i < 16; ++i) {
            printf("%.8x ", cmd[i]); // Print command
        }
        printf("\n");
        std::cout << std::dec << std::endl; 

        // Wait for completion
        std::cout << "Waiting for completion... " << cq_idx << std::endl;
        int wait_loops = 0;
        while (cpl[3] == 0xFFFFFFFF) {
            usleep(100); // Wait for completion
            if (++wait_loops > 100000) {
                std::cerr << "I/O completion timeout. CPL=" << std::hex
                          << cpl[0] << " " << cpl[1] << " " << cpl[2] << " " << cpl[3]
                          << std::dec << std::endl;
                return;
            }
        }

        // Process completion
        std::cout << "I/O Completion: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << std::hex << cpl[i] << " ";
        }
        std::cout << std::dec << std::endl;
        unsigned int status = (cpl[3] >> 17) & 0x7FFF;
        if (status != 0) {
            std::cerr << "I/O status error: 0x" << std::hex << status << std::dec << std::endl;
        }
        std::cout << std::dec << std::endl;
        cq_idx = (cq_idx + 1) % queue_depth; // Update CQ index
        if (cq_idx == 0) {
            cq_phase ^= 1;
        }
        *cq_db = cq_idx; // Update CQ doorbell
    }
};

class NVMeDevice {

private:
    std::string device;
    int fd;
    struct bar_info bar_info;
    struct dma_buffer buf_asq;
    struct dma_buffer buf_acq;
    volatile unsigned int *bar_ptr;
    volatile unsigned int *asq_ptr;
    volatile unsigned int *acq_ptr;

    unsigned int admin_queue_depth; 
    unsigned int asq_idx = 0;
    unsigned int acq_idx = 0;
    unsigned int acq_phase = 1;

    volatile unsigned int *asq_db;
    volatile unsigned int *acq_db;

    NVMeIOPair io_pair;


public:

    int getFD() const {
        return fd;
    }

    NVMeDevice(std::string device, unsigned int admin_queue_depth) : device(device), admin_queue_depth(admin_queue_depth) {
        this->fd = open(device.c_str(), O_RDWR);
        if (this->fd < 0) {
            perror("open");
            throw std::runtime_error("Failed to open NVMe device");
            exit(1);
        }

        if (ioctl(this->fd, IOCTL_GET_BAR, &this->bar_info) < 0) {
            perror("ioctl: get bar info");
            close(this->fd);
            throw std::runtime_error("Failed to get BAR info");
            exit(1);
        }

        if (ioctl(this->fd, IOCTL_SET_MMAP_TO_BAR, NULL) < 0) {
            perror("ioctl: set mmap to BAR");
            close(this->fd);
            throw std::runtime_error("Failed to set mmap to BAR");
            exit(1);
        }

        this->bar_ptr = (volatile unsigned int *) mmap(NULL, bar_info.size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
        if (this->bar_ptr == MAP_FAILED) {
            perror("mmap");
            close(this->fd);
            throw std::runtime_error("Failed to mmap BAR");
            exit(1);
        }

        std::cout << "NVMe device opened successfully: " << this->device << " \nBAR: start=0x" << std::hex << bar_info.phys_addr
                  << ", size=0x" << std::dec << bar_info.size << " mmap ptr: 0x" << std::hex << this->bar_ptr << std::dec << std::endl;

        asq_db = SQ_DBL(this->bar_ptr, 0, CAP$DSTRD(this->bar_ptr));
        acq_db = CQ_DBL(this->bar_ptr, 0, CAP$DSTRD(this->bar_ptr));
    }

    ~NVMeDevice(){
        std::cout << "Closing NVMe device: " << this->device << std::endl;
        std::cout << "Unmapping BAR: " << std::hex << this->bar_ptr << std::dec << std::endl;
        if (this->bar_ptr) {
            munmap((void *)this->bar_ptr, this->bar_info.size);
        }

        std::cout << "Free acq buffer: " << std::hex << this->buf_acq.addr << std::dec << std::endl;
        if(this->acq_ptr){
            munmap((void *)this->acq_ptr, this->buf_acq.size);
        }
        if(ioctl(this->fd, IOCTL_FREE_DMA_BUFFER, &this->buf_acq) < 0){
            perror("ioctl: free dma buffer acq");
        }

        std::cout << "Free asq buffer: " << std::hex << this->buf_asq.addr << std::dec << std::endl;
        if(this->asq_ptr){
            munmap((void *)this->asq_ptr, this->buf_asq.size);
        }
        if(ioctl(this->fd, IOCTL_FREE_DMA_BUFFER, &this->buf_asq) < 0){
            perror("ioctl: free dma buffer asq");
        }

        std::cout << "Closing NVMe device file descriptor: " << this->fd << std::endl;
        if (this->fd >= 0) {
            close(this->fd);
        }
    }

  
    void resetController() {

        volatile uint32_t* cc = CC(this->bar_ptr);
        *cc = *cc & ~1; // Disable controller
        usleep(CAP$TO(this->bar_ptr) * 500);
        while (CSTS$RDY(this->bar_ptr) != 0) {
            std::cout << "CSTS$RDY not ready\n";
            usleep(CAP$TO(this->bar_ptr) * 500);
        }
        printf("CAP_DSTRD: %llu\n", CAP$DSTRD(this->bar_ptr));
        printf("CAP_MQES: %llu\n", CAP$MQES(this->bar_ptr) + 1);
        printf("CAP_TO: %llu\n", CAP$TO(this->bar_ptr));
    }

    void registerAdminQueues() {
        this->buf_asq = dma_allocate(this->fd, this->admin_queue_depth * sizeof(uint32_t) * 16);
        if (this->buf_asq.vaddr == NULL) {
            throw std::runtime_error("Failed to allocate DMA buffer for ASQ");
        }

        this->asq_ptr = dma_mmap(this->fd, &this->buf_asq);
        if (this->asq_ptr == NULL) {
            dma_munmap_free(this->fd, this->asq_ptr, &this->buf_asq);
            throw std::runtime_error("Failed to mmap ASQ buffer");
        }
        memset((void *)this->asq_ptr, 0, this->buf_asq.size);

        this->buf_acq = dma_allocate(this->fd, this->admin_queue_depth * sizeof(uint32_t) * 4);
        if (this->buf_acq.vaddr == NULL) {
            dma_munmap_free(this->fd, this->asq_ptr, &this->buf_asq);
            throw std::runtime_error("Failed to allocate DMA buffer for ACQ");
        }

        this->acq_ptr = dma_mmap(this->fd, &this->buf_acq);
        if (this->acq_ptr == NULL) {
            dma_munmap_free(this->fd, this->asq_ptr, &this->buf_asq);
            dma_munmap_free(this->fd, this->acq_ptr, &this->buf_acq);
            throw std::runtime_error("Failed to mmap ACQ buffer");
        }
        memset((void *)this->acq_ptr, 0, this->buf_acq.size);

        std::cout << "Allocated DMA buffers for ASQ and ACQ: ASQ addr=0x" << std::hex << this->buf_asq.addr
                  << ", size=" << std::dec << this->buf_asq.size
                  << " ACQ addr=0x" << std::hex << this->buf_acq.addr
                  << ", size=" << std::dec << this->buf_acq.size << std::endl;

        volatile uint32_t* aqa = AQA(this->bar_ptr);
        *aqa = AQA$AQS(this->admin_queue_depth - 1) | AQA$AQC(this->admin_queue_depth - 1);

        volatile uint64_t* acq = ACQ(this->bar_ptr);
        *acq = this->buf_acq.addr;

        volatile uint64_t* asq = ASQ(this->bar_ptr);
        *asq = this->buf_asq.addr;

        volatile uint32_t* cc = CC(this->bar_ptr);
        *cc = CC$IOCQES(4) | CC$IOSQES(6) | CC$MPS(0) | CC$CSS(0) | CC$EN(1);
        usleep(CAP$TO(this->bar_ptr) * 500);    
        while (CSTS$RDY(this->bar_ptr) != 1) {
            std::cout << "CSTS$RDY not ready\n";
            usleep(CAP$TO(this->bar_ptr) * 500);
        }
    }

    NVMeIOPair * registerIOQueue(unsigned int queue_depth){
        struct dma_buffer buf_iocq = dma_allocate(this->fd, queue_depth * sizeof(uint32_t) * 4);
        if (buf_iocq.vaddr == NULL) {
            throw std::runtime_error("Failed to allocate DMA buffer for I/O CQ");
        }
        volatile unsigned int *iocq_ptr = dma_mmap(this->fd, &buf_iocq);
        if (iocq_ptr == NULL) {
            dma_munmap_free(this->fd, iocq_ptr, &buf_iocq);
            throw std::runtime_error("Failed to mmap I/O CQ buffer");
        }
        memset((void *)iocq_ptr, 0, buf_iocq.size);

        struct dma_buffer buf_iosq = dma_allocate(this->fd, queue_depth * sizeof(uint32_t) * 16);
        if (buf_iosq.vaddr == NULL) {
            dma_munmap_free(this->fd, iocq_ptr, &buf_iocq);
            throw std::runtime_error("Failed to allocate DMA buffer for I/O SQ");
        }
        volatile unsigned int *iosq_ptr = dma_mmap(this->fd, &buf_iosq);
        if (iosq_ptr == NULL) {
            dma_munmap_free(this->fd, iocq_ptr, &buf_iocq);
            dma_munmap_free(this->fd, iosq_ptr, &buf_iosq);
            throw std::runtime_error("Failed to mmap I/O SQ buffer");
        }
        memset((void *)iosq_ptr, 0, buf_iosq.size);

        std::cout << "Allocated DMA buffers for I/O SQ and CQ: SQ addr=0x" << std::hex << buf_iosq.addr
                  << ", size=" << std::dec << buf_iosq.size
                  << " CQ addr=0x" << std::hex << buf_iocq.addr
                  << ", size=" << std::dec << buf_iocq.size << std::endl;

        // Register I/O CQ
        volatile unsigned int *cmd = this->asq_ptr + this->asq_idx * 16;
        memset((void *)cmd, 0, 16 * sizeof(uint32_t));
        cmd[0] = 0x05 | (this->asq_idx << 16);
        cmd[6] = buf_iocq.addr & 0xFFFFFFFF;
        cmd[7] = (buf_iocq.addr >> 32) & 0xFFFFFFFF;
        cmd[10] = ((queue_depth - 1) << 16) | 1; // QSIZE=queue_depth-1, QID=1
        cmd[11] = 0x1; // PC=1
        volatile unsigned int *cpl = this->acq_ptr + this->acq_idx * 4;
        cpl[3] = 0xFFFFFFFF;
        __sync_synchronize();
        this->asq_idx = (this->asq_idx + 1) % this->admin_queue_depth;
        *this->asq_db = this->asq_idx;
        for (int i = 0; i < 16; ++i) {
            printf("%.8x ", cmd[i]); // Print command
        }
        printf("\n");
        this->waitCompletion();
        std::cout << "Registered I/O CQ: addr=0x" << std::hex << buf_iocq.addr
                  << ", size=" << std::dec << buf_iocq.size << std::endl;

        // Register I/O SQ
        cmd = this->asq_ptr + this->asq_idx * 16;
        memset((void *)cmd, 0, 16 * sizeof(uint32_t));
        cmd[0] = 0x01 | (this->asq_idx << 16);
        cmd[6] = buf_iosq.addr & 0xFFFFFFFF;
        cmd[7] = (buf_iosq.addr >> 32) & 0xFFFFFFFF;
        cmd[10] = ((queue_depth - 1) << 16) | 1; // QSIZE=queue_depth-1, QID=1
        cmd[11] = (1 << 16) | 0x1; // CQID=1 (upper 16), SQ flags: PC=1
        cpl = this->acq_ptr + this->acq_idx * 4;
        cpl[3] = 0xFFFFFFFF;
        __sync_synchronize();
        this->asq_idx = (this->asq_idx + 1) % this->admin_queue_depth;
        *this->asq_db = this->asq_idx;
        for (int i = 0; i < 16; ++i) {
            printf("%.8x ", cmd[i]); // Print command
        }
        printf("\n");
        this->waitCompletion();
        std::cout << "Registered I/O SQ: addr=0x" << std::hex << buf_iosq.addr
                  << ", size=" << std::dec << buf_iosq.size << std::endl;
        NVMeIOPair *io_pair = new NVMeIOPair();

        io_pair->sq_db = SQ_DBL(this->bar_ptr, 1, CAP$DSTRD(this->bar_ptr));
        io_pair->cq_db = CQ_DBL(this->bar_ptr, 1, CAP$DSTRD(this->bar_ptr));

        io_pair->sq_buf = buf_iosq;
        io_pair->cq_buf = buf_iocq;
        io_pair->sq_ptr = iosq_ptr;
        io_pair->cq_ptr = iocq_ptr;
        io_pair->queue_depth = queue_depth;
        io_pair->sq_idx = 0;
        io_pair->cq_idx = 0;
        std::cout << "Registered I/O queue with depth: " << queue_depth << std::endl;
        return io_pair;
    }


    void waitCompletion() {
        volatile unsigned int *cpl = this->acq_ptr + this->acq_idx * 4;
        int wait_loops = 0;
        while (cpl[3] == 0xFFFFFFFF) {
            usleep(100); // Wait for completion
            if (++wait_loops > 100000) {
                std::cerr << "Admin completion timeout. CPL=" << std::hex
                          << cpl[0] << " " << cpl[1] << " " << cpl[2] << " " << cpl[3]
                          << std::dec << std::endl;
                std::cerr << "CC=" << std::hex << *CC(this->bar_ptr)
                          << " CSTS=" << *CSTS(this->bar_ptr)
                          << " AQA=" << *AQA(this->bar_ptr)
                          << " ASQ=" << *ASQ(this->bar_ptr)
                          << " ACQ=" << *ACQ(this->bar_ptr)
                          << std::dec << std::endl;
                return;
            }
        }
        unsigned int status = (cpl[3] >> 17) & 0x7FFF;
        printf("Admin NVMe CPL: %.8x %.8x %.8x %.8x\n",
               cpl[0], cpl[1], cpl[2], cpl[3]);
        if (status != 0) {
            std::cerr << "Admin status error: 0x" << std::hex << status << std::dec << std::endl;
        }
        this->acq_idx = (this->acq_idx + 1) % this->admin_queue_depth;
        if (this->acq_idx == 0) {
            this->acq_phase ^= 1;
        }
        *this->acq_db = this->acq_idx;
    }

};

struct dma_buffer dma_allocate(int fd, uint32_t size) {
    struct dma_buffer buf;
    buf.size = size;
    
    if (ioctl(fd, IOCTL_ALLOCATE_DMA_BUFFER, &buf) < 0) {
        perror("ioctl: allocate dma buffer");
        buf.vaddr = NULL;
        return buf;
    }
    return buf;
}

volatile unsigned int * dma_mmap(int fd, struct dma_buffer *buf) {

    if (ioctl(fd, IOCTL_SET_DMA_BUFFER, buf) < 0) {
        perror("ioctl: set dma buffer");
        return NULL;
    }
    if (ioctl(fd, IOCTL_SET_MMAP_TO_DMA, NULL) < 0) {
        perror("ioctl: set mmap to dma");
        return NULL;
    }
    printf("dma_mmap DMA buffer set: addr=0x%llx size=%u\n",
           (unsigned long long)buf->addr, buf->size);
    volatile unsigned int *buf_ptr = (volatile unsigned int *) mmap(NULL, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (buf_ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    return buf_ptr;
}

void dma_munmap_free(int fd, volatile unsigned int *buf_ptr, struct dma_buffer *buf) {
    if (buf_ptr) {
        munmap((void *)buf_ptr, buf->size);
    }
    if (ioctl(fd, IOCTL_FREE_DMA_BUFFER, buf) < 0) {
        perror("ioctl: free dma buffer");
    }
}


#endif