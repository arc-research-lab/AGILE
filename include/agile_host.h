#ifndef AGILE_HOST
#define AGILE_HOST

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>



#include "agile_swcache.h"
#include "agile_ctrl.h"
#include "agile_dmem.h"
#include "nvme_reg_help.h"
#include "agile_helper.h"

#include "agile_nvme_driver.h"


#define CQ_ALIGN 64 // for 4K alignment, if 16 is used then the cq depth should be 256
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define AGILE_NVME_GET_BAR_SIZE _IOR('N', 1, size_t)
#define CPU_MEM_SIZE 68719476736L
#define CPU_MEM_ADDR 0x2000000000L
#define CPU_PAGE_SIZE 65536 // align with GPU

class NvmeConfig {
public:
    std::string dev;
    int fd;
    volatile unsigned int * bar;
    unsigned long bar_size;
    
    unsigned int ssd_blk_offset;
    unsigned int queue_num;
    unsigned int queue_depth;
    unsigned long gpu_mem_addr;

    unsigned int CAP_DSTRD; // Doorbell register stride
    unsigned int CAP_MQES; // Max Queue Entries size
    unsigned int CAP_TO; // Timeout
    

    struct dma_buffer asq_buf;
    struct dma_buffer acq_buf;

    unsigned int asq_pos;
    unsigned int acq_pos;
    unsigned int admin_q_depth;
    unsigned int phase;
    volatile unsigned int * cq_ptr;
    volatile unsigned int * sq_ptr;
    volatile unsigned int * admin_cqdb;
    volatile unsigned int * admin_sqdb;


    __host__ NvmeConfig(std::string dev, unsigned int ssd_blk_offset, unsigned int queue_num, unsigned int queue_depth) \
    : dev(dev), queue_num(queue_num), queue_depth(queue_depth), ssd_blk_offset(ssd_blk_offset) {
        this->asq_pos = 0;
        this->acq_pos = 0;
        this->admin_q_depth = 256; // TODO modify this
        this->phase = 0;

    }

    __host__ void openNvme(){
#if REGISTER_NVME
        this->fd = open(this->dev.c_str(), O_RDWR);
        if(this->fd == -1){
            std::cout << "open device fail: " << dev << std::endl;
            exit(0);
        }

        // get the BAR size
        struct bar_info info = {0};
        if(ioctl(this->fd, IOCTL_GET_BAR, &info) < 0){
            std::cout << "ioctl get bar fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }
        this->bar_size = info.size;

        // map the BAR
        this->bar = (volatile unsigned int *) mmap(NULL, this->bar_size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0); 
        if(this->bar == MAP_FAILED){
            std::cout << "mmap dev bar fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }

        // Initialize the NVMe controller
        this->CAP_DSTRD = CAP$DSTRD(this->bar);
        this->CAP_MQES = CAP$MQES(this->bar) + 1;
        this->CAP_TO = CAP$TO(this->bar);

        volatile uint32_t* cc = CC(this->bar);
        *cc = *cc & ~1;
        usleep(this->CAP_TO * 500);
        while (CSTS$RDY(this->bar) != 0){
            std::cout << "CSTS$RDY not ready\n";
            usleep(this->CAP_TO * 500);
        }

        // calculate admin queue size
        unsigned int cq_size = this->admin_q_depth * 4; // 4 bytes per CQ entry
        unsigned int sq_size = this->admin_q_depth * 16; // 16 bytes per SQ entry
        

        // allocate memory for admin queues
        this->acq_buf.size = cq_size * sizeof(unsigned int);
        if(ioctl(this->fd, IOCTL_ALLOCATE_DMA_BUFFER, &this->acq_buf) < 0){
            std::cout << "ioctl alloc acq buf fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }
        this->cq_ptr = (volatile unsigned int *) mmap(NULL, this->acq_buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0); // this->acq_buf.addr
        if(this->cq_ptr == MAP_FAILED){
            std::cout << "mmap admin cq fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }
        memset((void*)this->cq_ptr, 0, this->acq_buf.size);

        this->asq_buf.size = sq_size  * sizeof(unsigned int);
        if(ioctl(this->fd, IOCTL_ALLOCATE_DMA_BUFFER, &this->asq_buf) < 0){
            std::cout << "ioctl alloc asq buf fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }

        this->sq_ptr = (volatile unsigned int *) mmap(NULL, this->asq_buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0); // this->asq_buf.addr
        if(this->sq_ptr == MAP_FAILED){
            std::cout << "mmap admin queues fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }
        memset((void*)this->sq_ptr, 0, this->asq_buf.size);

        volatile uint32_t* aqa = AQA(this->bar);
        *aqa = AQA$AQS(this->admin_q_depth - 1) | AQA$AQC(this->admin_q_depth - 1);

        volatile uint64_t* acq = ACQ(this->bar);
        *acq = this->acq_buf.addr; 

        volatile uint64_t* asq = ASQ(this->bar);
        *asq = this->asq_buf.addr;

        
        *cc = CC$IOCQES(4) | CC$IOSQES(6) | CC$MPS(0) | CC$CSS(0) | CC$EN(1);
        usleep(this->CAP_TO * 500);
        while (CSTS$RDY(this->bar) != 1){
            std::cout << "CSTS$RDY not ready\n";
            usleep(this->CAP_TO * 500);
        }

        this->admin_sqdb = SQ_DBL(this->bar, 0, CAP_DSTRD);
        this->admin_cqdb = CQ_DBL(this->bar, 0, CAP_DSTRD);

        cuda_err_chk(cudaHostRegister(((void*)this->bar), this->bar_size, cudaHostRegisterIoMemory));
        for(unsigned int i = 0; i < this->admin_q_depth; ++i){
            for(unsigned int j = 0; j < 16; ++j){
                sq_ptr[i * 16 + j] = 0;
            }
            for(unsigned int j = 0; j < 4; ++j){
                cq_ptr[i * 4 + j] = 0;
            }
        }

        std::cout << "NVMe device opened: " << dev << std::endl;
        
        setQueueNums(queue_num);

#endif
    }

    __host__ void setQueueNums(unsigned int queue_num){
        printf("setQueueNums %d\n", queue_num);
        volatile unsigned int * cmd = sq_ptr + asq_pos * 16;

        cmd[0] = 0x09 | (asq_pos << 16);
        cmd[10] = 0x7;
        cmd[11] = ((queue_num - 1) << 16) | (queue_num - 1);
        asq_pos = (asq_pos + 1) % this->admin_q_depth;
        *admin_sqdb = asq_pos;
        this->wait_cpl();
    }


    __host__ void closeNvme(){
#if REGISTER_NVME
        volatile uint32_t* cc = CC(this->bar);
        *cc = *cc & ~1;
        usleep(this->CAP_TO * 500);
        while (CSTS$RDY(this->bar) != 0){
            std::cout << "CSTS$RDY not ready\n";
            usleep(this->CAP_TO * 500);
        }
        *cc = *cc & ~1;
        usleep(this->CAP_TO * 500);
        while (CSTS$RDY(this->bar) != 0){
            std::cout << "CSTS$RDY not ready\n";
            usleep(this->CAP_TO * 500);
        }
        cuda_err_chk(cudaHostUnregister(((void*)this->bar)));
        
        munmap(((void*)this->bar), this->bar_size);

        munmap((void*)this->cq_ptr, this->acq_buf.size);
        munmap((void*)this->sq_ptr, this->asq_buf.size);

        if(ioctl(this->fd, IOCTL_FREE_DMA_BUFFER, &this->asq_buf) < 0){
            std::cout << "ioctl free asq buf fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }
        if(ioctl(this->fd, IOCTL_FREE_DMA_BUFFER, &this->acq_buf) < 0){
            std::cout << "ioctl free acq buf fail: " << dev << std::endl;
            close(this->fd);
            exit(0);
        }

        close(this->fd);
#endif
    }

    __host__ ~NvmeConfig(){
        // printf("~NvmeConfig\n");
    }

    // __host__ void setCPUMem(void * cpu_dma_ptr, unsigned long cpu_dma_phy_addr, unsigned int cpu_dma_size){
    //     this->cpu_dma_ptr = cpu_dma_ptr;
    //     this->cpu_dma_phy_addr = cpu_dma_phy_addr;
    //     this->cpu_dma_size = cpu_dma_size;
    // }

    __host__ unsigned int getRequiredMemSize(){
        return queue_num * queue_depth * (64 + CQ_ALIGN); // TODO: check if CQ slot size can be 16
    }

    __host__ volatile unsigned int * getSQDB(unsigned int idx){
        return SQ_DBL(this->bar, idx + 1, this->CAP_DSTRD);
    }

    __host__ volatile unsigned int * getCQDB(unsigned int idx){
        return CQ_DBL(this->bar, idx + 1, this->CAP_DSTRD);
    }

    __host__ void wait_cpl(){
        volatile unsigned int * cpl = cq_ptr + acq_pos * 4;
        while(((cpl[3] >> 16) & 0x1) == this->phase){
            // std::cout << "wait_cpl: " << std::hex << cpl[0] << " " << cpl[1] << " " << cpl[2] << " " << cpl[3] << std::dec << std::endl;
            // usleep(1000);
        }
        if(((cpl[3] >> 17) & 0x1) != 0){
            std::cout << "Admin NVMe CPL: " << std::hex << cpl[0] << " " << cpl[1] << " " << cpl[2] << " " << cpl[3] << std::dec << std::endl;
        }

        // std::cout << "Admin NVMe CPL: " << std::hex << cpl[0] << " " << cpl[1] << " " << cpl[2] << " " << cpl[3] << std::dec << std::endl;
        if(((cpl[3] >> 17) & 0x1) != 0){
            exit(0);
        }

        acq_pos++;
        if(acq_pos == this->admin_q_depth){
            acq_pos = 0;
            this->phase = ~(this->phase) & 0x1;
        }
        *admin_cqdb = acq_pos;
    }

    // __host__ void deleteQueue(unsigned int idx){
    //     volatile unsigned int * cmd = sq_ptr + asq_pos * 16;

    //     // delete cq
    //     cmd[0] = 0x0 | (asq_pos << 16);
    //     cmd[1] = 0;
    //     cmd[10] = (idx + 1) & 0xffff;
    //     asq_pos = (asq_pos + 1) % this->admin_q_depth;
    //     *admin_sqdb = asq_pos;
    //     this->wait_cpl();

    //     // delete sq
    //     cmd = sq_ptr + asq_pos * 16;
    //     cmd[0] = 0x4 | (asq_pos << 16);
    //     cmd[1] = 0;
    //     cmd[10] = (idx + 1) & 0xffff;
    //     asq_pos = (asq_pos + 1) % this->admin_q_depth;
    //     *admin_sqdb = asq_pos;
    //     this->wait_cpl();
    // }

    __host__ void registerQueue(unsigned int idx, unsigned long sq_phy_addr, unsigned long cq_phy_addr, unsigned int *h_sq_ptr, unsigned int *h_cq_ptr){
        // std::cout << "registerQueue " << idx << std::endl;
        // printf("q %d this->queue_depth %d\n", idx + 1, this->queue_depth);
        // register cq
        // std::cout << "id: " << idx + 1 << " depth: " << this->queue_depth << " sq phy: " << std::hex << sq_phy_addr << " cq phy: " << cq_phy_addr << std::dec << std::endl;
        // deleteQueue(idx);
        volatile unsigned int * cmd = sq_ptr + asq_pos * 16;

        cmd[0] = 0x5 | (asq_pos << 16);
        cmd[6] = cq_phy_addr & 0xffffffff;
        cmd[7] = (cq_phy_addr >> 32) & 0xffffffff;
        cmd[10] = ((this->queue_depth - 1) << 16) | (idx + 1);
        cmd[11] = 0x1; 
        asq_pos = (asq_pos + 1) % this->admin_q_depth;
        *admin_sqdb = asq_pos;
        this->wait_cpl();


        // register sq
        cmd = sq_ptr + asq_pos * 16;
        cmd[0] = 0x1 | (asq_pos << 16);
        cmd[6] = sq_phy_addr & 0xffffffff;
        cmd[7] = (sq_phy_addr >> 32) & 0xffffffff;
        cmd[10] = ((this->queue_depth - 1) << 16) | (idx + 1);
        cmd[11] = 0x1 | (idx + 1) << 16;
        asq_pos = (asq_pos + 1) % this->admin_q_depth;
        *admin_sqdb = asq_pos;
        this->wait_cpl();

        for(unsigned int i = 0; i < this->queue_depth; ++i){
            for(unsigned int j = 0; j < 16; ++j){
                h_sq_ptr[i * 16 + j] = 0;
            }
            for(unsigned int j = 0; j < 4; ++j){
                h_cq_ptr[i * 4 + j] = 0;
            }

        }

    }


};

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__global__ void agile_evict_cache_to_nvme(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl);

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__global__ void start_agile_cq_service(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl);


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
class AgileHost {
    std::thread monitorThread;
    std::atomic<bool> stopFlag;

    std::vector<NvmeConfig> nvme_dev;
    GPUCacheImpl * h_gpu_cache, * d_gpu_cache;
    CPUCacheImpl * h_cpu_cache, * d_cpu_cache;
    ShareTableImpl * h_write_table, * d_write_table;

    AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * h_ctrl, * d_ctrl;
    AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * h_hierarchy, * d_hierarchy;
    AgilePollingList * h_pollingList, * d_pollingList;
    AgileNvmeDev * h_dev, * d_dev;

    
    unsigned int compute_blocks;
    unsigned int threads_per_block;
    unsigned int agile_blocks;

    unsigned int gpu_device_idx;

    CUdeviceptr d_gpu_ptr;
    void * h_gpu_ptr;

    // int agile_buffer_dev;

    // void * h_cpu_ptr;
    // void * d_cpu_ptr;

    // int cpu_mem_fd;
    // void * cpu_mem_ptr;
    // unsigned long cpu_mem_offset;  // maintain the 64K aglinment of the cpu memory

    // unsigned long * cpu_mem_table;
    // unsigned int cpu_mem_table_size;
    

    // void * h_cpu_cache_data_ptr;
    
    // void ** h_cpu_cache_data_ptr;
    // unsigned long *h_cpu_cache_table;
    // void ** d_cpu_cache_data_ptr;
    // unsigned long *d_cpu_cache_table;
    // unsigned int h_cpu_cache_table_size;

    unsigned int run;
    unsigned int total_pairs;
    cudaStream_t agile_cq;

public:
#if ENABLE_LOGGING
    AgileLogger *h_logger;
#endif
    unsigned int block_size;

    __host__ AgileHost(unsigned int gpu_device_idx, unsigned int block_size) : block_size(block_size), gpu_device_idx(gpu_device_idx) {
        cuda_err_chk(cudaMalloc(&(this->d_ctrl), sizeof(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>)));
        cuda_err_chk(cudaMalloc(&(this->d_hierarchy), sizeof(AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>)));
        cuda_err_chk(cudaMalloc(&(this->d_gpu_cache), sizeof(GPUCacheImpl)));
        cuda_err_chk(cudaMalloc(&(this->d_write_table), sizeof(ShareTableImpl)));
        cuda_err_chk(cudaMalloc(&(this->d_cpu_cache), sizeof(CPUCacheImpl)));
        
        // cuda_err_chk(cudaMalloc(&(agile_stop_signal), sizeof(unsigned int)));
        this->run = 0;
        this->total_pairs = 0;
        // init ctrl
        h_ctrl = new AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>;
        h_ctrl->buf_size = block_size;

        // init cache hierarchy
        h_hierarchy = new AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>;

        // this->cpu_mem_fd = open("/dev/agile_mem", O_RDWR);
        // if(this->cpu_mem_fd == -1){
        //     std::cerr << "open /dev/agile_mem fail" << std::endl;
        //     exit(0);
        // }

        // this->cpu_mem_ptr = (unsigned int *) mmap(NULL, CPU_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, this->cpu_mem_fd, 0);
        // if(this->cpu_mem_ptr == MAP_FAILED){
        //     std::cerr << "mmap /dev/agile_mem fail" << std::endl;
        //     exit(0);
        // }

        // this->cpu_mem_offset = 0;
        // cuda_err_chk(cudaHostRegister(this->cpu_mem_ptr, CPU_MEM_SIZE, cudaHostRegisterIoMemory));

        // printf("cpu memory addr: %p\n", this->cpu_mem_ptr);
        // printf("data %d\n", ((unsigned int*)this->cpu_mem_ptr)[0]);

    }

    __host__ void allocateBuffer(AgileBuf *& buf, unsigned int num){
        AgileBuf * h_buf = (AgileBuf *) malloc(sizeof(AgileBuf) * num);
        cuda_err_chk(cudaMalloc(&(buf), sizeof(AgileBuf) * num));
        for(unsigned int i = 0; i < num; ++i){
            new (&h_buf[i]) AgileBuf(this->block_size);
        }
        cuda_err_chk(cudaMemcpy(buf, h_buf, sizeof(AgileBuf) * num, cudaMemcpyHostToDevice));
        free(h_buf);
    }

    __host__ void freeBuffer(AgileBuf *& buf, unsigned int num){
        AgileBuf * h_buf = (AgileBuf *) malloc(sizeof(AgileBuf) * num);
        cuda_err_chk(cudaMemcpy(h_buf, buf, sizeof(AgileBuf) * num, cudaMemcpyDeviceToHost));
        for(unsigned int i = 0; i < num; ++i){
            cuda_err_chk(cudaFree(h_buf[i].data));
        }
        free(h_buf);
        cuda_err_chk(cudaFree(buf));
    }

    __host__ void appendBuf2File(std::string filepath, AgileBuf *& buf, unsigned int num){
        std::ofstream file(filepath, std::ios::binary | std::ios::app); // Open file in binary mode
        if (!file) {
            std::cerr << "Error opening file!" << std::endl;
            exit(0);
        }

        AgileBuf *h_buf = (AgileBuf *) malloc(sizeof(AgileBuf) * num);
        cuda_err_chk(cudaMemcpy(h_buf, buf, sizeof(AgileBuf) * num, cudaMemcpyDeviceToHost));
        void * h_data = (void *) malloc(this->block_size);
        for(unsigned int i = 0; i < num; ++i){
            cuda_err_chk(cudaMemcpy(h_data, h_buf[i].data, this->block_size, cudaMemcpyDeviceToHost));
            file.write((char *)h_data, this->block_size);
        }
        free(h_data);
        free(h_buf);
        file.close();
    }

    __host__ void loadFile2Buf(std::string filepath, unsigned long offset, AgileBuf *& buf, unsigned int num){
        std::ifstream file(filepath, std::ios::binary); // Open file in binary mode
        if (!file) {
            std::cerr << "Error opening file!" << std::endl;
            exit(0);
        }

        file.seekg(offset * this->block_size, std::ios::beg);
        AgileBuf * h_buf = (AgileBuf *) malloc(sizeof(AgileBuf) * num);
        cuda_err_chk(cudaMemcpy(h_buf, buf, sizeof(AgileBuf) * num, cudaMemcpyDeviceToHost));
        void * h_data = (void *) malloc(this->block_size);
        for(unsigned int i = 0; i < num; ++i){
            unsigned finished_size = 0;
            while(finished_size < this->block_size){
                file.read((char *)h_data + finished_size, this->block_size - finished_size);
                finished_size += file.gcount();
            }
            cuda_err_chk(cudaMemcpy(h_buf[i].data, h_data, this->block_size, cudaMemcpyHostToDevice));
        }
        free(h_data);
    }

    // can only be called once

    // get CPU pinned memory for the host with 64K alignment
    __host__ unsigned long getCPUPinnedMem(void *& ptr, unsigned long size){
        if(this->cpu_mem_offset + size > CPU_MEM_SIZE){
            std::cerr << "cpu memory is not enough" << std::endl;
            exit(0);
        }

        ptr = (((char *)this->cpu_mem_ptr) + this->cpu_mem_offset);
        unsigned long cpu_physical_addr = CPU_MEM_ADDR + this->cpu_mem_offset;
        // printf("cpu_physical_addr: %lx ptr: %p\n", cpu_physical_addr, ptr);
        this->cpu_mem_offset += size;
        this->cpu_mem_offset = (this->cpu_mem_offset + 65535) & ~65535;
        return cpu_physical_addr;
    }

    __host__ void initNvme(){
        for(unsigned int i = 0; i < this->nvme_dev.size(); ++i){
            this->nvme_dev[i].openNvme();
        }
    }

#if FAKE_NVME
    void * emuNmve;
    unsigned int emu_size;

    __host__ void setEmuNvme(unsigned long total_size){
        getCPUPinnedMem(emuNmve, total_size);
        printf("emuNmve: %p\n", emuNmve);
        emu_size = total_size;
        this->h_ctrl->emu_nvme = emuNmve;
        this->h_ctrl->emu_size = emu_size;
        // printf("allocate cpu memory for emulation block: %d\n", emu_size);
        // this->emu_size = emu_size;
        // assert(total_size % emu_size == 0);
        // this->emu_num_total = total_size / emu_size;
        // this->emu_num_per_dev = this->emu_num_total / dev_num;
        // emuNmve = (void **) malloc(sizeof(void *) * this->emu_num_total);

        // for(unsigned int i = 0; i < this->emu_num_total; ++i){
        //     void * ptr;
        //     unsigned long cpu_physical_addr = allocateCPUPinedMem(this->agile_buffer_dev, ptr, emu_size);
        //     cuda_err_chk(cudaHostRegister(ptr, emu_size, cudaHostRegisterMapped));
        //     emuNmve[i] = ptr;
            
        // }
        // cuda_err_chk(cudaMalloc(&(h_ctrl->emu_nvme), sizeof(void *) * this->emu_num_total));
        // cuda_err_chk(cudaMemcpy(h_ctrl->emu_nvme, emuNmve, sizeof(void *) * this->emu_num_total, cudaMemcpyHostToDevice));

        // h_ctrl->emu_num_total = this->emu_num_total;
        // h_ctrl->emu_num_per_dev = this->emu_num_per_dev;
        // h_ctrl->emu_size = emu_size;
        // printf("end cpu memory allocation\n");
    }

    __host__ unsigned int loadFile2Emu(std::string filepath, unsigned int block_offset){
        std::cout << "loading file...\n";
        std::ifstream file(filepath, std::ios::binary); // Open file in binary mode
        if (!file) {
            std::cerr << "Error opening file!" << std::endl;
            exit(0);
        }

        file.seekg(0, std::ios::end);  // Move to the end of the file
        std::streamsize size = file.tellg();  // Get file size
        file.seekg(0, std::ios::beg);  // Move back to the beginning

        unsigned long finish_size = 0;
        
        // load the file to the emuNvme
        std::cout << "loading file... size: " << size << std::endl;
        // for(unsigned int i = 0; i < 10; i++){
        //     ((char*)(emuNmve))[i] = file.get();
        // }
        // for(unsigned int i = 0; i < 10; i++){
        //     std::cout << ((unsigned int*)(emuNmve))[i] << std::endl;
        // }
        while(finish_size < size){

            file.read(((char*)(emuNmve) + block_offset * this->block_size + finish_size), 4096);
            finish_size += file.gcount();
            std::cout << "finish_size: " << finish_size << " total: " << size << " per: " << ((float)finish_size) / ((float) size) << std::endl;
        }


        // unsigned int start_idx = (block_offset * this->block_size) / emu_size;
        // unsigned int start_offset = (block_offset * this->block_size) % emu_size;

        // while(finish_size < size){
            
        //     unsigned int temp_size = MIN(emu_size - start_offset, size - finish_size);
        //     unsigned int temp_sum = 0;
        //     while(temp_sum < temp_size){
        //         file.read(((char*)(emuNmve[start_idx] + start_offset + temp_sum)), temp_size - temp_sum);
        //         temp_sum += file.gcount();
        //         finish_size += temp_sum;
        //     }
            
        //     start_offset = 0;
        //     start_idx++;
        // }
        return finish_size;
    }

    template<typename T>
    __host__ unsigned int saveEmu2File(std::string filepath, unsigned int block_offset, unsigned int datasize){
        std::ofstream outFile(filepath);
        if (!outFile) {
            std::cerr << "Failed to open file for writing.\n";
            return 1;
        }

        unsigned long finish_size = 0;
        while(finish_size < datasize){
            outFile << static_cast<T*>((emuNmve) + block_offset * this->block_size + finish_size) << std::endl;
            finish_size += sizeof(T);
        }

        // unsigned int start_idx = (block_offset * this->block_size) / emu_size;
        // unsigned int start_offset = ((block_offset * this->block_size) % emu_size) / sizeof(T);

        // unsigned int block_num = datasize * sizeof(T) / emu_size + (start_offset == 0 ? 0 : 1);
        // unsigned int count = 0;
        // for(unsigned int i = 0; i < block_num; ++i){
        //     for(unsigned int j = start_offset; j < emu_size / sizeof(T) && count < datasize; ++j, ++count){
        //         outFile << static_cast<T*>((emuNmve[start_idx + i]))[j] << std::endl;
        //     }
        //     start_offset = 0;
        // }
        // return 0;
    }

#endif

    // __host__ void initCPUCache(){

    //     this->h_cpu_cache->physical_addr = getCPUPinnedMem(this->h_cpu_cache->data, this->h_cpu_cache->getRequiredMemSize());
    //     printf("cpu cache physical addr: %lx ptr: %p\n", this->h_cpu_cache->physical_addr, this->h_cpu_cache->data);

    // }

    __host__ void initializeAgile(){
        // allocate memory
        std::cout << "allocating GPU pinned memory\n";
        unsigned long gpu_physical_addr = allocateGPUPinedMem(this->gpu_device_idx, this->getGPUPinnedMemSize(), this->d_gpu_ptr, this->h_gpu_ptr);
        
        // std::cout << "allocating CPU pinned memory\n";
        // this->initCPUCache();
        // assign pointers
        
        /*******copy to device******/
        
        // get total queue pair num
        
        
        for(unsigned int i = 0; i < nvme_dev.size(); ++i){
            total_pairs += nvme_dev[i].queue_num;
        }

        // std::cout << "Agile initializaing\n";

        this->h_pollingList = (AgilePollingList *) malloc(sizeof(AgilePollingList));
        this->h_pollingList->pairs = (AgileQueuePair *) malloc(sizeof(AgileQueuePair) * total_pairs);
        this->h_pollingList->num_pairs = total_pairs;

        // init queues for each dev
        this->h_dev = (AgileNvmeDev *) malloc(sizeof(AgileNvmeDev) * this->nvme_dev.size());
        cuda_err_chk(cudaMalloc(&(this->d_dev), sizeof(AgileNvmeDev) * this->nvme_dev.size()));
        
        unsigned int dev_id = 0;
        unsigned long gpu_offset = 0;
        unsigned int queue_id = 0;
        unsigned int pair_offset = 0;
        
        AgileQueuePair * d_total_pairs;
        cuda_err_chk(cudaMalloc(&d_total_pairs, sizeof(AgileQueuePair) * total_pairs));
        for(unsigned int i = 0; i < nvme_dev.size(); ++i){
            this->h_dev[dev_id] = AgileNvmeDev(nvme_dev[i].queue_num, nvme_dev[i].queue_depth);
            this->h_dev[dev_id].pairs = (AgileQueuePair *) malloc(sizeof(AgileQueuePair) * nvme_dev[i].queue_num);
            for(int j = 0; j < nvme_dev[i].queue_num; ++j){
                unsigned long sq_offset = gpu_offset;
                unsigned long cq_offset = sq_offset + 64 * nvme_dev[i].queue_depth;

                volatile unsigned int *sqdb = nvme_dev[i].getSQDB(j);
                volatile unsigned int *csqb = nvme_dev[i].getCQDB(j);
                // cuda_err_chk(cudaHostRegister(((void*)sqdb), sizeof(unsigned int), cudaHostRegisterIoMemory));
                // cuda_err_chk(cudaHostRegister(((void*)csqb), sizeof(unsigned int), cudaHostRegisterIoMemory));
#if DEBUG_NVME
                AgileQueuePair temp_pairs(((void*)this->d_gpu_ptr) + sq_offset, ((void*)this->d_gpu_ptr) + cq_offset, sqdb, csqb, nvme_dev[i].queue_depth, j, nvme_dev[i].ssd_blk_offset * this->block_size / 512);
#else
                AgileQueuePair temp_pairs(((void*)this->d_gpu_ptr) + sq_offset, ((void*)this->d_gpu_ptr) + cq_offset, sqdb, csqb, nvme_dev[i].queue_depth, nvme_dev[i].ssd_blk_offset * this->block_size / 512);
#endif


                nvme_dev[i].registerQueue(j, gpu_physical_addr + gpu_offset, gpu_physical_addr + cq_offset, (unsigned int *)(this->h_gpu_ptr + gpu_offset), (unsigned int *)(this->h_gpu_ptr + cq_offset));
                
                h_dev[dev_id].pairs[j] = temp_pairs;
                this->h_pollingList->pairs[queue_id] = temp_pairs;
                gpu_offset += (64 + CQ_ALIGN) * nvme_dev[i].queue_depth; // TODO: auto alignment
                queue_id++;
            }

            ///////////////////////////////////
            // AgileLock * h_lock7 = (AgileLock *) malloc(sizeof(AgileLock) * dev.queue_num);
            // printf("this->h_dev[dev_id].pairs[7].sq.cmd_locks: %p val: %d\n", this->h_dev[dev_id].pairs[7].sq.cmd_locks, this->h_dev[dev_id].pairs[7].sq.cmd_locks[0].lock);
            // cuda_err_chk(cudaMemcpy(h_lock7, this->h_dev[dev_id].pairs[7].sq.cmd_locks, sizeof(AgileLock) * dev.queue_num, cudaMemcpyDeviceToHost));
            // printf("lock ptr: %d\n", (h_lock7[0].lock));
            ///////////////////////////////////

            AgileQueuePair * d_pairs;
            cuda_err_chk(cudaMalloc(&d_pairs, sizeof(AgileQueuePair) * nvme_dev[i].queue_num));
            cuda_err_chk(cudaMemcpy(d_pairs, this->h_dev[dev_id].pairs, sizeof(AgileQueuePair) * nvme_dev[i].queue_num, cudaMemcpyHostToDevice));
            free(this->h_dev[dev_id].pairs);
            this->h_dev[dev_id].pairs = d_total_pairs + pair_offset;
            pair_offset += nvme_dev[i].queue_num;
            dev_id++;
        }

        
        // init polling list for Agile service
        
        cuda_err_chk(cudaMemcpy(d_total_pairs, this->h_pollingList->pairs, sizeof(AgileQueuePair) * total_pairs, cudaMemcpyHostToDevice));
        this->h_pollingList->pairs = d_total_pairs;
        cuda_err_chk(cudaMalloc(&(this->d_pollingList), sizeof(AgilePollingList)));
        cuda_err_chk(cudaMemcpy(this->d_pollingList, this->h_pollingList, sizeof(AgilePollingList), cudaMemcpyHostToDevice));

        // set GPU memory for gpu cache
        this->h_gpu_cache->setPinedMem(((void *)this->d_gpu_ptr + gpu_offset), gpu_physical_addr + gpu_offset);
        // std::cout << "gpu cache addr: " << std::hex << gpu_physical_addr + gpu_offset << std::dec << std::endl;
        gpu_offset += this->h_gpu_cache->getRequiredMemSize();
        // set device pointers
        this->h_ctrl->cache_hierarchy = this->d_hierarchy;
#if ENABLE_LOGGING
        AgileLogger * d_logger_ptr = ((AgileLogger *) ((void *)this->d_gpu_ptr + gpu_offset));
        cuda_err_chk(cudaMemcpyToSymbol(logger, &d_logger_ptr, sizeof(AgileLogger *)));
        this->h_logger = ((AgileLogger *) ((void *)this->h_gpu_ptr + gpu_offset));
#endif
        cuda_err_chk(cudaMemcpy(d_dev, h_dev, sizeof(AgileNvmeDev) * this->nvme_dev.size(), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMallocManaged(&(this->h_ctrl->stop_signal), sizeof(unsigned int)));
        this->h_ctrl->dev = d_dev;
        this->h_ctrl->list = this->d_pollingList;
        this->h_hierarchy->gpu_cache = this->d_gpu_cache;
        this->h_hierarchy->cpu_cache = this->d_cpu_cache;
        this->h_hierarchy->ctrl = this->d_ctrl;
        this->h_hierarchy->share_table = this->d_write_table;
        this->h_gpu_cache->cache_hierarchy = this->d_hierarchy;
        this->h_cpu_cache->cache_hierarchy = this->d_hierarchy;

        
        // copy obj to device
        cuda_err_chk(cudaMemcpy(this->d_hierarchy, this->h_hierarchy, sizeof(AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->d_ctrl, this->h_ctrl, sizeof(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->d_gpu_cache, this->h_gpu_cache, sizeof(GPUCacheImpl), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->d_write_table, this->h_write_table, sizeof(ShareTableImpl), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->d_cpu_cache, this->h_cpu_cache, sizeof(CPUCacheImpl), cudaMemcpyHostToDevice));
        

    }

    __host__ ~AgileHost(){
#if FAKE_NVME
        // for(unsigned int i = 0; i < this->emu_num_total; ++i){
        //     munmap(emuNmve[i], emu_size);
        // }
        // free(emuNmve);
#endif
        cuda_err_chk(cudaFree(this->d_ctrl));
        cuda_err_chk(cudaFree(this->d_hierarchy));
        cuda_err_chk(cudaFree(this->d_gpu_cache));
        cuda_err_chk(cudaFree(this->d_cpu_cache));

        // munmap(this->h_cpu_ptr, this->getCPUPinnedMemSize());
        // for(unsigned int i = 0; i < this->h_cpu_cache_table_size; ++i){
        //     munmap(this->h_cpu_cache_data_ptr[i], (1024*1024*4));
        // }
        // free(this->h_cpu_cache_table);
        // free(this->h_cpu_cache_data_ptr);

        delete h_hierarchy;
        delete h_ctrl;

        // for(unsigned int i = 0; i < this->h_cpu_cache_table_size; ++i){

        // }
        // close(this->agile_buffer_dev);
        // cuda_err_chk(cudaHostUnregister(this->cpu_mem_ptr));
        // munmap(this->cpu_mem_ptr, CPU_MEM_SIZE);
        // close(this->cpu_mem_fd);
    }

    __host__ void setShareTable(ShareTableImpl & write_table){
        this->h_write_table = &write_table;
    }
    
    __host__ void setGPUCache(GPUCacheImpl & gpu_cache){
        this->h_gpu_cache = &gpu_cache;
        assert(this->block_size == gpu_cache.slot_size);
    }

    __host__ void setCPUCache(CPUCacheImpl & cpu_cache){
        this->h_cpu_cache = &cpu_cache;
        assert(this->block_size == cpu_cache.slot_size);
    }

    [[deprecated]]
    __host__ void addNvmeDev(std::string dev, unsigned int bar_size, SSDBLK_TYPE block_offset, unsigned int queue_num, unsigned int queue_depth){
        this->nvme_dev.emplace_back(dev, bar_size, queue_num, queue_depth, block_offset);
    }

    __host__ void addNvmeDev(std::string dev, SSDBLK_TYPE block_offset, unsigned int queue_num, unsigned int queue_depth){
        this->nvme_dev.push_back(NvmeConfig(dev, block_offset, queue_num, queue_depth));
    }

    __host__ void closeNvme(){
        for(unsigned int i = 0; i < this->nvme_dev.size(); ++i){
            this->nvme_dev[i].closeNvme();
        }
    }
    __host__ unsigned int getGPUPinnedMemSize(){
        // calculate memory size for cache

#if ENABLE_LOGGING
        unsigned int required_pinned_gpu_mem_size = this->h_gpu_cache->getRequiredMemSize() + sizeof(AgileLogger);
#else
        unsigned int required_pinned_gpu_mem_size = this->h_gpu_cache->getRequiredMemSize();
#endif
        // calculate memory size for on device nvme queue
        for(unsigned int i = 0; i < this->nvme_dev.size(); ++i){
            required_pinned_gpu_mem_size += nvme_dev[i].getRequiredMemSize();
        }
        return required_pinned_gpu_mem_size;
    }


    __host__ AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> * getAgileCtrlDevicePtr(){
        return this->d_ctrl;
    }

    __host__ void configParallelism(BID_TYPE compute_blocks, TID_TYPE threads_per_block, BID_TYPE agile_blocks){
        this->h_ctrl->setComputeBlocks(compute_blocks);
        this->h_ctrl->setThreadsPreBlock(threads_per_block);
        this->h_ctrl->setAgileBlocks(agile_blocks);
        this->agile_blocks = agile_blocks;
        this->threads_per_block = threads_per_block;
        this->compute_blocks = compute_blocks;
        cuda_err_chk(cudaMemcpy(this->d_ctrl, this->h_ctrl, sizeof(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl>), cudaMemcpyHostToDevice));
    }

#if ENABLE_LOGGING
    __host__ void monitoring(){
        std::cout << "run: " << this->run << " prefetch_hit: " << this->h_logger->prefetch_hit << " prefetch_relaxed_hit: " << this->h_logger->prefetch_relaxed_hit << " prefetch_relaxed_miss: " << this->h_logger->prefetch_relaxed_miss << " prefetch_issue: " << this->h_logger->prefetch_issue << " runtime_issue: " << this->h_logger->runtime_issue << " warp_master_wait: " << this->h_logger->warp_master_wait << std::endl;
        std::cout << "issued_read: " << this->h_logger->issued_read << " issued_write: " << this->h_logger->issued_write << " attempt_fail: " << this->h_logger->attempt_fail << std::endl;
        std::cout << "finished_read: " << this->h_logger->finished_read << " diff: " << this->h_logger->issued_read - this->h_logger->finished_read << " finished_write: " << this->h_logger->finished_write << std::endl;
        std::cout << "find_new_cacheline: " << this->h_logger->find_new_cacheline << " gpu_cache_hit: " << this->h_logger->gpu_cache_hit << " gpu2cpu: " << this->h_logger->gpu2cpu << " gpu_cache_miss: " << this->h_logger->gpu_cache_miss << " gpu_cache_evict: " << this->h_logger->gpu_cache_evict << std::endl;
        std::cout << "cpu_cache_hit: " << this->h_logger->cpu_cache_hit << " cpu2buf: " << this->h_logger->cpu2buf << " cpu2gpu: " << this->h_logger->cpu2gpu << " cpu_cache_miss: " << this->h_logger->cpu_cache_miss << " cpu_cache_evict: " << this->h_logger->cpu_cache_evict << std::endl;
        std::cout << "finished_block: " << this->h_logger->finished_block << " finished_agile_warp: " << this->h_logger->finished_agile_warp << std::endl;
        std::cout << "service: " << this->h_logger->service << " wait buffer: " << this->h_logger->wating_buffer << " finish buffer: " << this->h_logger->finish_buffer << " local hit: " << this->h_logger->buffer_localhit << std::endl;
        std::cout << "waiting: " << this->h_logger->waiting << " waitTooMany: " << this->h_logger->waitTooMany << " deadlock_check: " << this->h_logger->deadlock_check << std::endl;
        std::cout << "propogate_time: " << this->h_logger->propogate_time << " appendbuf_count: " << this->h_logger->appendbuf_count << " propogate_count: " << this->h_logger->propogate_count << " self_propagate: " << this->h_logger->self_propagate << std::endl;
        std::cout << "push in table: " << this->h_logger->push_in_table << " pop out table count: " << this->h_logger->pop_in_table << std::endl;

        // for(int i = 0; i < 64; ++i){
        //     unsigned idx = i * 2;
        //     std::cout << idx << " last sq: " << this->h_logger->last_sq_pos[idx] << " curr sq: " << this->h_logger->curr_sq_pos[idx] << " last cq: " << this->h_logger->last_cq_pos[idx] << " curr cq: " << this->h_logger->curr_cq_pos[idx] << " waiting: " << this->h_logger->cq_waiting[idx] << " gap: " << this->h_logger->gap[idx] << " running " << this->h_logger->cq_running[idx] << " \t";
        //     idx++;
        //     std::cout << idx << " last sq: " << this->h_logger->last_sq_pos[idx] << " curr sq: " << this->h_logger->curr_sq_pos[idx] << " last cq: " << this->h_logger->last_cq_pos[idx] << " curr cq: " << this->h_logger->curr_cq_pos[idx] << " waiting: " << this->h_logger->cq_waiting[idx] << " gap: " << this->h_logger->gap[idx] << " running " << this->h_logger->cq_running[idx] << std::endl;
        // }
    }
#endif

    __host__ void startAgile(){
        // unsigned int stop = 0;
        // cuda_err_chk(cudaMemcpy(agile_stop_signal, &stop, sizeof(unsigned int), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaStreamCreateWithFlags(&(this->agile_cq), cudaStreamNonBlocking));
        *((volatile unsigned int*)this->h_ctrl->stop_signal) = 0;
        unsigned int warps = this->total_pairs;
        unsigned int threads = warps * 32;
        unsigned int blocks = threads / 1024 + (threads % 1024 == 0 ? 0 : 1);
        std::cout << "agile blocks: " << blocks << " threads: " << threads << std::endl;
        start_agile_cq_service<<<blocks, min(threads, 1024), 0, this->agile_cq>>>(this->d_ctrl);
        usleep(100);
    }
    
    __host__ void startAgile(unsigned int griddim, unsigned int blockdim){
        cuda_err_chk(cudaStreamCreateWithFlags(&(this->agile_cq), cudaStreamNonBlocking));
        *((volatile unsigned int*)this->h_ctrl->stop_signal) = 0;
        std::cout << "agile blocks: " << griddim << " threads: " << blockdim << std::endl;
        start_agile_cq_service<<<griddim, blockdim, 0, this->agile_cq>>>(this->d_ctrl);
        usleep(100);

    }
    __host__ void stopAgile(){
        *((volatile unsigned int*)this->h_ctrl->stop_signal) = 1;
        cuda_err_chk(cudaStreamSynchronize(this->agile_cq));
        cuda_err_chk(cudaStreamDestroy(this->agile_cq));
#if ENABLE_LOGGING
        this->monitoring();
#endif
    }


    template <typename Func>
    __host__ int queryOccupancy(Func kernel, unsigned int blockSize, unsigned int dynamicSmemSize) {
       int numBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSM,
            kernel,
            blockSize, // threads per block
            dynamicSmemSize    // dynamic shared memory
        );
        return numBlocksPerSM;
    }

    template <typename Func, typename... Args>
    __host__ void runKernel(Func kernel, Args... args) {

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        kernel<<<compute_blocks, threads_per_block, 0, stream>>>(args...);
#if ENABLE_LOGGING && SHOW_LOGGING
        while (true) {
            cudaError_t err = cudaStreamQuery(stream);
            if (err == cudaSuccess) {
                break;
            } else if (err == cudaErrorNotReady) {
                std::cout << "\033[2J\033[H";
                std::cout.flush();
                unsigned int line = 8 ;
                std::cout << "\033[" << line << "H"; 
                this->monitoring();
            } else {
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Non-blocking wait
        }
        cuda_err_chk(cudaStreamDestroy(stream));
        this->run++;
        this->monitoring();
#else
        cuda_err_chk(cudaStreamSynchronize(stream));
        cuda_err_chk(cudaStreamDestroy(stream));
#endif    
    }

    template <typename Func, typename... Args>
    __host__ void runKernel(Func kernel, dim3 gridDim, dim3 blockDim, Args... args) {

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        kernel<<<gridDim, blockDim, 0, stream>>>(args...);
#if ENABLE_LOGGING && SHOW_LOGGING
        while (true) {
            cudaError_t err = cudaStreamQuery(stream);
            if (err == cudaSuccess) {
                break;
            } else if (err == cudaErrorNotReady) {
                std::cout << "\033[2J\033[H";
                std::cout.flush();
                unsigned int line = 8 ;
                std::cout << "\033[" << line << "H"; 
                this->monitoring();
            } else {
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Non-blocking wait
        }
        cuda_err_chk(cudaStreamDestroy(stream));
        this->run++;
        this->monitoring();
#else
        cuda_err_chk(cudaStreamSynchronize(stream));
        cuda_err_chk(cudaStreamDestroy(stream));
#endif    
    }

    __host__ void applyCacheWrite2Nvme(){
        std::cout << "HOST: start applyCacheWrite2Nvme()\n";
        agile_evict_cache_to_nvme<<<compute_blocks + agile_blocks, threads_per_block>>>(this->d_ctrl);
        cuda_err_chk(cudaDeviceSynchronize());
        std::cout << "HOST: finish applyCacheWrite2Nvme()\n";
#if ENABLE_LOGGING
        this->monitoring();
#endif
    }

};


#define START_AIGLE_WRITE(ctrl) __shared__ unsigned int AGILE_BID; bool flag = ctrl->startAgile(AGILE_BID); if (flag) {
#define END_AIGLE_WRITE(ctrl) ctrl->stopAgile(AGILE_BID);} if(!flag) {ctrl->pollingServiceLast(AGILE_BID);}

template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__global__ void agile_evict_cache_to_nvme(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl){
    unsigned int AGILE_BID = blockIdx.x;
    if(AGILE_BID == 0){
        
        AgileLockChain chain;
        unsigned int tid = threadIdx.x;
        for(unsigned int i = tid; i < ctrl->cache_hierarchy->gpu_cache->slot_size; i += blockDim.x){
            ctrl->cache_hierarchy->gpu_cache->acquireBaseLock_lockStart(i, &chain);
            if(ctrl->cache_hierarchy->gpu_cache->getStatus_inLockArea(i) == AGILE_GPUCACHE_MODIFIED){
                unsigned int dev_idx = -1;
                unsigned int blk_idx = -1;
                static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(ctrl->cache_hierarchy)->getGPUCacheBasePtr()->getTaginfo_inLockArea(&dev_idx, &blk_idx, i);
                static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(ctrl->cache_hierarchy)->evictGPU2Nvme_inLockArea(dev_idx, blk_idx, i, &chain);
            }
            ctrl->cache_hierarchy->gpu_cache->releaseBaseLock_lockEnd(i, &chain);
        }
     
        __syncthreads();
    
        for(unsigned int i = tid; i < ctrl->cache_hierarchy->cpu_cache->slot_size; i += blockDim.x){
            ctrl->cache_hierarchy->cpu_cache->acquireBaseLock_lockStart(i, &chain);
            unsigned int stat = ctrl->cache_hierarchy->cpu_cache->getStatus_inLockArea(i);
            if(stat == AGILE_CPUCACHE_MODIFIED){
                unsigned int dev_idx = -1;
                unsigned int blk_idx = -1;
                static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(ctrl->cache_hierarchy)->getCPUCacheBasePtr()->getTaginfo_inLockArea(&dev_idx, &blk_idx, i);
                bool inprocessing;
                static_cast<AgileCacheHierarchy<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *>(ctrl->cache_hierarchy)->evictCPUCache_inLockArea(dev_idx, blk_idx, i, &inprocessing, &chain);
            }
            ctrl->cache_hierarchy->cpu_cache->releaseBaseLock_lockEnd(i, &chain);
        }
    }
}


template <typename GPUCacheImpl, typename CPUCacheImpl, typename ShareTableImpl>
__global__ void start_agile_cq_service(AgileCtrl<GPUCacheImpl, CPUCacheImpl, ShareTableImpl> *ctrl){
    ctrl->pollingService2();
}

#endif