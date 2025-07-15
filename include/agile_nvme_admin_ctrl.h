#include <stdio.h>
#include "nvme_reg_help.h"
// #include "config.h"
#include <iostream>
typedef struct __attribute__((aligned(16))) __dma_buffer__ {
    void *vir_addr;
    unsigned long phy_addr;
} dma_buffer;

typedef struct __attribute__((aligned(16))) __sqe__{
    unsigned int cmd[16];
} sqe;

typedef struct __attribute__((aligned(16))) __cqe__{
    unsigned int cpl[4];
} cqe;

typedef struct __attribute__((aligned(16))) __nvme_cq__ {
    dma_buffer buf;
    cqe * entity;
    unsigned int element_size; // element_size == 16 if it is SQ else 4
    unsigned int head_idx;
    unsigned int tail_idx;
    unsigned int qsize;
    int phase;
    volatile unsigned int *db; // pointer to nvme SQ/CQ doorbell register
} nvme_cq;

typedef struct __attribute__((aligned(16))) __nvme_sq__ {
    dma_buffer buf;
    sqe * entity;
    unsigned int element_size; // element_size == 16 if it is SQ else 4
    unsigned int head_idx;
    unsigned int tail_idx;
    unsigned int qsize;
    unsigned char identifier = 0;
    volatile unsigned int *db; // pointer to nvme SQ/CQ doorbell register
} nvme_sq;

typedef struct __attribute__((aligned(16))) __nvme_queue_pair__ {
    nvme_cq cq;
    nvme_sq sq;
    unsigned int pair_idx;
} nvme_queue_pair;

typedef struct __attribute__((aligned(16))) __nvme_admin_ctrl__ {
    int fd_dma; // the file descriptor for /dev/dma_buffer
    int fd_mem;
    void * mmap_ptr; // mapped pointer from NVME BAR0

    // nvme parameters; read from NVME BAR0
    unsigned int CAP_DSTRD; // Doorbell register stride
    unsigned int CAP_MQES; // Max Queue Entries size
    unsigned int CAP_TO; // Timeout

    // manager
    nvme_queue_pair *admin_queue_pair;
    unsigned int q_pair_num;

} nvme_admin_ctrl;

void alloc_dma_buf(nvme_admin_ctrl * ctrl, dma_buffer *ptr){
    ptr->vir_addr = mmap(0, sysconf(_SC_PAGESIZE) * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, ctrl->fd_dma, 0);
    ptr->phy_addr = ((unsigned long *)ptr->vir_addr)[0];
    ((unsigned long *)ptr->vir_addr)[0] = 0;
}

void free_dma_buf(dma_buffer *ptr){
    munmap(ptr->vir_addr, sysconf(_SC_PAGESIZE) * 1024);
    ptr->phy_addr = 0;
}

bool create_nvme_admin_ctrl(nvme_admin_ctrl * ctrl, unsigned long nvme_bar0){
    ctrl->fd_dma = open("/dev/dma_buffer", O_RDWR);
    ctrl->fd_mem = open("/dev/mem", O_RDWR);
    ctrl->mmap_ptr = mmap(0, sysconf(_SC_PAGESIZE) * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, ctrl->fd_mem, nvme_bar0);

    ctrl->CAP_DSTRD = CAP$DSTRD(ctrl->mmap_ptr);
    ctrl->CAP_MQES = CAP$MQES(ctrl->mmap_ptr) + 1;
    ctrl->CAP_TO = CAP$TO(ctrl->mmap_ptr);

    // printf("CAP_DSTRD:%d\n", ctrl->CAP_DSTRD);
    // printf("MQES:%d\n", ctrl->CAP_MQES);
    // printf("TO:%d\n", ctrl->CAP_TO);
    return true;
}

bool free_nvme_admin_ctrl(nvme_admin_ctrl * ctrl){
    munmap(ctrl->mmap_ptr, sysconf(_SC_PAGESIZE) * 1024);
    close(ctrl->fd_dma);
    close(ctrl->fd_mem);

    return true;
}

void create_register_admin_queue(nvme_queue_pair * q_pair, nvme_admin_ctrl * ctrl, unsigned int q_size){

    ctrl->q_pair_num = 1;

    // the SQ/CQ could use the same dma buffer since the dma buffer is usually very large
    alloc_dma_buf(ctrl, &(q_pair->sq.buf));
    alloc_dma_buf(ctrl, &(q_pair->cq.buf));

    q_pair->cq.db = CQ_DBL(ctrl->mmap_ptr, 0, 0);
    q_pair->cq.element_size = 4;
    q_pair->cq.entity = (cqe*) q_pair->cq.buf.vir_addr;
    q_pair->cq.head_idx = 0;
    q_pair->cq.qsize = q_size;
    q_pair->cq.tail_idx = 0;
    q_pair->cq.phase = 0;

    q_pair->sq.db = SQ_DBL(ctrl->mmap_ptr, 0, 0);
    q_pair->sq.element_size = 16;
    q_pair->sq.entity = (sqe*) q_pair->sq.buf.vir_addr;
    q_pair->sq.head_idx = 0;
    q_pair->sq.qsize = q_size;
    q_pair->sq.tail_idx = 0;

    // Set CC.EN to 0
    #if SIMULATION
    #else
        
    
    volatile uint32_t* cc = CC(ctrl->mmap_ptr);
    *cc = *cc & ~1;
    // This is the worst case time that host software shall wait for  CSTS.RDY to transition from: CC.EN
    usleep(ctrl->CAP_TO * 500);
    while (CSTS$RDY(ctrl->mmap_ptr) != 0){
        // printf("CSTS$RDY not ready\n");
        std::cout << "CSTS$RDY not ready\n";
        usleep(ctrl->CAP_TO * 500);
    }

    volatile uint32_t* aqa = AQA(ctrl->mmap_ptr);
    *aqa = AQA$AQS(q_pair->sq.qsize - 1) | AQA$AQC(q_pair->cq.qsize - 1);

    volatile uint64_t* acq = ACQ(ctrl->mmap_ptr);
    *acq = q_pair->cq.buf.phy_addr;

    volatile uint64_t* asq = ASQ(ctrl->mmap_ptr);
    *asq = q_pair->sq.buf.phy_addr;

    
    *cc = CC$IOCQES(4) | CC$IOSQES(6) | CC$MPS(0) | CC$CSS(0) | CC$EN(1);
    usleep(ctrl->CAP_TO * 500);
    while (CSTS$RDY(ctrl->mmap_ptr) != 1){
        // printf("CSTS$RDY not ready\n");
        std::cout << "CSTS$RDY not ready\n";
        usleep(ctrl->CAP_TO * 500);
    }
    #endif
}

void create_gpu_io_queue_pairs(nvme_queue_pair * q_pair, nvme_admin_ctrl * ctrl, unsigned int q_depth, unsigned int pair_size, unsigned long gpu_phy_addr){
    
    for(int i = 0; i < pair_size; ++i){
        // alloc_dma_buf(ctrl, &(q_pair[i].cq.buf));
        q_pair[i].pair_idx = ctrl->q_pair_num;
        ctrl->q_pair_num = ctrl->q_pair_num + 1;
        q_pair[i].cq.db = CQ_DBL(ctrl->mmap_ptr, q_pair->pair_idx, ctrl->CAP_DSTRD);
        q_pair[i].cq.buf.phy_addr = gpu_phy_addr + 64 * q_depth * pair_size + 32 * q_depth * i;
        // q_pair[i].cq.buf.phy_addr = gpu_phy_addr + 65536 * (i * 2 + 1);
        q_pair[i].cq.element_size = 4;
        // q_pair[i].cq.entity = (cqe*) q_pair->cq.buf.vir_addr;
        q_pair[i].cq.head_idx = 0;
        q_pair[i].cq.qsize = q_depth;
        q_pair[i].cq.tail_idx = 0;
        q_pair[i].cq.phase = 0;

        q_pair[i].sq.buf.phy_addr = gpu_phy_addr + 64 * q_depth * i;
        // q_pair[i].sq.buf.phy_addr = gpu_phy_addr + 65536 * (i * 2);

        q_pair[i].sq.element_size = 16;
        // q_pair[i].sq.entity = (sqe*) q_pair->sq.buf.vir_addr;

        q_pair[i].sq.head_idx = 0;
        q_pair[i].sq.qsize = q_depth;
        q_pair[i].sq.tail_idx = 0;
    }
}

void showcmd(unsigned int* ptr, unsigned int size){
    for(int i = 0; i < size; ++i){
        printf("%.8x ", ptr[i]);
    }
    printf("\n");
}

void submit_sq(nvme_queue_pair * qp, unsigned int cmd_num){
    #if SIMULATION
    #else
    // printf("submit cmd:%d\n", qp->sq.head_idx);
    for(int i = 0; i < cmd_num; ++i){
        qp->sq.entity[(qp->sq.head_idx + i) % qp->sq.qsize].cmd[0] |= qp->sq.identifier++ << 16; // set Command Identifier
        // printf("\t");
        // showcmd(qp->sq.entity[(qp->sq.head_idx + i) % qp->sq.qsize].cmd, qp->sq.element_size);
    }
    qp->sq.tail_idx = (qp->sq.tail_idx + cmd_num) % qp->sq.qsize;
    qp->sq.head_idx = qp->sq.tail_idx;
    *(qp->sq.db) = qp->sq.tail_idx;
    #endif
}

void inc_cq_header(nvme_queue_pair * qp){
    qp->cq.head_idx += 1;
    if(qp->cq.head_idx >= qp->cq.qsize){
        qp->cq.head_idx -= qp->cq.qsize;
        qp->cq.phase = (~qp->cq.phase) & 0x1;
    }
}

void wait_cpl(nvme_queue_pair * qp, unsigned int cmd_num){
    #if SIMULATION
    #else
    // printf("wait cpl:\n");
    for(int i = 0; i < cmd_num; ++i){
        while(((qp->cq.entity[qp->cq.head_idx].cpl[3] >> 16) & 0x1) == qp->cq.phase){
        }
        // printf("\t");
        // showcmd(qp->cq.entity[qp->cq.head_idx].cpl, qp->cq.element_size);
        if(qp->cq.entity[qp->cq.head_idx].cpl[3] >> 17 != 0){
            printf("NVME CMD error happens\n");
            exit(0);
        }
        inc_cq_header(qp);
        // qp->cq.head_idx = (qp->cq.head_idx + 1) % qp->cq.qsize;
    }
    if(qp->cq.db != nullptr){
        *(qp->cq.db) = qp->cq.head_idx;
    }
    #endif
}

void set_io_queue_num(nvme_queue_pair * admin_qp, unsigned int sq_num, unsigned int cq_num){
    unsigned int idx = admin_qp->sq.head_idx;
    admin_qp->sq.entity[idx].cmd[0] = 0x9;
    admin_qp->sq.entity[idx].cmd[10] = 0x7;
    admin_qp->sq.entity[idx].cmd[11] = (sq_num - 1) | ((cq_num - 1) << 16);
    submit_sq(admin_qp, 1);
    wait_cpl(admin_qp, 1);
}

void register_io_queue_pair(nvme_queue_pair * admin_qp, nvme_queue_pair * io_qp){
    // register CQ
    unsigned int idx = admin_qp->sq.head_idx;
    admin_qp->sq.entity[idx].cmd[0] = 0x5;
    // admin_qp->sq.entity[idx].cmd[1] = io_qp->pair_idx - 1;
    admin_qp->sq.entity[idx].cmd[6] = io_qp->cq.buf.phy_addr & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[7] = (io_qp->cq.buf.phy_addr >> 32) & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[10] = (io_qp->cq.qsize - 1) << 16 | io_qp->pair_idx;
    admin_qp->sq.entity[idx].cmd[11] = 0x1 | (io_qp->pair_idx << 16);

    idx = (idx + 1) % (admin_qp->sq.qsize);
    // submit_sq(admin_qp, 1);
    // wait_cpl(admin_qp, 1);

    // register SQ
    admin_qp->sq.entity[idx].cmd[0] = 0x1;
    // admin_qp->sq.entity[idx].cmd[1] = io_qp->pair_idx - 1;
    admin_qp->sq.entity[idx].cmd[6] = io_qp->sq.buf.phy_addr & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[7] = (io_qp->sq.buf.phy_addr >> 32) & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[10] = (io_qp->sq.qsize - 1) << 16 | io_qp->pair_idx;
    admin_qp->sq.entity[idx].cmd[11] = 0x00000001 | (io_qp->pair_idx << 16);
    submit_sq(admin_qp, 2);
    wait_cpl(admin_qp, 2);

}

void set_number_queues(nvme_queue_pair * admin_qp, unsigned int number){
    unsigned int idx = admin_qp->sq.head_idx;
    admin_qp->sq.entity[idx].cmd[0] = 0x09;
    admin_qp->sq.entity[idx].cmd[1] = 0x0;
    admin_qp->sq.entity[idx].cmd[10] = 0x07;
    admin_qp->sq.entity[idx].cmd[11] = ((number - 1) << 16) | (number - 1);

    submit_sq(admin_qp, 1);
    wait_cpl(admin_qp, 1);
}

void set_arbitration_burst(nvme_queue_pair * admin_qp){
    unsigned int idx = admin_qp->sq.head_idx;
    admin_qp->sq.entity[idx].cmd[0] = 0x09;
    admin_qp->sq.entity[idx].cmd[1] = 0x0;
    admin_qp->sq.entity[idx].cmd[10] = 0x01;
    admin_qp->sq.entity[idx].cmd[11] = 0x5;

    submit_sq(admin_qp, 1);
    wait_cpl(admin_qp, 1);
}

void get_error_log_page(nvme_queue_pair * admin_qp, dma_buffer *ptr){
    unsigned int idx = admin_qp->sq.head_idx;
    admin_qp->sq.entity[idx].cmd[0] = 0x2;
    admin_qp->sq.entity[idx].cmd[1] = 0x0;
    admin_qp->sq.entity[idx].cmd[6] = ptr->phy_addr & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[7] = (ptr->phy_addr >> 32) & 0xffffffff;

    admin_qp->sq.entity[idx].cmd[10] = ((512 << 16)|(0x1));

    submit_sq(admin_qp, 1);
    wait_cpl(admin_qp, 1);


}

void show_error_log(dma_buffer *ptr, unsigned int size){
    
    for(int i = 0; i < size; ++i){
        unsigned int * data = (unsigned int *) (ptr->vir_addr + i * 64);
        printf("---------------------\n");
        printf("error count: %d\n", data[0]);
        printf("sq id: %d\n", data[8]);
        printf("cmd id: %d\n", data[10]);
        printf("status: %d\n", data[12]);
        printf("parameter error location: %d\n", data[14]);
        printf("LBA: %d\n", data[16]);
        printf("nid: %d\n", data[24]);
    }
    printf("---------------------\n");
}

void register_gpu_io_queue_pair(nvme_queue_pair * admin_qp, nvme_queue_pair * gpu_io_qps, unsigned int queue_pair_size){
    for(int i = 0; i < queue_pair_size; ++i){
        register_io_queue_pair(admin_qp, &(gpu_io_qps[i]));
    }
}

void free_queue_pair(nvme_queue_pair * q_pair){
    free_dma_buf(&(q_pair->cq.buf));
    free_dma_buf(&(q_pair->sq.buf));
}