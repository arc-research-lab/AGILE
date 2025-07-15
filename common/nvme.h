#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>


#define _NVM_MASK(num_bits) \
    ((1ULL << (num_bits)) - 1)

#define _NVM_MASK_PART(hi, lo) \
    (_NVM_MASK((hi) + 1) - _NVM_MASK(lo))

/* Extract specific bits */
#define _RB(v, hi, lo)      \
    ( ( (v) & _NVM_MASK_PART((hi), (lo)) ) >> (lo) )

/* Set specifics bits */
#define _WB(v, hi, lo)      \
    ( ( (v) << (lo) ) & _NVM_MASK_PART((hi), (lo)) )

/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) ((volatile void*) (p))) + (offs)))


/* Calculate the base-2 logarithm of a number n */
static inline uint32_t _nvm_b2log(uint32_t n)
{
    uint32_t count = 0;

    while (n > 0)
    {
        ++count;
        n >>= 1;
    }

    return count - 1;
}




/* Controller registers */
#define CAP(p)          _REG(p, 0x0000, 64)     // Controller Capabilities
#define VER(p)          _REG(p, 0x0008, 32)     // NVM Express version
#define CC(p)           _REG(p, 0x0014, 32)     // Controller Configuration
#define CSTS(p)         _REG(p, 0x001c, 32)     // Controller Status
#define AQA(p)          _REG(p, 0x0024, 32)     // Admin Queue Attributes
#define ASQ(p)          _REG(p, 0x0028, 64)     // Admin Submission Queue Base Address
#define ACQ(p)          _REG(p, 0x0030, 64)     // Admin Completion Queue Base Address


/* Read bit fields */
#define CAP$MPSMAX(p)   _RB(*CAP(p), 55, 52)    // Memory Page Size Maximum
#define CAP$MPSMIN(p)   _RB(*CAP(p), 51, 48)    // Memory Page Size Minimum
#define CAP$DSTRD(p)    _RB(*CAP(p), 35, 32)    // Doorbell Stride
#define CAP$TO(p)       _RB(*CAP(p), 31, 24)    // Timeout
#define CAP$CQR(p)      _RB(*CAP(p), 16, 16)    // Contiguous Queues Required
#define CAP$MQES(p)     _RB(*CAP(p), 15,  0)    // Maximum Queue Entries Supported

#define CSTS$RDY(p)     _RB(*CSTS(p), 0,  0)    // Ready indicator


/* Write bit fields */
#define CC$IOCQES(v)    _WB(v, 23, 20)          // IO Completion Queue Entry Size
#define CC$IOSQES(v)    _WB(v, 19, 16)          // IO Submission Queue Entry Size
#define CC$MPS(v)       _WB(v, 10,  7)          // Memory Page Size
#define CC$CSS(v)       _WB(0,  3,  1)          // IO Command Set Selected (0=NVM Command Set)
#define CC$EN(v)        _WB(v,  0,  0)          // Enable

#define AQA$AQS(v)      _WB(v, 27, 16)          // Admin Completion Queue Size
#define AQA$AQC(v)      _WB(v, 11,  0)          // Admin Submission Queue Size


/* SQ doorbell register offset */
#define SQ_DBL(p, y, dstrd)    \
        ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


/* CQ doorbell register offset */
#define CQ_DBL(p, y, dstrd)    \
        ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )




typedef struct __dma_buffer__ {
    void *vir_addr;
    unsigned long phy_addr;
} dma_buffer;

typedef struct __sqe__{
    unsigned int cmd[16];
} sqe;

typedef struct __cqe__{
    unsigned int cpl[4];
} cqe;

typedef struct __nvme_cq__ {
    dma_buffer buf;
    cqe * entity;
    unsigned int element_size; // element_size == 16 if it is SQ else 4
    unsigned int head_idx;
    unsigned int tail_idx;
    unsigned int qsize;
    int phase;
    volatile unsigned int *db; // pointer to nvme SQ/CQ doorbell register
} nvme_cq;

typedef struct __nvme_sq__ {
    dma_buffer buf;
    sqe * entity;
    unsigned int element_size; // element_size == 16 if it is SQ else 4
    unsigned int head_idx;
    unsigned int tail_idx;
    unsigned int qsize;
    unsigned char identifier;
    volatile unsigned int *db; // pointer to nvme SQ/CQ doorbell register
} nvme_sq;

typedef struct __nvme_queue_pair__ {
    nvme_cq cq;
    nvme_sq sq;
    unsigned int pair_idx;
} nvme_queue_pair;

typedef struct __nvme_ctrl__ {
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

} nvme_ctrl;


void alloc_dma_buf(nvme_ctrl * ctrl, dma_buffer *ptr){
    ptr->vir_addr = mmap(0, sysconf(_SC_PAGESIZE) * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, ctrl->fd_dma, 0);
    ptr->phy_addr = ((unsigned long *)ptr->vir_addr)[0];
    ((unsigned long *)ptr->vir_addr)[0] = 0;
}

void free_dma_buf(dma_buffer *ptr){
    munmap(ptr->vir_addr, sysconf(_SC_PAGESIZE) * 1024);
    ptr->phy_addr = 0;
}

bool create_nvme_ctrl(nvme_ctrl * ctrl, unsigned long nvme_bar0){
    ctrl->fd_dma = open("/dev/dma_buffer", O_RDWR);
    ctrl->fd_mem = open("/dev/mem", O_RDWR);
    ctrl->mmap_ptr = mmap(0, sysconf(_SC_PAGESIZE) * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, ctrl->fd_mem, nvme_bar0);

    ctrl->CAP_DSTRD = CAP$DSTRD(ctrl->mmap_ptr);
    ctrl->CAP_MQES = CAP$MQES(ctrl->mmap_ptr) + 1;
    ctrl->CAP_TO = CAP$TO(ctrl->mmap_ptr);

    // printf("CAP_DSTRD:%d\n", ctrl->CAP_DSTRD);
    // printf("MQES:%d\n", ctrl->CAP_MQES);
    // printf("TO:%d\n", ctrl->CAP_TO);
    // TODO: check if all files are opened successfully 
    // if(fd_dma ){
    // return false; 
    // }
    return true;
}

bool free_nvme_ctrl(nvme_ctrl * ctrl){
    munmap(ctrl->mmap_ptr, sysconf(_SC_PAGESIZE) * 1024);
    close(ctrl->fd_dma);
    close(ctrl->fd_mem);

    return true;
}

/**
 * The size of CQ SQ is the same
*/
void create_register_admin_queue(nvme_queue_pair * q_pair, nvme_ctrl * ctrl, unsigned int q_size){

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
    volatile uint32_t* cc = CC(ctrl->mmap_ptr);
    *cc = *cc & ~1;
    // This is the worst case time that host software shall wait for  CSTS.RDY to transition from: CC.EN
    usleep(ctrl->CAP_TO * 500);
    while (CSTS$RDY(ctrl->mmap_ptr) != 0){
        printf("CSTS$RDY not ready\n");
        usleep(ctrl->CAP_TO * 500);
    }

    volatile uint32_t* aqa = AQA(ctrl->mmap_ptr);
    *aqa = AQA$AQS(q_pair->sq.qsize) | AQA$AQC(q_pair->cq.qsize);

    volatile uint64_t* acq = ACQ(ctrl->mmap_ptr);
    *acq = q_pair->cq.buf.phy_addr;

    volatile uint64_t* asq = ASQ(ctrl->mmap_ptr);
    *asq = q_pair->sq.buf.phy_addr;

    // TODO: check MPS
    *cc = CC$IOCQES(4) | CC$IOSQES(6) | CC$MPS(0) | CC$CSS(0) | CC$EN(1);
    usleep(ctrl->CAP_TO * 500);
    while (CSTS$RDY(ctrl->mmap_ptr) != 1){
        printf("CSTS$RDY not ready\n");
        usleep(ctrl->CAP_TO * 500);
    }
}

void create_io_queue_pair(nvme_queue_pair * q_pair, nvme_ctrl * ctrl, unsigned int q_size){
    alloc_dma_buf(ctrl, &(q_pair->sq.buf));
    alloc_dma_buf(ctrl, &(q_pair->cq.buf));

    q_pair->pair_idx = ctrl->q_pair_num++;
    q_pair->cq.db = CQ_DBL(ctrl->mmap_ptr, q_pair->pair_idx, ctrl->CAP_DSTRD);
    // printf("create_io_queue_pair: %d %d\n", q_pair->pair_idx, ctrl->CAP_DSTRD);
    q_pair->cq.element_size = 4;
    q_pair->cq.entity = (cqe*) q_pair->cq.buf.vir_addr;
    q_pair->cq.head_idx = 0;
    q_pair->cq.qsize = q_size;
    q_pair->cq.tail_idx = 0;
    q_pair->cq.phase = 0;

    q_pair->sq.db = SQ_DBL(ctrl->mmap_ptr, q_pair->pair_idx, ctrl->CAP_DSTRD);
    q_pair->sq.element_size = 16;
    q_pair->sq.entity = (sqe*) q_pair->sq.buf.vir_addr;
    q_pair->sq.head_idx = 0;
    q_pair->sq.qsize = q_size;
    q_pair->sq.tail_idx = 0;
    q_pair->sq.identifier = 0;
}

void create_io_queue_pairs(nvme_queue_pair * q_pair, nvme_ctrl * ctrl, unsigned int q_depth, unsigned int q_num){
    for(int i = 0; i < q_num; ++i){
        create_io_queue_pair(&(q_pair[i]), ctrl, q_depth);
    }
    
}

void create_fpga_io_queue_pairs(nvme_queue_pair * q_pair, nvme_ctrl * ctrl, unsigned int q_depth, unsigned int pair_size, unsigned long fpga_bar){
    
    for(int i = 0; i < pair_size; ++i){
        // alloc_dma_buf(ctrl, &(q_pair[i].cq.buf));
        q_pair[i].pair_idx = ctrl->q_pair_num;
        ctrl->q_pair_num = ctrl->q_pair_num + 1;
        q_pair[i].cq.db = CQ_DBL(ctrl->mmap_ptr, q_pair->pair_idx, ctrl->CAP_DSTRD);
        q_pair[i].cq.buf.phy_addr = fpga_bar + (512 / 8) * q_depth * pair_size + 4 * 4 * 2 * q_depth * i;
        q_pair[i].cq.element_size = 4;
        q_pair[i].cq.entity = (cqe*) q_pair->cq.buf.vir_addr;
        q_pair[i].cq.head_idx = 0;
        q_pair[i].cq.qsize = q_depth;
        q_pair[i].cq.tail_idx = 0;
        q_pair[i].cq.phase = 0;

        q_pair[i].sq.buf.phy_addr = fpga_bar + (512 / 8) * q_depth * i;
        q_pair[i].sq.element_size = 16;
        // q_pair[i].sq.entity = (sqe*) q_pair->sq.buf.vir_addr;
        q_pair[i].sq.head_idx = 0;
        q_pair[i].sq.qsize = q_depth;
        q_pair[i].sq.tail_idx = 0;
    }
}



void create_fpga_io_queue_pairs_test(nvme_queue_pair * q_pair, nvme_ctrl * ctrl, unsigned int q_depth, unsigned int pair_size, unsigned long fpga_bar){
    
    for(int i = 0; i < pair_size; ++i){
        alloc_dma_buf(ctrl, &(q_pair[i].cq.buf));
        q_pair[i].pair_idx = ctrl->q_pair_num++;
        // q_pair[i].cq.buf.phy_addr = fpga_bar + (512 / 8) * q_depth * pair_size + 4 * 4 * 2 * q_depth * i;
        q_pair[i].cq.element_size = 4;
        q_pair[i].cq.entity = (cqe*) q_pair->cq.buf.vir_addr;
        q_pair[i].cq.db = CQ_DBL(ctrl->mmap_ptr, q_pair->pair_idx, ctrl->CAP_DSTRD);
        q_pair[i].cq.head_idx = 0;
        q_pair[i].cq.qsize = q_depth;
        q_pair[i].cq.tail_idx = 0;
        q_pair[i].cq.phase = 0;

        q_pair[i].sq.buf.phy_addr = fpga_bar + (512 / 8) * q_depth * i;
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
    // printf("submit cmd:%d\n", qp->sq.head_idx);
    for(int i = 0; i < cmd_num; ++i){
        qp->sq.entity[(qp->sq.head_idx + i) % qp->sq.qsize].cmd[0] |= qp->sq.identifier++ << 16; // set Command Identifier
        // printf("\t");
        // showcmd(qp->sq.entity[(qp->sq.head_idx + i) % qp->sq.qsize].cmd, qp->sq.element_size);
    }
    qp->sq.tail_idx = (qp->sq.tail_idx + cmd_num) % qp->sq.qsize;
    qp->sq.head_idx = qp->sq.tail_idx;
    *(qp->sq.db) = qp->sq.tail_idx;
}

void inc_cq_header(nvme_queue_pair * qp){
    qp->cq.head_idx += 1;
    if(qp->cq.head_idx >= qp->cq.qsize){
        qp->cq.head_idx -= qp->cq.qsize;
        qp->cq.phase = (~qp->cq.phase) & 0x1;
    }
}

void wait_cpl(nvme_queue_pair * qp, unsigned int cmd_num){
    // printf("wait cpl:\n");
    for(int i = 0; i < cmd_num; ++i){
        while(((qp->cq.entity[qp->cq.head_idx].cpl[3] >> 16) & 0x1) == qp->cq.phase){
        }
        // printf("\t");
        // showcmd(qp->cq.entity[qp->cq.head_idx].cpl, qp->cq.element_size);
        inc_cq_header(qp);
        // qp->cq.head_idx = (qp->cq.head_idx + 1) % qp->cq.qsize;
    }
    if(qp->cq.db != nullptr){
        *(qp->cq.db) = qp->cq.head_idx;
    }
    
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
    admin_qp->sq.entity[idx].cmd[10] = io_qp->cq.qsize << 16 | io_qp->pair_idx;
    admin_qp->sq.entity[idx].cmd[11] = 0x1 | (io_qp->pair_idx << 16);

    idx = (idx + 1) % (admin_qp->sq.qsize);
    // submit_sq(admin_qp, 1);
    // wait_cpl(admin_qp, 1);

    // register SQ
    admin_qp->sq.entity[idx].cmd[0] = 0x1;
    // admin_qp->sq.entity[idx].cmd[1] = io_qp->pair_idx - 1;
    admin_qp->sq.entity[idx].cmd[6] = io_qp->sq.buf.phy_addr & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[7] = (io_qp->sq.buf.phy_addr >> 32) & 0xffffffff;
    admin_qp->sq.entity[idx].cmd[10] = io_qp->sq.qsize << 16 | io_qp->pair_idx;
    admin_qp->sq.entity[idx].cmd[11] = 0x00000001 | (io_qp->pair_idx << 16);
    submit_sq(admin_qp, 2);
    wait_cpl(admin_qp, 2);

}


void register_fpga_io_queue_pair(nvme_queue_pair * admin_qp, nvme_queue_pair * fpga_io_qps, unsigned int queue_pair_size){
    for(int i = 0; i < queue_pair_size; ++i){
        register_io_queue_pair(admin_qp, &(fpga_io_qps[i]));
    }
}



void free_queue_pair(nvme_queue_pair * q_pair){
    free_dma_buf(&(q_pair->cq.buf));
    free_dma_buf(&(q_pair->sq.buf));
}

void nvme_read(dma_buffer *buf, nvme_queue_pair *io_qp, unsigned int offset, unsigned int blocks){

    unsigned int idx = io_qp->sq.head_idx;
    io_qp->sq.entity[idx].cmd[0] = 0x2;
    io_qp->sq.entity[idx].cmd[1] = 0x1;
    io_qp->sq.entity[idx].cmd[6] = (buf->phy_addr) & 0xffffffff;
    io_qp->sq.entity[idx].cmd[7] = (buf->phy_addr >> 32) & 0xffffffff;
    io_qp->sq.entity[idx].cmd[10] = offset;
    io_qp->sq.entity[idx].cmd[12] = blocks - 1;
    submit_sq(io_qp, 1);
    wait_cpl(io_qp, 1);
}

void nvme_write(dma_buffer *buf, nvme_queue_pair *io_qp, unsigned int offset, unsigned int blocks){
    unsigned int idx = io_qp->sq.head_idx;
    io_qp->sq.entity[idx].cmd[0] = 0x1;
    io_qp->sq.entity[idx].cmd[1] = 0x1;
    io_qp->sq.entity[idx].cmd[6] = (buf->phy_addr) & 0xffffffff;
    io_qp->sq.entity[idx].cmd[7] = (buf->phy_addr >> 32) & 0xffffffff;
    io_qp->sq.entity[idx].cmd[10] = offset;
    io_qp->sq.entity[idx].cmd[12] = blocks - 1;
    submit_sq(io_qp, 1);
    wait_cpl(io_qp, 1);
}

dma_buffer * alloc_buf_arr(nvme_ctrl * ctrl, unsigned size){
    dma_buffer * buf = (dma_buffer *) malloc(sizeof(dma_buffer) * size);
    for(int i = 0; i < size; ++i){
        alloc_dma_buf(ctrl, (buf + i));
    }
    return buf;
}

void free_dma_buf_arr(dma_buffer * buf, unsigned size){
    for(int i = 0; i < size; ++i){
        free_dma_buf(buf + i);
    }
    free(buf);
}

void nvme_read(unsigned long phy_addr, unsigned int buf_len, nvme_queue_pair *io_qp){
    unsigned int idx = io_qp->sq.head_idx;
    for(int i = 0; i < buf_len; ++i){
        io_qp->sq.entity[idx].cmd[0] = 0x2;
        io_qp->sq.entity[idx].cmd[1] = 0x1;
        io_qp->sq.entity[idx].cmd[6] = (phy_addr + i * 512) & 0xffffffff;
        io_qp->sq.entity[idx].cmd[7] = ((phy_addr + i * 512) >> 32) & 0xffffffff;
        io_qp->sq.entity[idx].cmd[10] = i;
        io_qp->sq.entity[idx].cmd[12] = 0;
        idx = (idx + 1) % io_qp->sq.qsize;
    }

    submit_sq(io_qp, buf_len / 2);
    submit_sq(io_qp, buf_len / 2);
    wait_cpl(io_qp, buf_len);
}

void nvme_read_qps(unsigned long phy_addr, unsigned int buf_len, nvme_queue_pair *io_qp){
    // unsigned int idx = io_qp->sq.head_idx;
    // for(int i = 0; i < buf_len; ++i){
    //     io_qp->sq.entity[idx].cmd[0] = 0x2;
    //     io_qp->sq.entity[idx].cmd[1] = 0x1;
    //     io_qp->sq.entity[idx].cmd[6] = (phy_addr + i * 512) & 0xffffffff;
    //     io_qp->sq.entity[idx].cmd[7] = ((phy_addr + i * 512) >> 32) & 0xffffffff;
    //     io_qp->sq.entity[idx].cmd[10] = i;
    //     io_qp->sq.entity[idx].cmd[12] = 0;
    //     idx = (idx + 1) % io_qp->sq.qsize;
    // }

    // submit_sq(io_qp, buf_len / 2);
    // submit_sq(io_qp, buf_len / 2);
    // wait_cpl(io_qp, buf_len);
    int batch = 8;
    unsigned int q_idx = 0;
    for(int i = 0; i < buf_len; i += batch){
        unsigned int sq_head_idx = io_qp[q_idx].sq.head_idx;
        for(int j = 0; j < batch; ++j){
            io_qp[q_idx].sq.entity[sq_head_idx].cmd[0] = 0x2;
            io_qp[q_idx].sq.entity[sq_head_idx].cmd[1] = 1;
            io_qp[q_idx].sq.entity[sq_head_idx].cmd[6] = (phy_addr + (i + j) * 512) & 0xffffffff;
            io_qp[q_idx].sq.entity[sq_head_idx].cmd[7] = ((phy_addr + (i + j) * 512) >> 32) & 0xffffffff;
            io_qp[q_idx].sq.entity[sq_head_idx].cmd[10] = (i + j);
            io_qp[q_idx].sq.entity[sq_head_idx].cmd[12] = 0;
            sq_head_idx++;
        }
        submit_sq(&(io_qp[q_idx]), batch);
        wait_cpl(&(io_qp[q_idx]), batch);
        q_idx = (q_idx + 1) % 128;
    }
}

void two_io_queue_read(dma_buffer *buf0, nvme_queue_pair *io_qp0, dma_buffer *buf1, nvme_queue_pair *io_qp1, unsigned int buf_len){
    unsigned int idx0 = io_qp0->sq.head_idx;
    unsigned int idx1 = io_qp1->sq.head_idx;
    for(int i = 0; i < buf_len; ++i){
        io_qp0->sq.entity[idx0].cmd[0] = 0x2;
        io_qp0->sq.entity[idx0].cmd[1] = 0x1;
        io_qp0->sq.entity[idx0].cmd[6] = buf0[i].phy_addr & 0xffffffff;
        io_qp0->sq.entity[idx0].cmd[7] = (buf0[i].phy_addr >> 32) & 0xffffffff;
        io_qp0->sq.entity[idx0].cmd[12] = 1;
        idx0 = (idx0 + 1) % io_qp0->sq.qsize;

        io_qp1->sq.entity[idx1].cmd[0] = 0x2;
        io_qp1->sq.entity[idx1].cmd[1] = 0x1;
        io_qp1->sq.entity[idx1].cmd[6] = buf1[i].phy_addr & 0xffffffff;
        io_qp1->sq.entity[idx1].cmd[7] = (buf1[i].phy_addr >> 32) & 0xffffffff;
        io_qp1->sq.entity[idx1].cmd[12] = 1;
        idx1 = (idx1 + 1) % io_qp1->sq.qsize;
    }

    for(int i = 0; i < buf_len; ++i){
        io_qp0->sq.entity[(io_qp0->sq.head_idx + i) % io_qp0->sq.qsize].cmd[0] |= i << 16; // set Command Identifier
        io_qp1->sq.entity[(io_qp1->sq.head_idx + i) % io_qp1->sq.qsize].cmd[0] |= i << 16; // set Command Identifier
    }
    io_qp0->sq.tail_idx = (io_qp0->sq.tail_idx + buf_len) % io_qp0->sq.qsize;
    io_qp0->sq.head_idx = io_qp0->sq.tail_idx;
    io_qp1->sq.tail_idx = (io_qp1->sq.tail_idx + buf_len) % io_qp1->sq.qsize;
    io_qp1->sq.head_idx = io_qp1->sq.tail_idx;
    *(io_qp0->sq.db) = io_qp0->sq.tail_idx;
    *(io_qp1->sq.db) = io_qp1->sq.tail_idx;
    wait_cpl(io_qp0, buf_len);
    wait_cpl(io_qp1, buf_len);
}


void emu_fpga_wait_cpl(nvme_queue_pair * q_pair, unsigned int pair_size, unsigned int wait_num){
    unsigned int finished = 0;
    printf("emu_fpga_wait_cpl\n");
    for(int i = 0; i < pair_size; ++i){
        printf("wait :%d\n", i);
        wait_cpl(&(q_pair[i]), wait_num - finished > 32 ? 32 : wait_num - finished);
        finished += (wait_num - finished > 32 ? 32 : wait_num - finished);
        if(finished >= wait_num){
            break;
        }
    }
    printf("emu_fpga_wait_cpl finish\n");
}