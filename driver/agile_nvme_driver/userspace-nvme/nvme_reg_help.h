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
// static inline uint32_t _nvm_b2log(uint32_t n)
// {
//     uint32_t count = 0;

//     while (n > 0)
//     {
//         ++count;
//         n >>= 1;
//     }

//     return count - 1;
// }




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
// #define SQ_DBL(p, y, dstrd)    \
//         ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


/* CQ doorbell register offset */
#define CQ_DBL(p, y, dstrd)    \
        ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )

        