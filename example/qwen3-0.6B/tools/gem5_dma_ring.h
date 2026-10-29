#ifndef QWEN_GEM5_DMA_RING_H
#define QWEN_GEM5_DMA_RING_H
#include <stdint.h>

/* ABI v1: 128-byte little-endian records, 64-byte aligned ring, one producer.
 * Descriptor/source writes precede the doorbell; outputs are read after fence.
 * Sequence numbers are global, events are level bits explicitly cleared by
 * the consumer. Do not mix ring and legacy submissions in a batch. */
typedef struct {
    uint64_t id, system, spm, row_bytes, rows, system_stride, spm_stride;
    uint64_t flags, wait_sequence, wait_events, signal_events, reserved[5];
} qwen_dma_descriptor;
_Static_assert(sizeof(qwen_dma_descriptor) == 128, "DMA ring ABI");

typedef struct {
    volatile uint64_t *regs;
    volatile qwen_dma_descriptor *slots;
    uint64_t capacity, tail, sequence_base;
} qwen_dma_ring;

static inline void qwen_dma_memory_fence(void)
{
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static inline int qwen_ring_init(qwen_dma_ring *r,
                                 volatile qwen_dma_descriptor *slots,
                                 unsigned capacity)
{
    r->regs = (volatile uint64_t *)(uintptr_t)0x40010000;
    if (r->regs[24] != 1 || (r->regs[1] & 5) || !capacity ||
        capacity > 65536 || ((uintptr_t)slots & 63)) return -1;
    r->slots = slots; r->capacity = capacity; r->tail = 0;
    r->sequence_base = r->regs[13];
    r->regs[16] = 0;
    r->regs[15] = (uintptr_t)slots;
    r->regs[16] = capacity;
    return (r->regs[1] & 4) ? -1 : 0;
}

/* Returns the future completion ticket, zero if ring storage has no room.
 * The slot becomes reusable when ring head advances (descriptor copied).
 * Payload/SPM lifetimes instead follow completion and compute ownership. */
static inline uint64_t qwen_ring_put(qwen_dma_ring *r,
                                     const qwen_dma_descriptor *d)
{
    if ((r->regs[1] & 4) || r->tail - r->regs[17] >= r->capacity) return 0;
    volatile uint64_t *dst = (volatile uint64_t *)&r->slots[r->tail % r->capacity];
    const uint64_t *src = (const uint64_t *)d;
    for (unsigned i = 0; i < 16; ++i) dst[i] = src[i];
    return r->sequence_base + ++r->tail;
}

static inline uint64_t qwen_ring_submit(qwen_dma_ring *r)
{
    qwen_dma_memory_fence();
    r->regs[18] = r->tail;
    return r->sequence_base + r->tail;
}

/* Nonblocking fence check: 1 complete, 0 pending, -1 device error. */
static inline int qwen_ring_complete(qwen_dma_ring *r, uint64_t ticket)
{
    if (r->regs[1] & 4) return -1;
    if (r->regs[14] < ticket) return 0;
    if (r->regs[1] & 4) return -1;
    qwen_dma_memory_fence();
    return 1;
}

enum qwen_buffer_state { QWEN_FREE, QWEN_FILLING, QWEN_READY,
                         QWEN_COMPUTING, QWEN_DRAINING };
typedef struct {
    uintptr_t spm;
    uint64_t ticket;
    enum qwen_buffer_state state;
} qwen_spm_buffer;

static inline int qwen_buffer_fill(qwen_spm_buffer *b, uint64_t ticket)
{
    if (b->state != QWEN_FREE || !ticket) return -1;
    b->ticket = ticket; b->state = QWEN_FILLING;
    return 0;
}
static inline int qwen_buffer_acquire(qwen_dma_ring *r, qwen_spm_buffer *b)
{
    if (b->state != QWEN_FILLING) return -1;
    int done = qwen_ring_complete(r, b->ticket);
    if (done == 1) b->state = QWEN_READY;
    return done;
}
/* Use after an accelerator-side event dependency has consumed the DMA-ready
 * event.  The event is the completion proof, so a second CPU MMIO poll would
 * only repeat the same dependency and serialize the control path. */
static inline int qwen_buffer_acquire_event(qwen_spm_buffer *b)
{
    if (b->state != QWEN_FILLING) return -1;
    qwen_dma_memory_fence();
    b->state = QWEN_READY;
    return 0;
}
static inline int qwen_buffer_compute(qwen_spm_buffer *b)
{
    if (b->state != QWEN_READY) return -1;
    b->state = QWEN_COMPUTING;
    return 0;
}
/* Call only after AME's architectural completion. A zero output ticket
 * releases an input-only buffer; otherwise keep the output live for DMA. */
static inline int qwen_buffer_release(qwen_spm_buffer *b, uint64_t output_ticket)
{
    if (b->state != QWEN_COMPUTING) return -1;
    b->ticket = output_ticket;
    b->state = output_ticket ? QWEN_DRAINING : QWEN_FREE;
    return 0;
}
static inline int qwen_buffer_reclaim(qwen_dma_ring *r, qwen_spm_buffer *b)
{
    if (b->state != QWEN_DRAINING) return -1;
    int done = qwen_ring_complete(r, b->ticket);
    if (done == 1) b->state = QWEN_FREE;
    return done;
}
#endif
