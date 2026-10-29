/* Guest DMA staging around the 4x4 Ztt instruction kernel.  A/B contain four
 * packed 16-byte int8 squares (AMEUDSZ=32), while C is a 64-byte int32 tile.
 * The ring path batches descriptors and uses explicit four-slot ownership;
 * the compatibility path remains blocking.  Separate SPM regions prevent
 * artificial A/B/C aliasing.
 */
#include <stdint.h>
#include "gem5_m5ops.h"
#if QWEN_DMA_RING
#include "gem5_dma_ring.h"
enum { QWEN_DMA_RING_CAPACITY = 64, QWEN_DMA_SUBMIT_BATCH = 16 };
static qwen_dma_descriptor dma_ring_slots[QWEN_DMA_RING_CAPACITY]
    __attribute__((aligned(64)));
static qwen_dma_ring dma_ring;
static unsigned dma_ring_initialized;
static unsigned dma_ring_pending;
#endif

extern void qwen_ztt_direct(int32_t *, const int8_t *, const int8_t *);
static uint64_t descriptor_id;
static uint64_t last_ticket;
static volatile uint64_t *const dma_regs =
    (volatile uint64_t *)(uintptr_t)0x40010000;

/* This compatibility path submits MMIO descriptors. The real memory ring
 * API lives in gem5_dma_ring.h. Buffer indices alone do not imply ownership. */
enum { QWEN_DMA_BUFFER_COUNT = 4 };

unsigned qwen_dma_buffer_index(unsigned slot)
{
    return slot % QWEN_DMA_BUFFER_COUNT;
}

static void qwen_dma_flush(void)
{
#if QWEN_DMA_RING
    if (dma_ring_pending) {
        last_ticket = qwen_ring_submit(&dma_ring);
        dma_ring_pending = 0;
    }
#endif
}

static uint64_t qwen_dma_submit_dep(uintptr_t system, uintptr_t spm,
    unsigned bytes, unsigned to_system, uint64_t wait_sequence,
    uint64_t wait_events, uint64_t signal_events)
{
#if QWEN_DMA_RING
    if (!dma_ring_initialized) {
        if (qwen_ring_init(&dma_ring, dma_ring_slots,
                           QWEN_DMA_RING_CAPACITY)) m5_fail(0, 22);
        dma_ring_initialized = 1;
    }
    qwen_dma_descriptor d = {0};
    d.id = ++descriptor_id;
    d.system = system; d.spm = spm; d.row_bytes = bytes; d.rows = 1;
    d.system_stride = bytes; d.spm_stride = bytes; d.flags = !!to_system;
    d.wait_sequence = wait_sequence;
    d.wait_events = wait_events;
    d.signal_events = signal_events;
    while (!qwen_ring_put(&dma_ring, &d)) {
        if (dma_regs[1] & 4) m5_fail(0, 22);
        qwen_dma_flush();
    }
    last_ticket = dma_ring.sequence_base + dma_ring.tail;
    if (++dma_ring_pending == QWEN_DMA_SUBMIT_BATCH)
        qwen_dma_flush();
    return last_ticket;
#else
    const unsigned extended = dma_regs[0xc0 / 8] == 1;
    if (dma_regs[1] & 4) m5_fail(0, 22);
    while (extended ? dma_regs[0xb8 / 8] == 0 : (dma_regs[1] & 1)) {
        if (dma_regs[1] & 4) m5_fail(0, 22);
    }
    const uint64_t id = ++descriptor_id;
    dma_regs[2] = system;
    dma_regs[3] = spm;
    dma_regs[4] = bytes;
    dma_regs[5] = 1;
    dma_regs[6] = bytes;
    dma_regs[7] = bytes;
    dma_regs[10] = id;
    __asm__ volatile("fence rw, rw" ::: "memory");
    /* A start write snapshots the programmed descriptor into the DMA queue.
     * It does not wait for completion, allowing the next descriptor to be
     * submitted while the current one is in DDR/SPM service. */
    dma_regs[0] = 1 | (to_system ? 2 : 0);
    if (dma_regs[1] & 4) m5_fail(0, 22);
    last_ticket = extended ? dma_regs[0x68 / 8] : id;
    return last_ticket;
#endif
}

uint64_t qwen_dma_submit(uintptr_t system, uintptr_t spm, unsigned bytes,
                         unsigned to_system)
{
    return qwen_dma_submit_dep(system, spm, bytes, to_system, 0, 0, 0);
}

void qwen_dma_fence(uint64_t id)
{
    qwen_dma_flush();
    uint64_t status;
    do {
        status = dma_regs[1];
        if ((status & 4) != 0)
            m5_fail(0, 22);
    } while (dma_regs[0xc0 / 8] == 1 ? dma_regs[0x70 / 8] < id : (status & 1));
    if (dma_regs[1] & 4) m5_fail(0, 22);
    __asm__ volatile("fence rw, rw" ::: "memory");
}

void qwen_dma_copy(uintptr_t system, uintptr_t spm, unsigned bytes,
                     unsigned to_system) {
    const uint64_t id = qwen_dma_submit(system, spm, bytes, to_system);
    qwen_dma_fence(id);
}

void qwen_dma_tile(int32_t *out, const int8_t *a, const int8_t *b) {
    const uintptr_t sa = 0xc0000000, sb = 0xc0000040, sc = 0xc0000080;
    qwen_dma_copy((uintptr_t)a, sa, 64, 0);
    qwen_dma_copy((uintptr_t)b, sb, 64, 0);
    qwen_ztt_direct((int32_t *)sc, (const int8_t *)sa, (const int8_t *)sb);
    qwen_dma_copy((uintptr_t)out, sc, 64, 1);
}

#if QWEN_GROUP_ACC
extern void qwen_group_begin(void);
extern void qwen_group_step(const int8_t *, const int8_t *);
extern void qwen_group_load(const int8_t *, const int8_t *);
extern void qwen_group_mac(void);
extern void qwen_group_end(int32_t *);
extern void qwen_group_run8(int32_t *, const int8_t *, uint64_t);
extern void qwen_group_run8_batch(int32_t *, const int8_t *, unsigned,
                                  uint64_t);
extern void qwen_group_session_begin(void);
extern void qwen_group_run8_batch_session(int32_t *, const int8_t *, unsigned,
                                          uint64_t);
extern void qwen_group_run8_split_batch_session(int32_t *, const int8_t *,
                                                const int8_t *, unsigned,
                                                uint64_t);
extern void qwen_group_prefetch_session_begin(void);
extern void qwen_group_run8_split_batch_session_prefetch1(
    int32_t *, const int8_t *, const int8_t *, unsigned, uint64_t);
extern void qwen_group_run8_split_batch_session_prefetch2(
    int32_t *, const int8_t *, const int8_t *, unsigned, uint64_t);
extern void qwen_group_session_end(void);

#if QWEN_DMA_OVERLAP
/* Sources stay live until their completion event is consumed.  Alternate
 * weight buffers are disjoint; AME writes only separate output buffers. */
static void input_submit(const int8_t *source, uintptr_t target, unsigned bytes) {
    (void)qwen_dma_submit((uintptr_t)source, target, bytes, 0);
}
static void input_wait(void) {
    qwen_dma_fence(last_ticket);
}

#if QWEN_DMA_RING
static qwen_spm_buffer input_buffers[QWEN_DMA_BUFFER_COUNT] = {
    { 0xc0010000, 0, QWEN_FREE },
    { 0xc0011000, 0, QWEN_FREE },
    { 0xc0012000, 0, QWEN_FREE },
    { 0xc0013000, 0, QWEN_FREE },
};

/* Output buffers are disjoint from RVV scratch and from both weight slots.
 * An output descriptor is queued before AME starts and waits on the matching
 * AME-done event.  This makes output drain overlap the next input fill. */
static qwen_spm_buffer output_buffers[2] = {
    { 0xc0000800, 0, QWEN_FREE },
    { 0xc0000c00, 0, QWEN_FREE },
};

static qwen_spm_buffer *input_buffer(uintptr_t spm)
{
    for (unsigned i = 0; i < QWEN_DMA_BUFFER_COUNT; ++i)
        if (input_buffers[i].spm == spm) return &input_buffers[i];
    return 0;
}

static unsigned input_buffer_slot(uintptr_t spm)
{
    for (unsigned i = 0; i < QWEN_DMA_BUFFER_COUNT; ++i)
        if (input_buffers[i].spm == spm) return i;
    m5_fail(0, 40);
    return 0;
}

static void buffer_accept_event(qwen_spm_buffer *buffer)
{
    if (!buffer) m5_fail(0, 41);
    if (qwen_buffer_acquire_event(buffer)) m5_fail(0, 42);
    if (qwen_buffer_compute(buffer)) m5_fail(0, 43);
}
#endif

void qwen_dma_blocks_begin(const int8_t *data, uintptr_t spm, unsigned bytes) {
    /* The selected slot enters the ring first. Ownership transfers to DMA until AME
     * consumes the descriptor's completion event. */
    const unsigned slot = input_buffer_slot(spm);
    const uint64_t ready_event = 1ULL << slot;
    const uint64_t ticket = qwen_dma_submit_dep((uintptr_t)data, spm, bytes,
        0, 0, 0, ready_event);
#if QWEN_DMA_RING
    qwen_spm_buffer *buffer = input_buffer(spm);
    if (!buffer || qwen_buffer_fill(buffer, ticket)) m5_fail(0, 44);
#endif
    /* Keep each primed descriptor pending. qwen_dma_blocks_compute() appends
     * the current output, then publishes the batch with one doorbell. The
     * queue-local AME event wait supplies the required dependency. */
}

void qwen_dma_blocks_session_begin(void) {
#if QWEN_AME_OPERAND_PREFETCH
    qwen_group_prefetch_session_begin();
#else
    qwen_group_session_begin();
#endif
}

void qwen_dma_blocks_session_end(void) {
    qwen_group_session_end();
}

uint64_t qwen_dma_blocks_compute(int32_t *out, uintptr_t activation_spm,
    uintptr_t weight_spm, unsigned groups, const int8_t *next,
    uintptr_t next_spm, unsigned next_bytes, unsigned output_slot) {
    /* Select the current slot. Its i+4 refill is submitted only after AME has
     * consumed the ready event and finished reading it. */
#if QWEN_DMA_RING
    qwen_spm_buffer *current_buffer = input_buffer(weight_spm);
#else
    input_wait();
#endif
#if QWEN_DMA_RING
    output_slot &= 1u;
    qwen_spm_buffer *output_buffer = &output_buffers[output_slot];
    const uintptr_t output_spm = output_buffer->spm;
    const uint64_t output_event = 1ULL << (8u + output_slot);
    uint64_t output_ticket = 0;
    if (groups) {
        if (output_buffer->state != QWEN_FREE) m5_fail(0, 46);
        dma_regs[0xa8 / 8] = output_event;
        output_ticket = qwen_dma_submit_dep((uintptr_t)out, output_spm,
                                             groups * 64, 1, 0,
                                             output_event, 0);
        output_buffer->state = QWEN_COMPUTING;
        output_buffer->ticket = output_ticket;
    }
#else
    (void)output_slot;
    const uintptr_t output_spm = 0xc0000800;
    /* The non-ring path copies output synchronously.  Return a non-zero
     * completion token so the common two-slot loop still accumulates it. */
    uint64_t output_ticket = groups ? 1 : 0;
#endif

    /* Current input, next input and dependency-blocked output become visible
     * together.  Ready descriptors may execute out of order around the
     * blocked output descriptor. */
    qwen_dma_flush();

    if (groups) {
        /* The alternating MRF/ACC schedule is defined for one descriptor's
         * one or two K128 groups. */
        if (groups > 2) m5_fail(0, 49);
        const uint64_t ready_event = 1ULL << input_buffer_slot(weight_spm);
        /* One ownership epoch covers every output group in this buffer. */
#if QWEN_OFFLINE_AME_LAYOUT
#if QWEN_AME_OPERAND_PREFETCH
        if (groups == 1)
            qwen_group_run8_split_batch_session_prefetch1(
                                          (int32_t *)output_spm,
                                          (const int8_t *)activation_spm,
                                          (const int8_t *)weight_spm, groups,
                                          ready_event);
        else
            qwen_group_run8_split_batch_session_prefetch2(
                                          (int32_t *)output_spm,
                                          (const int8_t *)activation_spm,
                                          (const int8_t *)weight_spm, groups,
                                          ready_event);
#else
        qwen_group_run8_split_batch_session((int32_t *)output_spm,
                                      (const int8_t *)activation_spm,
                                      (const int8_t *)weight_spm, groups,
                                      ready_event);
#endif
#else
        qwen_group_run8_batch_session((int32_t *)output_spm,
                                      (const int8_t *)weight_spm, groups,
                                      ready_event);
#endif
#if QWEN_DMA_RING
        /* The AME event wait above consumed this buffer's DMA completion;
         * transfer software ownership without polling the same fence again. */
        buffer_accept_event(current_buffer);
#endif

        /* The AME store instructions have completed and published their SPM
         * payload before the C call returns.  Release the prequeued output
         * descriptor without another doorbell or a CPU polling loop. */
#if QWEN_DMA_RING
        dma_regs[0xa0 / 8] = output_event;
        if (qwen_buffer_release(output_buffer, output_ticket)) m5_fail(0, 47);
#else
        qwen_dma_copy((uintptr_t)out, output_spm, groups * 64, 1);
#endif
    }
#if QWEN_DMA_RING
    if (qwen_buffer_release(current_buffer, 0)) m5_fail(0, 48);
#endif

    /* Four buffers carry four initially-ready weight descriptors.  Refill the
     * slot just consumed with batch i+4.  Submission occurs after AME releases
     * ownership of the slot and remains four batches ahead of its consumer. */
    if (next_bytes) {
        const unsigned next_slot = input_buffer_slot(next_spm);
        const uint64_t next_event = 1ULL << next_slot;
        const uint64_t ticket = qwen_dma_submit_dep(
            (uintptr_t)next, next_spm, next_bytes, 0, 0, 0, next_event);
#if QWEN_DMA_RING
        qwen_spm_buffer *next_buffer = input_buffer(next_spm);
        if (!next_buffer || qwen_buffer_fill(next_buffer, ticket))
            m5_fail(0, 45);
#endif
    }
    return output_ticket;
}

void qwen_dma_blocks_wait(uint64_t ticket, unsigned output_slot) {
#if QWEN_DMA_RING
    if (!ticket) return;
    output_slot &= 1u;
    qwen_dma_fence(ticket);
    if (qwen_buffer_reclaim(&dma_ring, &output_buffers[output_slot]) != 1)
        m5_fail(0, 49);
#else
    (void)ticket;
    (void)output_slot;
#endif
}
#endif

void qwen_dma_group(int32_t *out, const int8_t *a, const int8_t *b, unsigned steps) {
    const uintptr_t base = 0xc0000000, sc = 0xc0000800;
#if QWEN_DMA_OVERLAP
    unsigned first = steps < 4 ? steps : 4;
    input_submit(a, base, first * 64);
    input_submit(b, base + 256, first * 64);
#endif
    qwen_group_begin();
    for (unsigned step = 0; step < steps; ++step) {
#if QWEN_DMA_OVERLAP
        const uintptr_t current = base + ((step / 4) & 1) * 512 + (step % 4) * 64;
        if (step % 4 == 0) input_wait();
        qwen_group_load((const int8_t *)current, (const int8_t *)(current + 256));
        if (step % 4 == 0 && step + 4 < steps) {
            const uintptr_t next = base + (((step / 4) + 1) & 1) * 512;
            unsigned count = steps - step - 4 < 4 ? steps - step - 4 : 4;
            input_submit(a + (step + 4) * 64, next, count * 64);
            input_submit(b + (step + 4) * 64, next + 256, count * 64);
        }
        qwen_group_mac();
#else
        const uintptr_t current = base;
        qwen_dma_copy((uintptr_t)(a + step * 64), current, 64, 0);
        qwen_dma_copy((uintptr_t)(b + step * 64), current + 64, 64, 0);
        qwen_group_step((const int8_t *)current, (const int8_t *)(current + 64));
#endif
    }
    qwen_group_end((int32_t *)sc);
    qwen_dma_copy((uintptr_t)out, sc, 64, 1);
}
#endif
