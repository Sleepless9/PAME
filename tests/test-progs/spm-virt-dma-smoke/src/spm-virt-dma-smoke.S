/* Exercise two-dimensional, bidirectional SPMVirtDma transfers. */

    .equ DMA_BASE,          0x40010000
    .equ SPM_DATA,          0x90000020
    .equ REG_CONTROL,       0x00
    .equ REG_STATUS,        0x08
    .equ REG_SYSTEM_ADDR,   0x10
    .equ REG_SPM_ADDR,      0x18
    .equ REG_ROW_BYTES,     0x20
    .equ REG_ROWS,          0x28
    .equ REG_SYSTEM_STRIDE, 0x30
    .equ REG_SPM_STRIDE,    0x38
    .equ REG_DONE_BYTES,    0x40
    .equ REG_DONE_ROWS,     0x48
    .equ REG_DESCRIPTOR_ID, 0x50

    .equ ROW_BYTES,         80
    .equ ROWS,              4
    .equ SOURCE_STRIDE,     96
    .equ DEST_STRIDE,       112
    .equ SPM_STRIDE,        128
    .equ PADDING_VALUE,     0xa5

    .section .text.init
    .globl _start
_start:
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* Build deterministic source rows and non-payload padding. */
    la      s1, source
    li      t0, ROWS
    li      t1, 1
1:
    mv      t2, s1
    li      t3, ROW_BYTES
2:
    sb      t1, 0(t2)
    addi    t1, t1, 1
    addi    t2, t2, 1
    addi    t3, t3, -1
    bnez    t3, 2b
    li      t3, SOURCE_STRIDE - ROW_BYTES
    li      t4, 0xcc
3:
    sb      t4, 0(t2)
    addi    t2, t2, 1
    addi    t3, t3, -1
    bnez    t3, 3b
    addi    s1, s1, SOURCE_STRIDE
    addi    t0, t0, -1
    bnez    t0, 1b

    la      s2, destination
    li      t0, ROWS * DEST_STRIDE
    li      t1, PADDING_VALUE
4:
    sb      t1, 0(s2)
    addi    s2, s2, 1
    addi    t0, t0, -1
    bnez    t0, 4b

    /* Descriptor 0x11: DRAM -> SPM. */
    li      s0, DMA_BASE
    la      t0, source
    sd      t0, REG_SYSTEM_ADDR(s0)
    li      t0, SPM_DATA
    sd      t0, REG_SPM_ADDR(s0)
    li      t0, ROW_BYTES
    sd      t0, REG_ROW_BYTES(s0)
    li      t0, ROWS
    sd      t0, REG_ROWS(s0)
    li      t0, SOURCE_STRIDE
    sd      t0, REG_SYSTEM_STRIDE(s0)
    li      t0, SPM_STRIDE
    sd      t0, REG_SPM_STRIDE(s0)
    li      t0, 0x11
    sd      t0, REG_DESCRIPTOR_ID(s0)
    li      t0, 5                  /* START | CLEAR_STATUS */
    fence   iorw, iorw
    sd      t0, REG_CONTROL(s0)
    call    wait_and_check
    bnez    a0, fail

    /* Descriptor 0x12: SPM -> DRAM with a different destination stride. */
    la      t0, destination
    sd      t0, REG_SYSTEM_ADDR(s0)
    li      t0, DEST_STRIDE
    sd      t0, REG_SYSTEM_STRIDE(s0)
    li      t0, 0x12
    sd      t0, REG_DESCRIPTOR_ID(s0)
    li      t0, 7                  /* START | SPM_TO_SYSTEM | CLEAR_STATUS */
    fence   iorw, iorw
    sd      t0, REG_CONTROL(s0)
    call    wait_and_check
    bnez    a0, fail

    /* Compare each payload byte and verify the destination row padding. */
    la      s1, source
    la      s2, destination
    li      t0, ROWS
5:
    li      t1, ROW_BYTES
6:
    lbu     t2, 0(s1)
    lbu     t3, 0(s2)
    bne     t2, t3, fail
    addi    s1, s1, 1
    addi    s2, s2, 1
    addi    t1, t1, -1
    bnez    t1, 6b

    addi    s1, s1, SOURCE_STRIDE - ROW_BYTES
    li      t1, DEST_STRIDE - ROW_BYTES
    li      t2, PADDING_VALUE
7:
    lbu     t3, 0(s2)
    bne     t2, t3, fail
    addi    s2, s2, 1
    addi    t1, t1, -1
    bnez    t1, 7b
    addi    t0, t0, -1
    bnez    t0, 5b

pass:
    li      a0, 0x20
    la      a1, pass_args
    j       semihost_exit

fail:
    li      a0, 0x20
    la      a1, fail_args
    j       semihost_exit

/* Return a0=0 only for DONE, no ERROR, 320 bytes and four rows. */
wait_and_check:
8:
    fence   iorw, iorw
    ld      t0, REG_STATUS(s0)
    andi    t1, t0, 1
    bnez    t1, 8b
    andi    t1, t0, 4
    bnez    t1, 9f
    andi    t1, t0, 2
    beqz    t1, 9f
    ld      t0, REG_DONE_BYTES(s0)
    li      t1, ROW_BYTES * ROWS
    bne     t0, t1, 9f
    ld      t0, REG_DONE_ROWS(s0)
    li      t1, ROWS
    bne     t0, t1, 9f
    li      a0, 0
    ret
9:
    li      a0, 1
    ret

semihost_exit:
    .option push
    .option norvc
    slli    zero, zero, 0x1f
    ebreak
    srai    zero, zero, 7
    .option pop
10:
    j       10b

    .section .rodata
    .align 3
pass_args:
    .dword  0x20026
    .dword  0
fail_args:
    .dword  0x20026
    .dword  1

    .section .bss
    .align 6
source:
    .skip   ROWS * SOURCE_STRIDE
destination:
    .skip   ROWS * DEST_STRIDE
