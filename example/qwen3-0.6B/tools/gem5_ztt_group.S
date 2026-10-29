/* Accumulate all K blocks of one quantization group in int32 Acc0.
 * Each input register holds four 4x4 int8 squares (K=16 per instruction).
 * a0=C, a1=A packed blocks, a2=B packed blocks, a3=number of blocks.
 */
.section .text.qwen_ztt_group,"ax"
.globl qwen_ztt_group
.type qwen_ztt_group,@function
qwen_ztt_group:
    li t1, 0x3c
    .word 0xa49303ab
    li t0, 0x40000008
    .word 0xa232802b
    .word 0xa232822b
    li t0, 0x40000020
    .word 0xa212802b
    .word 0xa600002b
1:
    .word 0xa385802b
    .word 0xa386022b
    .word 0x9440002b
    addi a1, a1, 64
    addi a2, a2, 64
    addi a3, a3, -1
    bnez a3, 1b
    li t0, 0x40000020
    .word 0xa232802b
    .word 0xa3c0002b
    .word 0xa3b5002b
    .word 0xa4a0002b
    ret
.size qwen_ztt_group, .-qwen_ztt_group
