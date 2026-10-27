/* RISC-V register-ABI gem5 pseudo-ops used by the standalone Qwen guest. */

.macro gem5_m5op name, func
    .globl \name
    .type \name, @function
\name:
    .long 0x0000007b | (\func << 25)
    ret
    .size \name, . - \name
.endm

.section .text
gem5_m5op m5_exit,       0x21
gem5_m5op m5_fail,       0x22
gem5_m5op m5_checkpoint, 0x43
gem5_m5op m5_work_begin, 0x5a
gem5_m5op m5_work_end,   0x5b
