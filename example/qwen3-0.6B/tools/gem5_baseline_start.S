    .section .text.init,"ax"
    .globl _start
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, __stack_top
    li t0, 0x1806600
    csrs mstatus, t0
    la t0, baseline_trap
    csrw mtvec, t0
    call main
    li a0, 0
    call m5_exit
1:  j 1b
    .align 2
baseline_trap:
    csrr a0, mcause
    csrr a1, mepc
    li a2, 0
    call handle_trap
    li a0, 0
    li a1, 99
    call m5_fail
    j baseline_trap
