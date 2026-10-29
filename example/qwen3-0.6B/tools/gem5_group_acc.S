    .text
    .align 2
    .globl qwen_group_begin
qwen_group_begin:
    li t1, 0x3c
    ame.acquire t2, t1
    li t0, 0x40000008
    msettyp m0, t0
    msettyp m4, t0
    li t0, 0x40000020
    asettyp acc0, t0
    mzero.2d.acc acc0
    ret

    .globl qwen_group_step
qwen_group_step:
    mls.rm m0, a0
    mls.rm m4, a1
    mmulacc.2d acc0, m0, m4
    ret

    .globl qwen_group_load
qwen_group_load:
    mls.rm m0, a0
    mls.rm m4, a1
    ret

    .globl qwen_group_mac
qwen_group_mac:
    mmulacc.2d acc0, m0, m4
    ret

    .globl qwen_group_end
qwen_group_end:
    li t0, 0x40000020
    msettyp m0, t0
    mmov.m.a m0, acc0
    mss.rm m0, a0
    ame.release
    ret

    /* K=128 macro command: eight packed K=16 operands are laid out as
     * [A:64B, B:64B].  Keeping this straight-line removes seven host calls,
     * the C loop branch, and repeated step*128 address calculations. */
    .globl qwen_group_run8
qwen_group_run8:
    mv t5, a0
    mv t3, a1
    slli t1, a2, 8
    ori t1, t1, 0x3c
    ame.acquire t2, t1
    li t0, 0x40000008
    msettyp m0, t0
    msettyp m1, t0
    msettyp m2, t0
    msettyp m3, t0
    msettyp m4, t0
    msettyp m5, t0
    msettyp m6, t0
    msettyp m7, t0
    msettyp m8, t0
    msettyp m9, t0
    msettyp m10, t0
    msettyp m11, t0
    msettyp m12, t0
    msettyp m13, t0
    msettyp m14, t0
    msettyp m15, t0
    li t0, 0x40000020
    asettyp acc0, t0
    msettyp m16, t0
    mzero.2d.acc acc0
    mls.rm m0, t3
    addi t4, t3, 64
    mls.rm m8, t4
    addi t4, t3, 128
    mls.rm m1, t4
    addi t4, t3, 192
    mls.rm m9, t4
    addi t4, t3, 256
    mls.rm m2, t4
    addi t4, t3, 320
    mls.rm m10, t4
    addi t4, t3, 384
    mls.rm m3, t4
    addi t4, t3, 448
    mls.rm m11, t4
    addi t4, t3, 512
    mls.rm m4, t4
    addi t4, t3, 576
    mls.rm m12, t4
    addi t4, t3, 640
    mls.rm m5, t4
    addi t4, t3, 704
    mls.rm m13, t4
    addi t4, t3, 768
    mls.rm m6, t4
    addi t4, t3, 832
    mls.rm m14, t4
    addi t4, t3, 896
    mls.rm m7, t4
    addi t4, t3, 960
    mls.rm m15, t4
    /* Internal funct7=0x49, rs2=m8, rs1=m0, rd=acc0, CUSTOM-1. */
    .4byte 0x9280002b
    mmov.m.a m16, acc0
    mss.rm m16, t5
    ame.release
    ret

    /* Variable group batch (normally four Qwen output tiles).  Acquire and
     * release bracket the whole SPM buffer, while each group keeps its K=128
     * accumulation local.  Arguments: output SPM, input SPM, groups, event. */
    .globl qwen_group_run8_batch
qwen_group_run8_batch:
    beqz a2, .Lbatch_done
    mv t5, a0
    mv t3, a1
    mv t6, a2
    slli t1, a3, 8
    ori t1, t1, 0x3c
    ame.acquire t2, t1
    li t0, 0x40000008
    msettyp m0, t0
    msettyp m1, t0
    msettyp m2, t0
    msettyp m3, t0
    msettyp m4, t0
    msettyp m5, t0
    msettyp m6, t0
    msettyp m7, t0
    msettyp m8, t0
    msettyp m9, t0
    msettyp m10, t0
    msettyp m11, t0
    msettyp m12, t0
    msettyp m13, t0
    msettyp m14, t0
    msettyp m15, t0
    li t0, 0x40000020
    asettyp acc0, t0
    msettyp m16, t0
.Lbatch_group:
    mzero.2d.acc acc0
    mls.rm m0, t3
    addi t4, t3, 64
    mls.rm m8, t4
    addi t4, t3, 128
    mls.rm m1, t4
    addi t4, t3, 192
    mls.rm m9, t4
    addi t4, t3, 256
    mls.rm m2, t4
    addi t4, t3, 320
    mls.rm m10, t4
    addi t4, t3, 384
    mls.rm m3, t4
    addi t4, t3, 448
    mls.rm m11, t4
    addi t4, t3, 512
    mls.rm m4, t4
    addi t4, t3, 576
    mls.rm m12, t4
    addi t4, t3, 640
    mls.rm m5, t4
    addi t4, t3, 704
    mls.rm m13, t4
    addi t4, t3, 768
    mls.rm m6, t4
    addi t4, t3, 832
    mls.rm m14, t4
    addi t4, t3, 896
    mls.rm m7, t4
    addi t4, t3, 960
    mls.rm m15, t4
    addi t3, t3, 1024
    .4byte 0x9280002b
    mmov.m.a m16, acc0
    mss.rm m16, t5
    addi t5, t5, 64
    addi t6, t6, -1
    bnez t6, .Lbatch_group
    ame.release
.Lbatch_done:
    ret

    /* Persistent AME session used by the complete Qwen invocation.  Matrix
     * type state is invariant across K=128 projections, so ownership and
     * configuration remain live across all layers, prefill and decode. */
    .globl qwen_group_session_begin
qwen_group_session_begin:
    li t1, 0x3c
    ame.acquire t2, t1
    li t0, 0x40000008
    msettyp m0, t0
    msettyp m1, t0
    msettyp m2, t0
    msettyp m3, t0
    msettyp m4, t0
    msettyp m5, t0
    msettyp m6, t0
    msettyp m7, t0
    msettyp m8, t0
    msettyp m9, t0
    msettyp m10, t0
    msettyp m11, t0
    msettyp m12, t0
    msettyp m13, t0
    msettyp m14, t0
    msettyp m15, t0
    li t0, 0x40000020
    asettyp acc0, t0
    msettyp m16, t0
    ret

    /* RD is repurposed as a two-bit immediate event ID because event_wait has
     * no integer result.  Fixed encodings avoid an O3 scalar-snapshot race
     * while retaining a non-serializing accelerator queue dependency. */
    .macro qwen_wait_event mask, tmp
    li \tmp, 1
    beq \mask, \tmp, 1f
    li \tmp, 2
    beq \mask, \tmp, 2f
    li \tmp, 4
    beq \mask, \tmp, 3f
    .4byte 0xa4b001ab             /* ame.event.wait id=3 */
    j 4f
1:  .4byte 0xa4b0002b             /* ame.event.wait id=0 */
    j 4f
2:  .4byte 0xa4b000ab             /* ame.event.wait id=1 */
    j 4f
3:  .4byte 0xa4b0012b             /* ame.event.wait id=2 */
4:
    .endm

    /* Compute one ready DMA buffer while retaining AME ownership and type
     * state. */
    .globl qwen_group_run8_batch_session
qwen_group_run8_batch_session:
    beqz a2, .Lsession_batch_done
    mv t5, a0
    mv t3, a1
    mv t6, a2
    qwen_wait_event a3, t0
.Lsession_batch_group:
    mzero.2d.acc acc0
    mls.rm m0, t3
    addi t4, t3, 64
    mls.rm m8, t4
    addi t4, t3, 128
    mls.rm m1, t4
    addi t4, t3, 192
    mls.rm m9, t4
    addi t4, t3, 256
    mls.rm m2, t4
    addi t4, t3, 320
    mls.rm m10, t4
    addi t4, t3, 384
    mls.rm m3, t4
    addi t4, t3, 448
    mls.rm m11, t4
    addi t4, t3, 512
    mls.rm m4, t4
    addi t4, t3, 576
    mls.rm m12, t4
    addi t4, t3, 640
    mls.rm m5, t4
    addi t4, t3, 704
    mls.rm m13, t4
    addi t4, t3, 768
    mls.rm m6, t4
    addi t4, t3, 832
    mls.rm m14, t4
    addi t4, t3, 896
    mls.rm m7, t4
    addi t4, t3, 960
    mls.rm m15, t4
    addi t3, t3, 1024
    .4byte 0x9280002b
    mmov.m.a m16, acc0
    mss.rm m16, t5
    addi t5, t5, 64
    addi t6, t6, -1
    bnez t6, .Lsession_batch_group
.Lsession_batch_done:
    ret

    /* DMA-ready v7 path. A and B live in distinct SPM regions.  A is packed
     * once by RVV and reused for every output quartet; B arrives directly
     * from the offline-packed model through the DMA ring.
     * a0=output, a1=A packed, a2=B packed, a3=groups, a4=DMA event. */
    .globl qwen_group_run8_split_batch_session
qwen_group_run8_split_batch_session:
    beqz a3, .Lsplit_batch_done
    mv t5, a0
    mv t2, a1
    mv t3, a2
    mv t6, a3
    qwen_wait_event a4, t0
.Lsplit_batch_group:
    mzero.2d.acc acc0
    mls.rm m0, t2
    addi t4, t2, 64
    mls.rm m1, t4
    addi t4, t2, 128
    mls.rm m2, t4
    addi t4, t2, 192
    mls.rm m3, t4
    addi t4, t2, 256
    mls.rm m4, t4
    addi t4, t2, 320
    mls.rm m5, t4
    addi t4, t2, 384
    mls.rm m6, t4
    addi t4, t2, 448
    mls.rm m7, t4
    mls.rm m8, t3
    addi t4, t3, 64
    mls.rm m9, t4
    addi t4, t3, 128
    mls.rm m10, t4
    addi t4, t3, 192
    mls.rm m11, t4
    addi t4, t3, 256
    mls.rm m12, t4
    addi t4, t3, 320
    mls.rm m13, t4
    addi t4, t3, 384
    mls.rm m14, t4
    addi t4, t3, 448
    mls.rm m15, t4
    addi t2, t2, 512
    addi t3, t3, 512
    .4byte 0x9280002b
    mmov.m.a m16, acc0
    mss.rm m16, t5
    addi t5, t5, 64
    addi t6, t6, -1
    bnez t6, .Lsplit_batch_group
.Lsplit_batch_done:
    ret

    /* Two physical operand banks for cross-group preload:
     * bank 0 = m0..m15, bank 1 = m16..m31.  Loads for the next K=128 group
     * enter the AME queue immediately after the current K128 command, so the
     * M-LSU fills the idle bank while the systolic array is active. */
    .globl qwen_group_prefetch_session_begin
qwen_group_prefetch_session_begin:
    li t1, 0x3c
    ame.acquire t2, t1
    li t0, 0x40000008
    msettyp m0, t0
    msettyp m1, t0
    msettyp m2, t0
    msettyp m3, t0
    msettyp m4, t0
    msettyp m5, t0
    msettyp m6, t0
    msettyp m7, t0
    msettyp m8, t0
    msettyp m9, t0
    msettyp m10, t0
    msettyp m11, t0
    msettyp m12, t0
    msettyp m13, t0
    msettyp m14, t0
    msettyp m15, t0
    msettyp m16, t0
    msettyp m17, t0
    msettyp m18, t0
    msettyp m19, t0
    msettyp m20, t0
    msettyp m21, t0
    msettyp m22, t0
    msettyp m23, t0
    msettyp m24, t0
    msettyp m25, t0
    msettyp m26, t0
    msettyp m27, t0
    msettyp m28, t0
    msettyp m29, t0
    msettyp m30, t0
    msettyp m31, t0
    li t0, 0x40000020
    asettyp acc0, t0
    asettyp acc1, t0
    ret

    .macro qwen_load_bank0 aptr, bptr, tmp
    mls.rm m0, \aptr
    addi \tmp, \aptr, 64
    mls.rm m1, \tmp
    addi \tmp, \aptr, 128
    mls.rm m2, \tmp
    addi \tmp, \aptr, 192
    mls.rm m3, \tmp
    addi \tmp, \aptr, 256
    mls.rm m4, \tmp
    addi \tmp, \aptr, 320
    mls.rm m5, \tmp
    addi \tmp, \aptr, 384
    mls.rm m6, \tmp
    addi \tmp, \aptr, 448
    mls.rm m7, \tmp
    mls.rm m8, \bptr
    addi \tmp, \bptr, 64
    mls.rm m9, \tmp
    addi \tmp, \bptr, 128
    mls.rm m10, \tmp
    addi \tmp, \bptr, 192
    mls.rm m11, \tmp
    addi \tmp, \bptr, 256
    mls.rm m12, \tmp
    addi \tmp, \bptr, 320
    mls.rm m13, \tmp
    addi \tmp, \bptr, 384
    mls.rm m14, \tmp
    addi \tmp, \bptr, 448
    mls.rm m15, \tmp
    .endm

    .macro qwen_load_bank1 aptr, bptr, tmp
    mls.rm m16, \aptr
    addi \tmp, \aptr, 64
    mls.rm m17, \tmp
    addi \tmp, \aptr, 128
    mls.rm m18, \tmp
    addi \tmp, \aptr, 192
    mls.rm m19, \tmp
    addi \tmp, \aptr, 256
    mls.rm m20, \tmp
    addi \tmp, \aptr, 320
    mls.rm m21, \tmp
    addi \tmp, \aptr, 384
    mls.rm m22, \tmp
    addi \tmp, \aptr, 448
    mls.rm m23, \tmp
    mls.rm m24, \bptr
    addi \tmp, \bptr, 64
    mls.rm m25, \tmp
    addi \tmp, \bptr, 128
    mls.rm m26, \tmp
    addi \tmp, \bptr, 192
    mls.rm m27, \tmp
    addi \tmp, \bptr, 256
    mls.rm m28, \tmp
    addi \tmp, \bptr, 320
    mls.rm m29, \tmp
    addi \tmp, \bptr, 384
    mls.rm m30, \tmp
    addi \tmp, \bptr, 448
    mls.rm m31, \tmp
    .endm

    .macro qwen_drain bankreg, accreg
    li t0, 0x40000020
    msettyp \bankreg, t0
    mmov.m.a \bankreg, \accreg
    mss.rm \bankreg, t5
    li t0, 0x40000008
    msettyp \bankreg, t0
    addi t5, t5, 64
    .endm

    /* Keep the one- and two-group schedules as separate straight-line entry
     * points.  The two-group branch used to sit behind the long bank-0 K128
     * operation, giving O3 enough time to submit wrong-path AME commands.
     * Both entry points begin with the DMA event barrier, so a mispredicted C
     * call cannot expose operand loads before the call path is committed. */
    .globl qwen_group_run8_split_batch_session_prefetch1
qwen_group_run8_split_batch_session_prefetch1:
    mv t5, a0
    mv t2, a1
    mv t3, a2
    qwen_wait_event a4, t0
    qwen_load_bank0 t2, t3, t4
    mzero.2d.acc acc0
    .4byte 0x9280002b              /* K128 bank 0: A=m0, B=m8. */
    qwen_drain m0, acc0
    /* Scalar readback is the sole O3-visible completion token. */
    mgettyp t0, m0
    ret

    .globl qwen_group_run8_split_batch_session_prefetch2
qwen_group_run8_split_batch_session_prefetch2:
    mv t5, a0
    mv t2, a1
    mv t3, a2
    qwen_wait_event a4, t0

    qwen_load_bank0 t2, t3, t4
    addi t2, t2, 512
    addi t3, t3, 512
    mzero.2d.acc acc0
    .4byte 0x9280002b              /* K128 bank 0: A=m0, B=m8. */

    /* Bank-1 load descriptors execute in the MLSU while bank-0 occupies the
     * systolic array.  A distinct accumulator removes the false WAW/RAW edge. */
    qwen_load_bank1 t2, t3, t4   /* overlaps the preceding bank-0 K128 */
    qwen_drain m0, acc0
    mzero.2d.acc acc1
    .4byte 0x938800ab              /* K128 bank 1 -> acc1. */
    qwen_drain m16, acc1
    mgettyp t0, m16
    ret

    .globl qwen_group_session_end
qwen_group_session_end:
    ame.release
    ret
