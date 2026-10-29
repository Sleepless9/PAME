    .text
    .align 2
    .globl ztt_i8_i8_i32_4x4
    .type ztt_i8_i8_i32_4x4,@function
/*
 * ztt_i8_i8_i32_4x4(int32_t *c, const int8_t *a, const int8_t *b)
 *
 * A and B are packed row-major 4x4 tiles.  The result is an int32 4x4 tile.
 * Use Ztt v0.6 mnemonics so the assembler and disassembler validate the same
 * instruction spelling and operand order as ISA/test/ztt_direct.
 */
ztt_i8_i8_i32_4x4:
    li      t1, 0x3c
    ame.acquire t2, t1
    li      t0, 0x40000008          # signed int8
    msettyp m0, t0
    msettyp m4, t0
    li      t0, 0x40000020          # signed int32 accumulator
    asettyp acc0, t0
    mzero.2d.acc acc0
    mls.rm  m0, a1
    mls.rm  m4, a2
    mmulacc.2d acc0, m0, m4
    msettyp m0, t0
    mmov.m.a m0, acc0
    mss.rm  m0, a0
    ame.release
    ret
    .size ztt_i8_i8_i32_4x4, .-ztt_i8_i8_i32_4x4
