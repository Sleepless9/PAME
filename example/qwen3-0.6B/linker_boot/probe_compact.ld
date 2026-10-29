OUTPUT_ARCH("riscv")
ENTRY(_start)

/* AME GEMM probe layout: tiny standalone test, no embedded model blob.
 * Single RAM segment from 0x86400000 (256 MiB is plenty). */

RAM_ORIGIN = 0x86400000;
RAM_LENGTH = 0x10000000;   /* 256 MiB */

PHDRS
{
  text  PT_LOAD FLAGS(5);
  data  PT_LOAD FLAGS(6);
}

MEMORY
{
  RAM (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
}

UART0_BASE = 0x10000000;

SECTIONS
{
  . = ORIGIN(RAM);

  /DISCARD/ : { *(.note.gnu.build-id) }
  .text.init : { *(.text.init) } > RAM :text
  .text      : { *(.text .text.*) } > RAM :text
  .rodata    : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM :text
  .rodata.str1.1 : { *(.rodata.str1.1) } > RAM :text

  . = ALIGN(64);
  .data : { *(.data .data.*) } > RAM :data

  . = ALIGN(16);
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM :data

  . = ALIGN(16);
  .bss (NOLOAD) : {
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.bss .bss.*)
    *(COMMON)
  } > RAM :data

  .tdata : { _tdata_begin = .; *(.tdata .tdata.*) _tdata_end = .; } > RAM :data
  .tbss  (NOLOAD) : { *(.tbss .tbss.*) _tbss_end = .; } > RAM :data

  . = ALIGN(64);
  _end = .;
}
