OUTPUT_ARCH("riscv")
ENTRY(_start)

/*
 * Qwen3 bare-metal layout:
 * - Keep code/data at the reset region starting at 0x86400000.
 * - Place the embedded model blob immediately after code/data so flat binary
 *   loaders do not need to materialize a large address hole.
 * - Put the heap after the model blob, using the remaining RAM up to
 *   0xbfffffff.
 */

RAM_ORIGIN = 0x86400000;
RAM_END = 0xc0000000;
RAM_LENGTH = RAM_END - RAM_ORIGIN;

/* Reserve the 1 MiB below the image for the single-hart stack/TLS. */
__stack_bottom = RAM_ORIGIN - 0x100000;
__stack_top = RAM_ORIGIN;

PHDRS
{
  text  PT_LOAD FLAGS(5);
  data  PT_LOAD FLAGS(6);
  model PT_LOAD FLAGS(4);
}

MEMORY
{
  RAM   (rwx) : ORIGIN = RAM_ORIGIN, LENGTH = RAM_LENGTH
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
  .rela.dyn  : { *(.rela.dyn) *(.rela.*) } > RAM :text

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

  .model_blob ALIGN(64) : { KEEP(*(.model_blob)) } > RAM :model
  . = ALIGN(64);
  __model_end = .;

  __heap_start = __model_end;
  __heap_end = RAM_END;

  ASSERT(__heap_start < __heap_end, "No heap space left after model blob")
}
