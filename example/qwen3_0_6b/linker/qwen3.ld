OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    ram (rwx) : ORIGIN = 0x80000000, LENGTH = 0x3ffff000
}

SECTIONS
{
    . = ORIGIN(ram);

    .text : ALIGN(16)
    {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > ram

    .rodata : ALIGN(64)
    {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        KEEP(*(.tokenizer_blob))
    } > ram

    .data : ALIGN(16)
    {
        __data_start = .;
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.data .data.*)
        *(.sdata .sdata.*)
        __data_end = .;
    } > ram

    .bss (NOLOAD) : ALIGN(64)
    {
        __bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        __bss_end = .;
    } > ram

    .stack (NOLOAD) : ALIGN(64)
    {
        __stack_bottom = .;
        . += 8M;
        __stack_top = .;
    } > ram

    .model_blob : ALIGN(64)
    {
        KEEP(*(.model_blob))
    } > ram

    . = ALIGN(64);
    __heap_start = .;
    __heap_end = ORIGIN(ram) + LENGTH(ram);

    ASSERT(__heap_start < __heap_end, "Qwen model leaves no heap")

    /DISCARD/ :
    {
        *(.comment)
        *(.note .note.*)
        *(.eh_frame .eh_frame.*)
    }
}
