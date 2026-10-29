    .section .rodata, "a", @progbits
    .align 3
    .globl _tokenizer_blob_start
_tokenizer_blob_start:
    .incbin "../tokenizer/tokenizer_data/tokenizer.bin"
    .globl _tokenizer_blob_end
_tokenizer_blob_end:
