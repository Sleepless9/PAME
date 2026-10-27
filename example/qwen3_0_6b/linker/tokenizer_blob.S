    .section .tokenizer_blob, "a", @progbits
    .balign 64
    .globl _tokenizer_blob_start
_tokenizer_blob_start:
    .incbin "tokenizer/data/tokenizer.bin"
    .globl _tokenizer_blob_end
_tokenizer_blob_end:
    .section .note.GNU-stack,"",@progbits
