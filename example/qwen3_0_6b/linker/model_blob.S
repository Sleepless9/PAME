    .section .model_blob, "a", @progbits
    .balign 64
    .globl _model_blob_start
_model_blob_start:
    .incbin "model/qwen3-0.6b-int4-tile-v6.bin"
    .globl _model_blob_end
_model_blob_end:
    .section .note.GNU-stack,"",@progbits
