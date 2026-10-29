    .section .model_blob, "a", @progbits
    .align 6
    .globl _model_blob_start
_model_blob_start:
    .incbin "../core/model_data/qwen3-0.6b-int4-tile.bin"
    .globl _model_blob_end
_model_blob_end:
