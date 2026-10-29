# Ztt Qwen3 W4A8 Overlay

This overlay is migrated from `ISA/test/ztt_llm/qwen3-0.6b/model/ztt-qwen3-w4a8`.
It supplies the AME backend's Ztt W4A8 operator while reusing the Qwen front-end,
model blob, tokenizer, platform, and linker files in the parent example.

The implementation is deliberately prefixed with `ztt`:

- `core/ztt_backend.h` and `core/ztt_qwen_backend.c` keep the shared `ame_*`
  entry points while forwarding to Ztt kernels.
- `kernel/include/ztt_w4a8.h` declares the W4A8 tile API.
- `kernel/src/ztt_matmul.c` implements the tile scheduling and scale handling.
- `kernel/src/ztt_matmul.S` emits `rv64gcv_xxiangshanztt` instructions.

`core/Makefile` links these objects into `qwen3_chat_w4a8_ame_fpga`; the gem5
configuration still selects `AMEEngine`, so Ztt instructions are dispatched
through the AME MinorInstQueue channel and complete through the AME timing model.
