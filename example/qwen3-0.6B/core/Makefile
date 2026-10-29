# Makefile for Qwen3-0.6B W4A8 bare-metal FPGA builds.
#
# Default:
#   make                              -> fpga_all
#
# FPGA ELF targets (offline-chat: compile-time prompt + chat template, single
# prefill_all + generate, UART output, no OS). The default `make` builds both
# flavors AND their flat .bin exports.
#
#   make chat_ame_fpga                -> build/qwen3_chat_w4a8_ame_fpga[.bin]
#       AME path. Ztt tile instructions + int8 GEMM via ztt_matmul.{c,S}.
#   make chat_scalar_fpga             -> build/qwen3_chat_w4a8_scalar_fpga[.bin]
#       Scalar W4A8 reference. Same int4 tile blob.
#
# Multi-turn UART chat REPL (-DCHAT_BUILD=1):
#   make chat_repl_all                -> both REPL ELFs + .bins
#   make chat_repl_ame_fpga           -> build/qwen3_chat_repl_w4a8_ame_fpga[.bin]
#   make chat_repl_scalar_fpga        -> build/qwen3_chat_repl_w4a8_scalar_fpga[.bin]
#
# .bin-only target groups:
#   make bins                         -> both single-shot .bin files
#   make chat_repl_bins               -> both REPL .bin files
#   make chat_ame_fpga_bin            -> just one .bin
#
#   make clean
#
# Toolchain: clang-21 (same as qwen3 chat FPGA build).

RESEARCH_ROOT ?= ../../../..
TOOLCHAIN_ROOT ?= $(RESEARCH_ROOT)
override RESEARCH_ROOT := $(abspath $(RESEARCH_ROOT))
override TOOLCHAIN_ROOT := $(abspath $(TOOLCHAIN_ROOT))
LLVM_BUILD ?= $(TOOLCHAIN_ROOT)/llvm-project_riscv/build_AME
CLANG    ?= $(if $(wildcard $(LLVM_BUILD)/bin/clang-21),$(LLVM_BUILD)/bin/clang-21,clang-21)
OBJCOPY  ?= $(if $(wildcard $(LLVM_BUILD)/bin/llvm-objcopy),$(LLVM_BUILD)/bin/llvm-objcopy,riscv64-linux-gnu-objcopy)
CC       := $(CLANG) --target=riscv64-linux-gnu --sysroot=/usr/riscv64-linux-gnu --gcc-toolchain=/usr
PY ?= python3

BUILD    ?= build
KERNEL_DIR ?= ../kernel
W4_KERNEL_DIR ?= ../kernel
PLAT_DIR := ../platform
TOK_DIR  := ../tokenizer
LINK_DIR := ../linker_boot
MODEL_DIR := model_data
TOKDATA_DIR := $(TOK_DIR)/tokenizer_data
W4_SCALAR_SRC := ../kernel/src/scalar/matmul_quant_scalar_w4.c

INC_DIRS := -I. -I$(KERNEL_DIR)/include -I$(W4_KERNEL_DIR)/include -I$(PLAT_DIR) -I$(TOK_DIR) -I$(LINK_DIR)

.PHONY: all fpga_all bins clean \
       chat_rvv_fpga    chat_rvv_fpga_bin \
       chat_ame_fpga    chat_ame_fpga_bin \
       chat_scalar_fpga chat_scalar_fpga_bin \
       chat_repl_all chat_repl_bins \
       chat_repl_ame_fpga    chat_repl_ame_fpga_bin \
       chat_repl_scalar_fpga chat_repl_scalar_fpga_bin \
       dma_sram_qual_fpga dma_sram_qual_fpga_bin \
       dma_sram_qual_fast_fpga dma_sram_qual_fast_fpga_bin \
       dma_sram_wave_fpga dma_sram_wave_fpga_elf dma_sram_wave_fpga_bin

all: fpga_all

# ============================
# Shared flag groups
# ============================

RVV_VLEN_BITS  ?= 512
RVV_VBIT_FLAGS ?= -mllvm -riscv-v-vector-bits-min=$(RVV_VLEN_BITS) -mllvm -riscv-v-vector-bits-max=$(RVV_VLEN_BITS)
NOVEC_FLAGS    ?= -fno-vectorize -fno-slp-vectorize
ZTT_MARCH      ?= rv64gcv_xxiangshanztt
BUILD_TIMESTAMP ?= $(shell date '+%Y-%m-%d %H:%M:%S %z')
AME_MIX_PROFILE ?= 1
BARE_INJECT ?=
UART_REG_FLAGS ?=
# Hardware clock used by UART divisor setup and cycle-based chat metrics.
# Override on 25MHz boards with: make ... FPGA_CLOCK_HZ=25000000
FPGA_CLOCK_HZ ?= 50000000

BARE_COMMON := -O3 -std=gnu11 -Wall -Wextra -Wno-unused-parameter \
               -mabi=lp64d -mcmodel=medany -nostdlib -fno-builtin \
               -fno-pie -static $(UART_REG_FLAGS) -DFPGA_CLOCK_HZ=$(FPGA_CLOCK_HZ) \
               -DBUILD_TIMESTAMP='"$(BUILD_TIMESTAMP)"' $(INC_DIRS)

BARE_AME_CFLAGS     := $(BARE_COMMON) -march=$(ZTT_MARCH) -DUSE_AME=1 $(BARE_INJECT)
BARE_AME_RVV_CFLAGS := $(filter-out -march=%,$(BARE_AME_CFLAGS)) -march=$(ZTT_MARCH) -DAME_MIX_PROFILE=$(AME_MIX_PROFILE) $(RVV_VBIT_FLAGS) $(NOVEC_FLAGS)
BARE_RVV_CFLAGS     := $(filter-out -march=%,$(BARE_COMMON)) -march=rv64gcv -DUSE_RVV=1 $(BARE_INJECT) $(RVV_VBIT_FLAGS) $(NOVEC_FLAGS)
BARE_SCALAR_CFLAGS  := $(BARE_COMMON) -march=rv64imafdc $(BARE_INJECT)
BARE_BLOB_CFLAGS    := $(BARE_COMMON) -march=rv64imafdc $(BARE_INJECT)

BARE_LDSCRIPT := $(LINK_DIR)/test_compact.ld
BARE_LDLIBS := $(BUILD)/gem5_m5ops.o -lgcc
CHAT_REPL_DEF := -DCHAT_BUILD=1

AME_OPS_OBJS := $(BUILD)/kernel_ops_mix_fpga.o \
                $(BUILD)/kernel_rvv_aux_fpga.o
# The AME execution model uses the Ztt operator implementation.  Keep every
# migrated source explicitly ztt-prefixed; the wrapper preserves the AME API
# consumed by the shared Qwen front-end while instructions are Ztt encoded.
ZTT_KERNEL_OBJS := $(BUILD)/ztt_qwen_backend_fpga.o \
                   $(BUILD)/ztt_matmul_fpga.o \
                   $(BUILD)/ztt_matmul_asm_fpga.o \
                   $(AME_OPS_OBJS)
AME_KERNEL_OBJS := $(ZTT_KERNEL_OBJS)
RVV_KERNEL_OBJS := $(BUILD)/matmul_quant_rvv_tile_fpga.o \
                   $(BUILD)/kernel_ops_mix_rvv_fpga.o \
                   $(BUILD)/kernel_rvv_aux_rvv_fpga.o
SCALAR_KERNEL_OBJS := $(BUILD)/matmul_quant_scalar_w4_fpga.o

fpga_all: chat_ame_fpga    chat_ame_fpga_bin \
          chat_scalar_fpga chat_scalar_fpga_bin

bins: chat_ame_fpga_bin chat_scalar_fpga_bin

# ============================
# W4A8 + RVV, offline-chat, gem5
# ============================
chat_rvv_fpga:     $(BUILD)/qwen3_chat_w4a8_rvv_fpga
chat_rvv_fpga_bin: $(BUILD)/qwen3_chat_w4a8_rvv_fpga.bin

$(BUILD)/qwen3_chat_w4a8_rvv_fpga: $(BUILD)/qwen3_chat_w4a8_rvv_fpga.o \
                                    $(BUILD)/bare_crt_uart_rvv_fpga.o \
                                    $(BUILD)/bare_uart_rvv_fpga.o \
                                    $(BUILD)/bare_syscalls_uart_rvv_fpga.o \
                                    $(RVV_KERNEL_OBJS) \
                                    $(BUILD)/model_blob_int4_tile.o \
                                    $(BUILD)/tokenizer_blob_rvv_fpga.o \
                                    $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_rvv_fpga.o \
		$(BUILD)/qwen3_chat_w4a8_rvv_fpga.o \
		$(BUILD)/bare_uart_rvv_fpga.o \
		$(BUILD)/bare_syscalls_uart_rvv_fpga.o \
		$(RVV_KERNEL_OBJS) \
		$(BUILD)/model_blob_int4_tile.o \
		$(BUILD)/tokenizer_blob_rvv_fpga.o $(BARE_LDLIBS)

$(BUILD)/qwen3_chat_w4a8_rvv_fpga.o: qwen3_bare_w4a8_prefill_all.c \
                                     w4_common.h \
                                     $(TOK_DIR)/tokenizer_bare.h \
                                     $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                                     | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/matmul_quant_rvv_tile_fpga.o: \
        $(KERNEL_DIR)/src/backends/rvv/core/matmul_quant_rvv_tile.c \
        $(KERNEL_DIR)/include/backends/rvv/core/matmul_quant_rvv_tile.h | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/kernel_ops_mix_rvv_fpga.o: $(KERNEL_DIR)/src/backends/mix/core/ops_mix.c \
                                    $(KERNEL_DIR)/include/backends/mix/core/ops_mix.h \
                                    $(KERNEL_DIR)/include/backends/ops_lib.h | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/kernel_rvv_aux_rvv_fpga.o: $(KERNEL_DIR)/src/backends/rvv/core/ops_rvv_aux.c \
                                    $(KERNEL_DIR)/include/backends/mix/core/ops_mix.h \
                                    $(KERNEL_DIR)/include/backends/ops_lib.h | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/bare_crt_uart_rvv_fpga.o: $(LINK_DIR)/crt_uart.S $(LINK_DIR)/encoding.h | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -c -o $@ $<

$(BUILD)/bare_uart_rvv_fpga.o: $(PLAT_DIR)/uart.c $(PLAT_DIR)/uart.h | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -c -o $@ $<

$(BUILD)/bare_syscalls_uart_rvv_fpga.o: $(PLAT_DIR)/bare_syscalls_uart.c \
                                           $(PLAT_DIR)/uart.h \
                                           $(LINK_DIR)/gem5_m5ops.h \
                                           $(BUILD)/gem5_m5ops.o | $(BUILD)
	$(CC) $(BARE_RVV_CFLAGS) -c -o $@ $<

$(BUILD)/tokenizer_blob_rvv_fpga.o: $(LINK_DIR)/tokenizer_blob.S $(TOKDATA_DIR)/tokenizer.bin | $(BUILD)
	$(CC) $(BARE_BLOB_CFLAGS) -c -o $@ $<

$(BUILD)/qwen3_chat_w4a8_rvv_fpga.bin: $(BUILD)/qwen3_chat_w4a8_rvv_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

# ============================
# W4A8 + AME, offline-chat, FPGA
# ============================
chat_ame_fpga:     $(BUILD)/qwen3_chat_w4a8_ame_fpga
chat_ame_fpga_bin: $(BUILD)/qwen3_chat_w4a8_ame_fpga.bin

$(BUILD)/qwen3_chat_w4a8_ame_fpga: $(BUILD)/qwen3_chat_w4a8_ame_fpga.o \
                                    $(BUILD)/bare_crt_uart_ame_fpga.o \
                                    $(BUILD)/bare_uart_ame_fpga.o \
                                    $(BUILD)/bare_syscalls_uart_ame_fpga.o \
                                    $(AME_KERNEL_OBJS) \
                                    $(BUILD)/model_blob_int4_tile.o \
                                    $(BUILD)/tokenizer_blob_ame_fpga.o \
                                    $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_ame_fpga.o \
		$(BUILD)/qwen3_chat_w4a8_ame_fpga.o \
		$(BUILD)/bare_uart_ame_fpga.o \
		$(BUILD)/bare_syscalls_uart_ame_fpga.o \
		$(AME_KERNEL_OBJS) \
		$(BUILD)/model_blob_int4_tile.o \
		$(BUILD)/tokenizer_blob_ame_fpga.o $(BARE_LDLIBS)

$(BUILD)/qwen3_chat_w4a8_ame_fpga.o: qwen3_bare_w4a8_prefill_all.c \
                                     ame_mix_offline_profile.h w4_common.h \
                                     $(TOK_DIR)/tokenizer_bare.h \
                                     $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                                     | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/qwen3_chat_w4a8_ame_fpga.bin: $(BUILD)/qwen3_chat_w4a8_ame_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

$(BUILD)/ztt_qwen_backend_fpga.o: ztt_qwen_backend.c \
                                  ztt_backend.h ame_mix_offline_profile.h \
                                  $(KERNEL_DIR)/include/ztt_w4a8.h | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ztt_matmul_fpga.o: $(KERNEL_DIR)/src/ztt_matmul.c \
                            $(KERNEL_DIR)/include/ztt_w4a8.h | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ztt_matmul_asm_fpga.o: $(KERNEL_DIR)/src/ztt_matmul.S | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -c -o $@ $<

$(BUILD)/kernel_ops_mix_fpga.o: $(KERNEL_DIR)/src/backends/mix/core/ops_mix.c \
                                $(KERNEL_DIR)/include/backends/mix/core/ops_mix.h \
                                $(KERNEL_DIR)/include/backends/ops_lib.h | $(BUILD)
	$(CC) $(BARE_AME_RVV_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/kernel_rvv_aux_fpga.o: $(KERNEL_DIR)/src/backends/rvv/core/ops_rvv_aux.c \
                                $(KERNEL_DIR)/include/backends/mix/core/ops_mix.h \
                                $(KERNEL_DIR)/include/backends/ops_lib.h | $(BUILD)
	$(CC) $(BARE_AME_RVV_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/bare_crt_uart_ame_fpga.o: $(LINK_DIR)/crt_uart.S $(LINK_DIR)/encoding.h | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -c -o $@ $<

$(BUILD)/bare_uart_ame_fpga.o: $(PLAT_DIR)/uart.c $(PLAT_DIR)/uart.h | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -c -o $@ $<

$(BUILD)/bare_syscalls_uart_ame_fpga.o: $(PLAT_DIR)/bare_syscalls_uart.c \
                                            $(PLAT_DIR)/uart.h \
                                            $(LINK_DIR)/gem5_m5ops.h \
                                            $(BUILD)/gem5_m5ops.o | $(BUILD)
	$(CC) $(BARE_AME_RVV_CFLAGS) -c -o $@ $<

$(BUILD)/tokenizer_blob_ame_fpga.o: $(LINK_DIR)/tokenizer_blob.S $(TOKDATA_DIR)/tokenizer.bin | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -c -o $@ $<

# ============================
# DMA/SRAM qualification gate, FPGA
# ============================
dma_sram_qual_fpga:     $(BUILD)/dma_sram_qual_fpga
dma_sram_qual_fpga_bin: $(BUILD)/dma_sram_qual_fpga.bin
dma_sram_qual_fast_fpga:     $(BUILD)/dma_sram_qual_fast_fpga
dma_sram_qual_fast_fpga_bin: $(BUILD)/dma_sram_qual_fast_fpga.bin

$(BUILD)/dma_sram_qual_fpga: $(BUILD)/dma_sram_qual_fpga.o \
                             $(BUILD)/bare_crt_uart_ame_fpga.o \
                             $(BUILD)/bare_uart_ame_fpga.o \
                             $(BUILD)/bare_syscalls_uart_ame_fpga.o \
                             $(AME_KERNEL_OBJS) \
                             $(BUILD)/model_blob_int4_tile.o \
                             $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_ame_fpga.o \
		$(BUILD)/dma_sram_qual_fpga.o \
		$(BUILD)/bare_uart_ame_fpga.o \
		$(BUILD)/bare_syscalls_uart_ame_fpga.o \
		$(AME_KERNEL_OBJS) \
		$(BUILD)/model_blob_int4_tile.o $(BARE_LDLIBS)

$(BUILD)/dma_sram_qual_fpga.o: dma_sram_qual_fpga.c \
                               ame_mix_offline_profile.h w4_common.h \
                               $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                               | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/dma_sram_qual_fpga.bin: $(BUILD)/dma_sram_qual_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

$(BUILD)/dma_sram_qual_fast_fpga: $(BUILD)/dma_sram_qual_fast_fpga.o \
                                  $(BUILD)/bare_crt_uart_ame_fpga.o \
                                  $(BUILD)/bare_uart_ame_fpga.o \
                                  $(BUILD)/bare_syscalls_uart_ame_fpga.o \
                                  $(AME_KERNEL_OBJS) \
                                  $(BUILD)/model_blob_int4_tile.o \
                                  $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_ame_fpga.o \
		$(BUILD)/dma_sram_qual_fast_fpga.o \
		$(BUILD)/bare_uart_ame_fpga.o \
		$(BUILD)/bare_syscalls_uart_ame_fpga.o \
		$(AME_KERNEL_OBJS) \
		$(BUILD)/model_blob_int4_tile.o $(BARE_LDLIBS)

$(BUILD)/dma_sram_qual_fast_fpga.o: dma_sram_qual_fpga.c \
                                    ame_mix_offline_profile.h w4_common.h \
                                    $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                                    | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -DDMA_SRAM_QUAL_FAST=1 -MMD -MP -c -o $@ $<

$(BUILD)/dma_sram_qual_fast_fpga.bin: $(BUILD)/dma_sram_qual_fast_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

dma_sram_wave_fpga:     $(BUILD)/dma_sram_wave_fpga.elf $(BUILD)/dma_sram_wave_fpga
dma_sram_wave_fpga_elf: $(BUILD)/dma_sram_wave_fpga.elf
dma_sram_wave_fpga_bin: $(BUILD)/dma_sram_wave_fpga.bin

$(BUILD)/dma_sram_wave_fpga.elf: $(BUILD)/dma_sram_wave_fpga.o \
                                 $(BUILD)/bare_crt_uart_scalar_fpga.o \
                                 $(BUILD)/bare_uart_scalar_fpga.o \
                                 $(BUILD)/bare_syscalls_uart_scalar_fpga.o \
                                 $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_scalar_fpga.o \
		$(BUILD)/dma_sram_wave_fpga.o \
		$(BUILD)/bare_uart_scalar_fpga.o \
		$(BUILD)/bare_syscalls_uart_scalar_fpga.o $(BARE_LDLIBS)

$(BUILD)/dma_sram_wave_fpga: $(BUILD)/dma_sram_wave_fpga.elf | $(BUILD)
	cp $< $@

$(BUILD)/dma_sram_wave_fpga.o: dma_sram_wave_fpga.c \
                               $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                               | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/dma_sram_wave_fpga.bin: $(BUILD)/dma_sram_wave_fpga.elf | $(BUILD)
	$(OBJCOPY) -O binary $< $@

# ============================
# W4A8 + scalar, offline-chat, FPGA
# ============================
chat_scalar_fpga:     $(BUILD)/qwen3_chat_w4a8_scalar_fpga
chat_scalar_fpga_bin: $(BUILD)/qwen3_chat_w4a8_scalar_fpga.bin

$(BUILD)/qwen3_chat_w4a8_scalar_fpga: $(BUILD)/qwen3_chat_w4a8_scalar_fpga.o \
                                      $(BUILD)/bare_crt_uart_scalar_fpga.o \
                                      $(BUILD)/bare_uart_scalar_fpga.o \
                                      $(BUILD)/bare_syscalls_uart_scalar_fpga.o \
                                      $(SCALAR_KERNEL_OBJS) \
                                      $(BUILD)/model_blob_int4_tile.o \
                                      $(BUILD)/tokenizer_blob_scalar_fpga.o \
                                      $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_scalar_fpga.o \
		$(BUILD)/qwen3_chat_w4a8_scalar_fpga.o \
		$(BUILD)/matmul_quant_scalar_w4_fpga.o \
		$(BUILD)/bare_uart_scalar_fpga.o \
		$(BUILD)/bare_syscalls_uart_scalar_fpga.o \
		$(BUILD)/model_blob_int4_tile.o \
		$(BUILD)/tokenizer_blob_scalar_fpga.o $(BARE_LDLIBS)

$(BUILD)/qwen3_chat_w4a8_scalar_fpga.o: qwen3_bare_w4a8_prefill_all.c w4_common.h \
                                       $(TOK_DIR)/tokenizer_bare.h \
                                       $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                                       | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/matmul_quant_scalar_w4_fpga.o: $(W4_SCALAR_SRC) w4_common.h | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/qwen3_chat_w4a8_scalar_fpga.bin: $(BUILD)/qwen3_chat_w4a8_scalar_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

$(BUILD)/bare_crt_uart_scalar_fpga.o: $(LINK_DIR)/crt_uart.S $(LINK_DIR)/encoding.h | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -c -o $@ $<

$(BUILD)/bare_uart_scalar_fpga.o: $(PLAT_DIR)/uart.c $(PLAT_DIR)/uart.h | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -c -o $@ $<

$(BUILD)/bare_syscalls_uart_scalar_fpga.o: $(PLAT_DIR)/bare_syscalls_uart.c \
                                               $(PLAT_DIR)/uart.h \
                                               $(LINK_DIR)/gem5_m5ops.h \
                                               $(BUILD)/gem5_m5ops.o | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -c -o $@ $<

$(BUILD)/tokenizer_blob_scalar_fpga.o: $(LINK_DIR)/tokenizer_blob.S $(TOKDATA_DIR)/tokenizer.bin | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -c -o $@ $<

# ============================
# Shared blob rules
# ============================
$(BUILD)/model_blob_int4_tile.o: $(LINK_DIR)/model_blob_int4_tile.S \
                                 $(MODEL_DIR)/qwen3-0.6b-int4-tile.bin | $(BUILD)
	$(CC) $(BARE_BLOB_CFLAGS) -c -o $@ $<

$(BUILD)/gem5_m5ops.o: $(LINK_DIR)/gem5_m5ops.S | $(BUILD)
	$(CC) $(BARE_BLOB_CFLAGS) -c -o $@ $<

$(MODEL_DIR)/qwen3-0.6b-int4-tile.bin:
	@test -f "$@" || { \
		echo "Missing prepacked model blob: $@" >&2; \
		false; \
	}

$(TOKDATA_DIR)/tokenizer.bin:
	@test -f "$@" || { \
		echo "Missing packed tokenizer blob: $@" >&2; \
		false; \
	}

$(BUILD):
	mkdir -p $(BUILD)

# ============================
# Chat REPL flavors (multi-turn UART chat: -DCHAT_BUILD=1).
# Same backends/blob/link recipe as the single-shot chat_* targets.
# ============================
chat_repl_all: chat_repl_ame_fpga    chat_repl_ame_fpga_bin \
               chat_repl_scalar_fpga chat_repl_scalar_fpga_bin

chat_repl_bins: chat_repl_ame_fpga_bin chat_repl_scalar_fpga_bin

chat_repl_ame_fpga:     $(BUILD)/qwen3_chat_repl_w4a8_ame_fpga
chat_repl_ame_fpga_bin: $(BUILD)/qwen3_chat_repl_w4a8_ame_fpga.bin

$(BUILD)/qwen3_chat_repl_w4a8_ame_fpga: $(BUILD)/qwen3_chat_repl_w4a8_ame_fpga.o \
                                        $(BUILD)/bare_crt_uart_ame_fpga.o \
                                        $(BUILD)/bare_uart_ame_fpga.o \
                                        $(BUILD)/bare_syscalls_uart_ame_fpga.o \
                                        $(AME_KERNEL_OBJS) \
                                        $(BUILD)/model_blob_int4_tile.o \
                                        $(BUILD)/tokenizer_blob_ame_fpga.o \
                                        $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_ame_fpga.o \
		$(BUILD)/qwen3_chat_repl_w4a8_ame_fpga.o \
		$(BUILD)/bare_uart_ame_fpga.o \
		$(BUILD)/bare_syscalls_uart_ame_fpga.o \
		$(AME_KERNEL_OBJS) \
		$(BUILD)/model_blob_int4_tile.o \
		$(BUILD)/tokenizer_blob_ame_fpga.o $(BARE_LDLIBS)

$(BUILD)/qwen3_chat_repl_w4a8_ame_fpga.o: qwen3_bare_w4a8_prefill_all.c \
                                          ame_mix_offline_profile.h w4_common.h \
                                          $(TOK_DIR)/tokenizer_bare.h \
                                          $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                                          | $(BUILD)
	$(CC) $(BARE_AME_CFLAGS) $(CHAT_REPL_DEF) -MMD -MP -c -o $@ $<

$(BUILD)/qwen3_chat_repl_w4a8_ame_fpga.bin: $(BUILD)/qwen3_chat_repl_w4a8_ame_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

chat_repl_scalar_fpga:     $(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga
chat_repl_scalar_fpga_bin: $(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga.bin

$(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga: $(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga.o \
                                           $(BUILD)/bare_crt_uart_scalar_fpga.o \
                                           $(BUILD)/bare_uart_scalar_fpga.o \
                                           $(BUILD)/bare_syscalls_uart_scalar_fpga.o \
                                           $(SCALAR_KERNEL_OBJS) \
                                           $(BUILD)/model_blob_int4_tile.o \
                                           $(BUILD)/tokenizer_blob_scalar_fpga.o \
                                           $(BARE_LDSCRIPT) | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) -T $(BARE_LDSCRIPT) -o $@ \
		$(BUILD)/bare_crt_uart_scalar_fpga.o \
		$(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga.o \
		$(BUILD)/matmul_quant_scalar_w4_fpga.o \
		$(BUILD)/bare_uart_scalar_fpga.o \
		$(BUILD)/bare_syscalls_uart_scalar_fpga.o \
		$(BUILD)/model_blob_int4_tile.o \
		$(BUILD)/tokenizer_blob_scalar_fpga.o $(BARE_LDLIBS)

$(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga.o: qwen3_bare_w4a8_prefill_all.c w4_common.h \
                                             $(TOK_DIR)/tokenizer_bare.h \
                                             $(PLAT_DIR)/uart_helper.c $(PLAT_DIR)/uart.h \
                                             | $(BUILD)
	$(CC) $(BARE_SCALAR_CFLAGS) $(CHAT_REPL_DEF) -MMD -MP -c -o $@ $<

$(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga.bin: $(BUILD)/qwen3_chat_repl_w4a8_scalar_fpga | $(BUILD)
	$(OBJCOPY) -O binary $< $@

clean:
	@rm -rf $(BUILD) 2>/dev/null || true
	@if [ -d "$(BUILD)" ]; then \
		find "$(BUILD)" -mindepth 1 -maxdepth 1 ! -name '.nfs*' -exec rm -rf {} + 2>/dev/null || true; \
		rmdir "$(BUILD)" 2>/dev/null || true; \
	fi

-include $(BUILD)/*.d
