# Standalone RVV + Ztt Qwen3-0.6B guest build.

RESEARCH_ROOT ?= ../../../..
TOOLCHAIN_ROOT ?= $(RESEARCH_ROOT)
override RESEARCH_ROOT := $(abspath $(RESEARCH_ROOT))
override TOOLCHAIN_ROOT := $(abspath $(TOOLCHAIN_ROOT))
LLVM_BUILD ?= $(TOOLCHAIN_ROOT)/llvm-project_riscv/build_AME
CLANG ?= $(if $(wildcard $(LLVM_BUILD)/bin/clang-21),$(LLVM_BUILD)/bin/clang-21,clang-21)
OBJCOPY ?= $(if $(wildcard $(LLVM_BUILD)/bin/llvm-objcopy),$(LLVM_BUILD)/bin/llvm-objcopy,riscv64-linux-gnu-objcopy)
CC := $(CLANG) --target=riscv64-linux-gnu --sysroot=/usr/riscv64-linux-gnu --gcc-toolchain=/usr

BUILD ?= build
QWEN_DIR := ..
CORE_DIR := $(QWEN_DIR)/core
KERNEL_DIR := $(QWEN_DIR)/kernel
PLATFORM_DIR := $(QWEN_DIR)/platform
TOKENIZER_DIR := $(QWEN_DIR)/tokenizer
LINK_DIR := $(QWEN_DIR)/linker_boot
MODEL_DIR := $(CORE_DIR)/model_data

RVV_VLEN_BITS ?= 512
RVV_VBIT_FLAGS ?= -mllvm -riscv-v-vector-bits-min=$(RVV_VLEN_BITS) -mllvm -riscv-v-vector-bits-max=$(RVV_VLEN_BITS)
NOVEC_FLAGS ?= -fno-vectorize -fno-slp-vectorize
ZTT_MARCH ?= rv64gcv_xxiangshanztt
FPGA_CLOCK_HZ ?= 50000000
BUILD_TIMESTAMP ?= $(shell date '+%Y-%m-%d %H:%M:%S %z')
BARE_INJECT ?=
UART_REG_FLAGS ?=
SPM_BASE_ADDR ?= 0x70000000ULL
SPM_DMA_BASE_ADDR ?= 0x40010000ULL

INC_DIRS := -Iinclude -Isrc -I$(CORE_DIR) -I$(KERNEL_DIR)/include \
            -I$(PLATFORM_DIR) -I$(TOKENIZER_DIR) -I$(LINK_DIR)
COMMON_FLAGS := -O3 -std=gnu11 -Wall -Wextra -Wno-unused-parameter \
                -mabi=lp64d -mcmodel=medany -nostdlib -fno-builtin \
                -fno-pie -static $(UART_REG_FLAGS) \
                -DFPGA_CLOCK_HZ=$(FPGA_CLOCK_HZ) \
                -DBUILD_TIMESTAMP='"$(BUILD_TIMESTAMP)"' $(INC_DIRS)
HYBRID_FLAGS := $(COMMON_FLAGS) -march=$(ZTT_MARCH) \
                -DUSE_AME=1 -DUSE_RVV=1 -DUSE_ZTT=1 \
                -DSPM_BASE_ADDR=$(SPM_BASE_ADDR) \
                -DSPM_DMA_BASE_ADDR=$(SPM_DMA_BASE_ADDR) \
                $(RVV_VBIT_FLAGS) $(NOVEC_FLAGS) $(BARE_INJECT)
QWEN_FLAGS := $(HYBRID_FLAGS) -include include/qwen_hybrid_prelude.h
BLOB_FLAGS := $(COMMON_FLAGS) -march=rv64imafdc $(BARE_INJECT)

TARGET := $(BUILD)/qwen3_chat_w4a8_hybrid_fpga
LDSCRIPT := $(LINK_DIR)/test_compact.ld
MODEL_BLOB := $(MODEL_DIR)/qwen3-0.6b-int4-tile.bin
TOKENIZER_BLOB := $(TOKENIZER_DIR)/tokenizer_data/tokenizer.bin

OBJS := \
    $(BUILD)/qwen_hybrid.o \
    $(BUILD)/ztt_qwen_backend.o \
    $(BUILD)/ztt_matmul_spm.o \
    $(BUILD)/ztt_matmul_asm.o \
    $(BUILD)/matmul_quant_rvv_tile.o \
    $(BUILD)/ops_mix.o \
    $(BUILD)/ops_rvv_aux.o \
    $(BUILD)/ops_rvv_core.o \
    $(BUILD)/ops_rvv_rope.o \
    $(BUILD)/ops_rvv_reduce.o \
    $(BUILD)/ops_rvv_elementwise.o \
    $(BUILD)/ops_scalar_elementwise.o \
    $(BUILD)/hybrid_flash_attention.o \
    $(BUILD)/spm_virt_dma.o \
    $(BUILD)/hybrid_runtime.o \
    $(BUILD)/crt_uart.o \
    $(BUILD)/uart.o \
    $(BUILD)/bare_syscalls_uart.o \
    $(BUILD)/gem5_m5ops.o \
    $(BUILD)/model_blob.o \
    $(BUILD)/tokenizer_blob.o

.PHONY: all clean bin smoke

all: $(TARGET)
bin: $(TARGET).bin

SMOKE_BUILD ?= smoke-build
SMOKE_TARGET := $(SMOKE_BUILD)/hybrid_smoke
SMOKE_OBJS := $(SMOKE_BUILD)/hybrid_smoke.o \
              $(SMOKE_BUILD)/ztt_matmul_asm.o \
              $(SMOKE_BUILD)/spm_virt_dma.o \
              $(SMOKE_BUILD)/crt_uart.o \
              $(SMOKE_BUILD)/uart.o \
              $(SMOKE_BUILD)/bare_syscalls_uart.o \
              $(SMOKE_BUILD)/gem5_m5ops.o

smoke: $(SMOKE_TARGET)

$(TARGET): $(OBJS) $(LDSCRIPT) | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -T $(LDSCRIPT) -o $@ $(filter-out $(BUILD)/gem5_m5ops.o,$(OBJS)) $(BUILD)/gem5_m5ops.o -lgcc

$(BUILD)/qwen_hybrid.o: $(CORE_DIR)/qwen3_bare_w4a8_prefill_all.c \
        src/qwen_hybrid_backend.inc include/qwen_hybrid_prelude.h | $(BUILD)
	$(CC) $(QWEN_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ztt_qwen_backend.o: $(CORE_DIR)/ztt_qwen_backend.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ztt_matmul_spm.o: src/ztt_matmul_spm.c include/spm_virt_dma.h | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ztt_matmul_asm.o: $(KERNEL_DIR)/src/ztt_matmul.S | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(BUILD)/matmul_quant_rvv_tile.o: $(KERNEL_DIR)/src/backends/rvv/core/matmul_quant_rvv_tile.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_mix.o: $(KERNEL_DIR)/src/backends/mix/core/ops_mix.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_rvv_aux.o: $(KERNEL_DIR)/src/backends/rvv/core/ops_rvv_aux.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_rvv_core.o: $(KERNEL_DIR)/src/backends/rvv/core/ops_rvv_core.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_rvv_rope.o: $(KERNEL_DIR)/src/backends/rvv/core/ops_rvv_rope.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_rvv_reduce.o: $(KERNEL_DIR)/src/backends/rvv/reduce/ops_rvv_reduce.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_rvv_elementwise.o: $(KERNEL_DIR)/src/backends/rvv/elementwise/ops_rvv_elementwise.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/ops_scalar_elementwise.o: $(KERNEL_DIR)/src/backends/scalar/elementwise/ops_scalar_elementwise.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/hybrid_flash_attention.o: src/hybrid_flash_attention.c include/hybrid_flash_attention.h | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/spm_virt_dma.o: src/spm_virt_dma.c include/spm_virt_dma.h | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/hybrid_runtime.o: src/hybrid_runtime.c include/qwen_hybrid_runtime.h | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/crt_uart.o: $(LINK_DIR)/crt_uart.S $(LINK_DIR)/encoding.h | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(BUILD)/uart.o: $(PLATFORM_DIR)/uart.c $(PLATFORM_DIR)/uart.h | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(BUILD)/bare_syscalls_uart.o: $(PLATFORM_DIR)/bare_syscalls_uart.c | $(BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(BUILD)/gem5_m5ops.o: $(LINK_DIR)/gem5_m5ops.S | $(BUILD)
	$(CC) $(BLOB_FLAGS) -c -o $@ $<

$(BUILD)/model_blob.o: $(LINK_DIR)/model_blob_int4_tile.S $(MODEL_BLOB) | $(BUILD)
	$(CC) $(BLOB_FLAGS) -c -o $@ $<

$(BUILD)/tokenizer_blob.o: $(LINK_DIR)/tokenizer_blob.S $(TOKENIZER_BLOB) | $(BUILD)
	$(CC) $(BLOB_FLAGS) -c -o $@ $<

$(TARGET).bin: $(TARGET)
	$(OBJCOPY) -O binary $< $@

$(SMOKE_TARGET): $(SMOKE_OBJS) $(LDSCRIPT) | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -T $(LDSCRIPT) -o $@ $(filter-out $(SMOKE_BUILD)/gem5_m5ops.o,$(SMOKE_OBJS)) $(SMOKE_BUILD)/gem5_m5ops.o -lgcc

$(SMOKE_BUILD)/hybrid_smoke.o: smoke/hybrid_smoke.c include/spm_virt_dma.h | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(SMOKE_BUILD)/ztt_matmul_asm.o: $(KERNEL_DIR)/src/ztt_matmul.S | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(SMOKE_BUILD)/spm_virt_dma.o: src/spm_virt_dma.c include/spm_virt_dma.h | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -MMD -MP -c -o $@ $<

$(SMOKE_BUILD)/crt_uart.o: $(LINK_DIR)/crt_uart.S $(LINK_DIR)/encoding.h | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(SMOKE_BUILD)/uart.o: $(PLATFORM_DIR)/uart.c $(PLATFORM_DIR)/uart.h | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(SMOKE_BUILD)/bare_syscalls_uart.o: $(PLATFORM_DIR)/bare_syscalls_uart.c | $(SMOKE_BUILD)
	$(CC) $(HYBRID_FLAGS) -c -o $@ $<

$(SMOKE_BUILD)/gem5_m5ops.o: $(LINK_DIR)/gem5_m5ops.S | $(SMOKE_BUILD)
	$(CC) $(BLOB_FLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $@

$(SMOKE_BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD) $(SMOKE_BUILD)

-include $(BUILD)/*.d $(SMOKE_BUILD)/*.d
