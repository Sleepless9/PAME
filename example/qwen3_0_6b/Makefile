CASE_BINARY := test_llm_qwen3_0_6b
CASE_DESCRIPTION := Qwen3-0.6B real-checkpoint W4A8 full prefill/decode inference
REQUIRED_MNEMONICS := ame.acquire ame.release msettyp asettyp mzero.2d.acc mls.rm mmulacc.2d mmov.m.a mss.rm
TOOLCHAIN_MODE := source
MEM_SIZE := 1GiB
GEM5_CPU := atomic

MODEL_DIR := $(CURDIR)/model
TOKENIZER_DIR := $(CURDIR)/tokenizer
QUANT_DIR := $(CURDIR)/quantization
MODEL_V5 := $(MODEL_DIR)/qwen3-0.6b-int4-v5.bin
MODEL_V6 := $(MODEL_DIR)/qwen3-0.6b-int4-tile-v6.bin
SAFETENSORS := $(MODEL_DIR)/model.safetensors
TOKENIZER_BIN := $(TOKENIZER_DIR)/data/tokenizer.bin

COMMON_DEFINES := \
	-DUSE_AME=1 -DUSE_ZTT=1 \
	-DQWEN_PACKED_K16=1 -DQWEN_GROUP_ACC=1 -DQWEN_ZTT_SQUARES=4 \
	-DGEM5_MINIMAL_PROMPT=1 -DPROMPT_TEXT='"Hello"' \
	-DQWEN_PRETOKENIZED_ID=9707 \
	-DQWEN_PROGRESS_LAYERS=1 \
	-DGEN_STEPS=2 -DMAX_SEQ=4 -DSAMPLE_TOP_K=1 \
	-DBUILD_TIMESTAMP='"real-w4a8"' \
	-I$(CURDIR) -I$(CURDIR)/operators/include -I$(CURDIR)/platform \
	-I$(CURDIR)/tokenizer

HOST_EXTRA_CFLAGS := $(COMMON_DEFINES) -DQWEN_BARE_METAL=0 \
	-DZTT_SCALAR_FALLBACK=1 -ffp-contract=off
TARGET_EXTRA_CFLAGS := $(COMMON_DEFINES) -DQWEN_BARE_METAL=1
TARGET_LINKER_SCRIPT := $(CURDIR)/linker/qwen3.ld
TARGET_EXTRA_LDFLAGS := -Wl,--no-check-sections
HOST_SOURCES := \
	$(CURDIR)/main.c \
	$(CURDIR)/operators.c \
	$(CURDIR)/operators/src/ztt_qwen_backend.c \
	$(CURDIR)/operators/src/ztt_matmul.c \
	$(CURDIR)/linker/model_blob.S \
	$(CURDIR)/linker/tokenizer_blob.S

TARGET_SOURCES := \
	$(CURDIR)/main.c \
	$(CURDIR)/operators.c \
	$(CURDIR)/operators/src/ztt_qwen_backend.c \
	$(CURDIR)/operators/src/ztt_matmul.c \
	$(CURDIR)/operators/src/ztt_matmul.S \
	$(CURDIR)/linker/model_blob.S \
	$(CURDIR)/linker/tokenizer_blob.S \
	$(CURDIR)/platform/gem5_m5ops.S

CASE_DEPENDENCIES := $(MODEL_V6) $(TOKENIZER_BIN) \
	$(CURDIR)/w4_common.h $(CURDIR)/tokenizer/tokenizer_bare.h

include ../common/case_rules.mk

PYTHON ?= /nfs/home/xiangyuanyang/virtualenv/py3_10/bin/python3
ASSET_STAMP := $(GENERATED_DIR)/assets.ok

.PHONY: asset-check quantize-v5 repack-v6 requantize tokenizer-bin full clean-assets minor-run

all: asset-check

asset-check: $(ASSET_STAMP)

$(ASSET_STAMP): $(SAFETENSORS) $(MODEL_V5) $(MODEL_V6) $(TOKENIZER_BIN) \
		$(MODEL_DIR)/SHA256SUMS $(CURDIR)/tools/verify_assets.py
	@mkdir -p $(dir $@)
	$(PYTHON) $(CURDIR)/tools/verify_assets.py $(CURDIR)
	@touch $@

quantize-v5: $(SAFETENSORS) $(MODEL_DIR)/config.json
	cd $(QUANT_DIR) && $(PYTHON) convert_to_int4.py \
		--input $(SAFETENSORS) --config $(MODEL_DIR)/config.json \
		--output $(MODEL_V5) --group-size 128 --scale-method mse \
		--mse-grid 16 --chunk-rows 1024 --smooth-alpha 0.5 \
		--smooth-max-scale 8.0

repack-v6: $(MODEL_V5) $(MODEL_DIR)/config.json
	cd $(QUANT_DIR) && $(PYTHON) repack_to_tile_w4.py \
		--input $(MODEL_V5) --output $(MODEL_V6) \
		--config $(MODEL_DIR)/config.json

requantize: quantize-v5 repack-v6
	sha256sum $(SAFETENSORS) $(MODEL_V5) $(MODEL_V6) $(TOKENIZER_BIN) \
		> $(MODEL_DIR)/SHA256SUMS

tokenizer-bin: $(TOKENIZER_DIR)/data/tokenizer.json
	cd $(QUANT_DIR) && $(PYTHON) convert_tokenizer_bin.py \
		$(TOKENIZER_DIR)/data/tokenizer.json $(TOKENIZER_BIN)

full: test

# Full instruction-by-instruction timing simulation is intentionally separate:
# 18,624,512 logical K16 tiles map one-to-one to physical Ztt MAC
# instructions. MinorCPU adds detailed pipeline timing, while the standard full
# validation uses the same ELF and executes every instruction on AtomicSimpleCPU.
minor-run:
	$(MAKE) run GEM5_CPU=minor

clean-assets:
	rm -f $(MODEL_V5) $(MODEL_V6) $(TOKENIZER_BIN) $(ASSET_STAMP)

TRACE_REPORT := $(REPORT_DIR)/$(CASE_BINARY).minor.trace.txt

.PHONY: trace
trace: run
	@tr -d '\r' < $(LOG_DIR)/minor-run.log | \
		sed -n '/^=== Qwen3-0.6B W4A8+ztt_w4 ===$$/,/^=== done ===$$/p' \
		> $(TRACE_REPORT)
	@test -s $(TRACE_REPORT)
	@grep -Fq 'model: v6 W4A8 h=1024 L=28 V=151936 gs=128 qk=1 i8tile' $(TRACE_REPORT)
	@grep -Fq '[4] prefill_all 1 tokens' $(TRACE_REPORT)
	@grep -Fq '[5] generate 2 tokens' $(TRACE_REPORT)
	@grep -Fq 'layer_passes=56' $(HOST_REPORT)
	@grep -Fq 'layer_passes=56' $(MINOR_REPORT)
	@grep -Fq 'status=PASS' $(MINOR_REPORT)
	@grep -Fq '=== done ===' $(TRACE_REPORT)
	@echo "[PASS] qwen3_0_6b: full real-weight trace $(TRACE_REPORT)"

report: trace
