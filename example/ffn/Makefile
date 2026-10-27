CASE_BINARY := test_llm_ffn
CASE_DESCRIPTION := Two-layer FP32 FFN with base-2 sigmoid GELU approximation
REQUIRED_MNEMONICS := mls.rm mss.rm mmulacc.2d msqrt.ew msub.ew mexp2.ew madd.ew mrec.ew mmul.ew
OPERATOR_CONFIG := $(CURDIR)/config/cases.json
HOST_LDLIBS += -lm
include ../common/case_rules.mk

$(HOST_BINARY) $(TARGET_BINARY): operators.h
