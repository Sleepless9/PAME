CASE_BINARY := test_llm_softmax
CASE_DESCRIPTION := Row-wise base-2 Softmax
REQUIRED_MNEMONICS := mmax.ew msub.ew mexp2.ew mreduceadd.row mrec.ew mmul.ew
OPERATOR_CONFIG := $(CURDIR)/config/cases.json
HOST_LDLIBS += -lm
include ../common/case_rules.mk

$(HOST_BINARY) $(TARGET_BINARY): operators.h
