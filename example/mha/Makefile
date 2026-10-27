CASE_BINARY := test_llm_mha
CASE_DESCRIPTION := FP32 multi-head projections, scaled base-2 Softmax, and weighted values
REQUIRED_MNEMONICS := mls.rm mss.rm mmulacc.2d mmulbtacc.2d mmax.ew msub.ew mexp2.ew mreduceadd.row mrec.ew mmul.ew
OPERATOR_CONFIG := $(CURDIR)/config/cases.json
HOST_LDLIBS += -lm
include ../common/case_rules.mk

$(HOST_BINARY) $(TARGET_BINARY): operators.h
