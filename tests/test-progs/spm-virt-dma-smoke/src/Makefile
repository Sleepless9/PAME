CROSS_COMPILE ?= riscv64-unknown-elf-
CC = $(CROSS_COMPILE)gcc
CFLAGS = -march=rv64imac_zicsr -mabi=lp64 -nostdlib -nostartfiles -static
LDFLAGS = -Wl,-Ttext=0x80000000 -Wl,-e,_start
OUT = spm-virt-dma-smoke

.PHONY: all clean

all: $(OUT)

$(OUT): spm-virt-dma-smoke.S
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(OUT)

