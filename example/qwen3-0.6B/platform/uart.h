// Copyright OpenHW Group contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#define UART_BASE 0x10000000

// Register spacing: 1 byte by default, 4 bytes for FPGA MMIO.
#ifdef FPGA_UART_4BYTE
#define UART_REG_SHIFT 2   // 4-byte spacing
#else
#define UART_REG_SHIFT 0   // 1-byte spacing
#endif

#define UART_RBR              (UART_BASE + (0 << UART_REG_SHIFT))
#define UART_THR              (UART_BASE + (0 << UART_REG_SHIFT))
#define UART_DLAB_LSB         (UART_BASE + (0 << UART_REG_SHIFT))
#define UART_DLAB_MSB         (UART_BASE + (1 << UART_REG_SHIFT))
#define UART_INTERRUPT_ENABLE (UART_BASE + (1 << UART_REG_SHIFT))
#define UART_INTERRUPT_IDENT  (UART_BASE + (2 << UART_REG_SHIFT))
#define UART_FIFO_CONTROL     (UART_BASE + (2 << UART_REG_SHIFT))
#define UART_LINE_CONTROL     (UART_BASE + (3 << UART_REG_SHIFT))
#define UART_MODEM_CONTROL    (UART_BASE + (4 << UART_REG_SHIFT))
#define UART_LINE_STATUS      (UART_BASE + (5 << UART_REG_SHIFT))
#define UART_MODEM_STATUS     (UART_BASE + (6 << UART_REG_SHIFT))

void write_reg_u8(uintptr_t addr, uint8_t value);
uint8_t read_reg_u8(uintptr_t addr);
int is_transmit_empty(void);
char is_transmit_empty_altera(void);
int is_receive_empty(void);
void write_serial(char a);
int read_serial(uint8_t *res);
void init_uart(uint32_t freq, uint32_t baud);
void print_uart(const char *str);
void print_uart_int(uint32_t val);
void print_uart_addr(uint64_t addr);
void print_uart_byte(uint8_t byte);
void bin_to_hex(uint8_t inp, uint8_t res[2]);
extern uint8_t bin_to_hex_table[16];
