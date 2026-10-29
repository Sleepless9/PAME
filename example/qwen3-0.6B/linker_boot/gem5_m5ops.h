#ifndef QWEN_GEM5_M5OPS_H
#define QWEN_GEM5_M5OPS_H

#include <stdint.h>

void m5_exit(uint64_t delay);
void m5_fail(uint64_t delay, uint64_t code);
void m5_checkpoint(uint64_t ns_delay, uint64_t ns_period);
void m5_work_begin(uint64_t work_id, uint64_t thread_id);
void m5_work_end(uint64_t work_id, uint64_t thread_id);

#endif
