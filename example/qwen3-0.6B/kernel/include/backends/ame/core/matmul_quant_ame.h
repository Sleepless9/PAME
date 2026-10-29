#ifndef AME_MATMUL_Q_H
#define AME_MATMUL_Q_H

#include "backends/ops_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

void ame_memcpy_acc2(void *dst, const void *src, size_t nbytes);
void ame_memcpy_acc4(void *dst, const void *src, size_t nbytes);
void ame_memcpy_acc6(void *dst, const void *src, size_t nbytes);
void ame_memcpy_acc8(void *dst, const void *src, size_t nbytes);
void ame_memcpy_i8_acc2(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i8_acc4(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i8_acc6(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i8_acc8(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i32_acc2(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_i32_acc4(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_i32_acc6(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_i32_acc8(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_f32_acc2(float *dst, const float *src, size_t count);
void ame_memcpy_f32_acc4(float *dst, const float *src, size_t count);
void ame_memcpy_f32_acc6(float *dst, const float *src, size_t count);
void ame_memcpy_f32_acc8(float *dst, const float *src, size_t count);

int ame_matmul_abt_i8_i8_f32(const int8_t *A,
                          const int8_t *B,
                          float *C,
                          int m,
                          int n,
                          int k);
int ame_matmul_singleacc_abt_i8_i8_f32(const int8_t *A,
                                        const int8_t *B,
                                        float *C,
                                        int m,
                                        int n,
                                        int k);
int ame_matmul_dualacc_abt_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k);
int ame_matmul_quadacc_abt_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k);

int ame_matmul_ab_i8_i8_f32(const int8_t *A,
                         const int8_t *B,
                         float *C,
                         int m,
                         int n,
                         int k);
int ame_matmul_singleacc_ab_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k);
int ame_matmul_quadacc_ab_i8_i8_f32(const int8_t *A,
                                    const int8_t *B,
                                    float *C,
                                    int m,
                                    int n,
                                    int k);

int ame_probe_matmul_ab_i8_i8_f32_once(const int8_t *A,
                                       const int8_t *B,
                                       float *C,
                                       int req_m,
                                       int req_n,
                                       int req_k,
                                       int *got_m,
                                       int *got_n,
                                       int *got_k);
int ame_probe_matmul_abt_i8_i8_f32_once(const int8_t *A,
                                        const int8_t *B,
                                        float *C,
                                        int req_m,
                                        int req_n,
                                        int req_k,
                                        int *got_m,
                                        int *got_n,
                                        int *got_k);

#ifdef __cplusplus
}
#endif

#endif
