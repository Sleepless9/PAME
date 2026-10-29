#include "backends/ops_lib.h"

/* AME backend rope entry.
 *
 * AME matmul instructions target i8/i32 matrix math, while RoPE requires
 * fp32 trig/rotate semantics. Keep a dedicated AME-namespaced entry so the
 * AME flavor has an explicit backend hook, and use the scalar reference math
 * for correctness.
 */
void ame_rope_rotate_f32(float *v, int head_dim, int pos, float theta) {
  scalar_rope_rotate_f32(v, head_dim, pos, theta);
}
