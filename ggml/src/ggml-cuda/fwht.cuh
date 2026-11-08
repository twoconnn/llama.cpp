#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// GGML_OP_HADAMARD: Walsh-Hadamard transform along the last dim (n = 2^k), same type in/out.
void ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
