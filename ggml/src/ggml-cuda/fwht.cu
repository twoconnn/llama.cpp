#include "common.cuh"
#include "convert.cuh"
#include "fwht.cuh"

// wide FWHT blocks use one row per thread block with this many threads
#define GGML_CUDA_FWHT_BLOCK_NT 256

template <int N, typename T, typename D>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, D * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Wide blocks: one row per thread block instead of per warp, so each thread keeps N/NT
// values rather than N/32. Stages below the warp width still shuffle, those up to the
// block width go through shared memory, and the rest stay in registers.
template <int N, int NT, typename T, typename D>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, D * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

// Small sizes (n < 64): one thread per row, butterfly in registers.
template <typename T, typename D>
__global__ void hadamard_cuda_small(const T * src, D * dst, const int n, const int64_t n_rows, const float scale) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * n;
    dst += r * n;

    float buf[64];
    for (int j = 0; j < n; ++j) {
        buf[j] = ggml_cuda_cast<float>(src[j]) * scale;
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        for (int blk = 0; blk < n; blk += len) {
            for (int j = blk; j < blk + half; ++j) {
                const float a = buf[j];
                const float b = buf[j + half];
                buf[j]        = a + b;
                buf[j + half] = a - b;
            }
        }
    }
    for (int j = 0; j < n; ++j) {
        dst[j] = ggml_cuda_cast<D>(buf[j]);
    }
}
template <typename T, typename D>
static void ggml_cuda_fwht_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T * src_d = (const T *) src->data;
    D *       dst_d = (D *)       dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    // small sizes: one thread per row
    if (n < 64) {
        const int64_t nthreads = 256;
        const int64_t nblocks  = (rows + nthreads - 1) / nthreads;
        dim3 grid_dims_s(nblocks, 1, 1);
        dim3 block_dims_s(nthreads, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_s =
            ggml_cuda_kernel_launch_params(grid_dims_s, block_dims_s, 0, stream);
        ggml_cuda_kernel_launch(hadamard_cuda_small<T, D>, launch_params_s, src_d, dst_d, n, rows, scale);
        return;
    }

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T, D>, launch_params, src_d, dst_d, rows, scale);
            return;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, T, D>, launch_params, src_d, dst_d, rows, scale);
            return;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, T, D>, launch_params, src_d, dst_d, rows, scale);
            return;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, T, D>, launch_params, src_d, dst_d, rows, scale);
            return;
        default:
            break;
    }

    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;

        dim3 grid_dims_w(rows, 1, 1);
        dim3 block_dims_w(nt, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_w =
            ggml_cuda_kernel_launch_params(grid_dims_w, block_dims_w, 0, stream);

        switch (n) {
            case 1024:
                ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T, D>, launch_params_w, src_d, dst_d, rows, scale);
                return;
            case 2048:
                ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T, D>, launch_params_w, src_d, dst_d, rows, scale);
                return;
            case 4096:
                ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T, D>, launch_params_w, src_d, dst_d, rows, scale);
                return;
#if !defined(GGML_USE_MUSA)
            // 32 KB of shared memory, above the MUSA limit; falls back there
            case 8192:
                ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T, D>, launch_params_w, src_d, dst_d, rows, scale);
                return;
#endif // !defined(GGML_USE_MUSA)
            default:
                GGML_ABORT("fwht: unsupported size %d", n);
        }
    }
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_cuda_fwht_launch<T, float>(ctx, src, dst);
    return true;
}

void ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->type == src->type);

    switch (src->type) {
        case GGML_TYPE_F32:
            ggml_cuda_fwht_launch<float, float>(ctx, src, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_fwht_launch<half, half>(ctx, src, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_fwht_launch<nv_bfloat16, nv_bfloat16>(ctx, src, dst);
            break;
        default:
            GGML_ABORT("hadamard: unsupported type %d", src->type);
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_cuda_op_fwht_impl<half>(ctx, src, dst);
        default:
            return false;
    }
}
