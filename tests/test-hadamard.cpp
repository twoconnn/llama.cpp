// Smoke test for GGML_OP_HADAMARD: butterfly vs explicit Sylvester matrix.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>

int main() {
    struct ggml_init_params ip = { 256*1024*1024, NULL, false };
    ggml_threadpool_t nul = NULL;
    for (int n : {2, 4, 8, 16, 64, 128, 512}) {
        struct ggml_init_params ipn = { 64*1024*1024, NULL, false };
        struct ggml_context * ctx = ggml_init(ipn);

        // one row of n floats, random
        std::mt19937 rng(42);
        std::vector<float> x(n);
        for (auto & v : x) v = ((float)(int)(rng() % 2000) - 1000.0f) / 100.0f;

        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        memcpy(a->data, x.data(), n*sizeof(float));

        struct ggml_tensor * y = ggml_hadamard(ctx, a);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        // reference: explicit Sylvester H (normalized)
        std::vector<float> ref(n, 0.0f);
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int s = __builtin_popcount((unsigned)(j & k)) & 1;
                ref[j] += (s ? -1.0f : 1.0f) * x[k];
            }
            ref[j] /= std::sqrt((float)n);
        }

        const float * out = (const float *) y->data;
        float max_err = 0;
        for (int j = 0; j < n; ++j) {
            float e = fabsf(out[j] - ref[j]);
            if (e > max_err) max_err = e;
        }
        printf("n=%4d  max_err=%.5f  %s\n", n, max_err, max_err < 1e-3 ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    return 0;
}