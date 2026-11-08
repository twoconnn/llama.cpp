// Smoke test: GGML_OP_HADAMARD on the CUDA backend vs CPU reference.
// Build: g++ -std=c++17 -O2 -I ggml/include -I ggml/src tests/test-hadamard-cuda.cpp \
//   -L build/bin -lggml -lggml-base -lggml-cpu -lggml-cuda -o build/bin/test-hadamard-cuda
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static double max_diff(const float * a, const float * b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

static std::vector<float> run_on(ggml_backend_t backend, const float * in, int64_t n, int64_t rows) {
    struct ggml_init_params ip = { 2*ggml_tensor_overhead() + 256*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
    struct ggml_tensor * h = ggml_hadamard(ctx, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    ggml_backend_tensor_set(x, in, 0, ggml_nbytes(x));

    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, h);
    ggml_backend_graph_compute(backend, g);

    std::vector<float> out(n*rows);
    ggml_backend_tensor_get(h, out.data(), 0, ggml_nbytes(h));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main() {
    ggml_backend_dev_t dev_gpu = nullptr;
    ggml_backend_dev_t dev_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if ((ggml_backend_dev_type(d) == (enum ggml_backend_dev_type)1 ||
             ggml_backend_dev_type(d) == (enum ggml_backend_dev_type)2) && !dev_gpu) dev_gpu = d;
        if (ggml_backend_dev_type(d) == (enum ggml_backend_dev_type)0 && !dev_cpu) dev_cpu = d;
    }
    if (!dev_gpu || !dev_cpu) { printf("FAIL: backends not found\n"); return 1; }
    printf("gpu: %s\n", ggml_backend_dev_description(dev_gpu));

    ggml_backend_t backend     = ggml_backend_dev_init(dev_gpu, nullptr);
    ggml_backend_t backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);

    int fails = 0;
    const int64_t sizes[] = {2, 64, 128, 512, 1024};
    const int64_t rows = 3; // multi-row to check row handling

    for (int64_t n : sizes) {
        std::vector<float> in(n*rows);
        for (int64_t i = 0; i < n*rows; ++i) in[i] = (float)((i*37 % 97) - 48) / 7.0f;

        std::vector<float> ref = run_on(backend_cpu, in.data(), n, rows);
        std::vector<float> out = run_on(backend,      in.data(), n, rows);

        double md = max_diff(ref.data(), out.data(), n*rows);
        double tol = 4e-4 * std::sqrt((double)n);
        bool ok = md <= tol;
        if (!ok) ++fails;
        printf("n=%5lld rows=%lld max_err=%9.6f tol=%.6f %s\n",
               (long long)n, (long long)rows, md, tol, ok ? "OK" : "FAIL");
    }

    ggml_backend_free(backend);
    ggml_backend_free(backend_cpu);
    printf(fails ? "FAILURES: %d\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
}