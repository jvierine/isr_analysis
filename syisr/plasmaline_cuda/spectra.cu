// CUDA backends for diagnostic STFTs and sliding code-matched power spectra.
// Input samples are interleaved signed int32 Q,I; complex voltage is I + iQ.
// pl_* computes un-decoded Hann-windowed diagnostic spectra.
// pl_match_* multiplies each sliding gate by the selected code-phase template,
// FFTs the gate, and accumulates power. This is a matched-periodogram estimator;
// standard alternating-code lag-product decoding is not implemented here.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
static thread_local char error_text[256];
#define CUDA(x)                                                                                    \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            snprintf(error_text, 256, "%s", cudaGetErrorString(e));                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
#define FFT(x)                                                                                     \
    do {                                                                                           \
        cufftResult e = (x);                                                                       \
        if (e != CUFFT_SUCCESS) {                                                                  \
            snprintf(error_text, 256, "cuFFT error %d", int(e));                                   \
            return 2;                                                                              \
        }                                                                                          \
    } while (0)
struct Context {
    int n, batch;
    int32_t *input;
    cufftComplex *voltage;
    float *power;
    cufftHandle plan;
};
// Convert Q,I to complex voltage and apply one symmetric Hann per FFT window.
__global__ void prepare(const int32_t *in, cufftComplex *out, int n, int total) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total)
        return;
    float w = 0.5f - 0.5f * cosf(6.283185307179586f * (k % n) / (n - 1));
    out[k] = make_cuFloatComplex(float(in[2 * k + 1]) * w, float(in[2 * k]) * w);
}
// Center the FFT frequency axis and normalize by the Hann window energy.
__global__ void powers(const cufftComplex *x, float *out, int n, int total) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total)
        return;
    // For supported window lengths, the symmetric Hann energy is 3*(n-1)/8.
    auto v = x[(k / n) * n + (k % n + n / 2) % n];
    out[k] = (v.x * v.x + v.y * v.y) / (3.f * (n - 1) / 8.f);
}
extern "C" const char *pl_error() { return error_text; }
extern "C" int pl_create(int device, int n, int batch, void **result) {
    CUDA(cudaSetDevice(device));
    auto c = new Context{};
    c->n = n;
    c->batch = batch;
    CUDA(cudaMalloc(&c->input, size_t(n) * batch * 8));
    CUDA(cudaMalloc(&c->voltage, size_t(n) * batch * 8));
    CUDA(cudaMalloc(&c->power, size_t(n) * batch * 4));
    FFT(cufftPlan1d(&c->plan, n, CUFFT_C2C, batch));
    *result = c;
    return 0;
}
extern "C" int pl_power(void *ptr, const int32_t *input, float *output) {
    auto c = (Context *)ptr;
    int total = c->n * c->batch;
    CUDA(cudaMemcpy(c->input, input, size_t(total) * 8, cudaMemcpyHostToDevice));
    prepare<<<(total + 255) / 256, 256>>>(c->input, c->voltage, c->n, total);
    CUDA(cudaGetLastError());
    FFT(cufftExecC2C(c->plan, c->voltage, c->voltage, CUFFT_FORWARD));
    powers<<<(total + 255) / 256, 256>>>(c->voltage, c->power, c->n, total);
    CUDA(cudaGetLastError());
    CUDA(cudaMemcpy(output, c->power, size_t(total) * 4, cudaMemcpyDeviceToHost));
    return 0;
}
extern "C" void pl_destroy(void *ptr) {
    auto c = (Context *)ptr;
    if (!c)
        return;
    cufftDestroy(c->plan);
    cudaFree(c->input);
    cudaFree(c->voltage);
    cudaFree(c->power);
    delete c;
}

// Adapted sliding conjugate-template/echo FFT architecture from Nate Hilliard's
// PlasmaLine_CUDA (CUDA 7-era archive supplied by the user). Bounds checks,
// persistent plans, explicit normalization and voltage whitening are new.
struct Match {
    int ne, nt, nfft, nr, step, nout, nwhite;
    int32_t *qi;
    float *gain, *codes, *out;
    double *sum, *square;
    cufftComplex *echo, *batch;
    cufftHandle whiteplan, plan;
};
// Zero-pad the raw record for the record-wide voltage-whitening FFT.
__global__ void unpack(const int32_t *qi, cufftComplex *echo, int ne, int nw) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nw)
        echo[k] =
            k < ne ? make_cuFloatComplex(qi[2 * k + 1], qi[2 * k]) : make_cuFloatComplex(0, 0);
}
// Apply the supplied real whitening gain. cuFFT inverse transforms omit 1/N,
// so include that normalization here before transforming back to voltage.
__global__ void apply_gain(cufftComplex *x, const float *gain, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        x[k].x *= gain[k] / n;
        x[k].y *= gain[k] / n;
    }
}
// Each batch row is a gate starting at r*step. Real +/-1 code samples equal
// their conjugates. Pad the template-weighted gate to the FFT length.
__global__ void code_echo(const cufftComplex *echo, const float *codes, cufftComplex *batch,
                          int phase, int nt, int nf, int nr, int step) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nf * nr)
        return;
    int r = k / nf, j = k % nf;
    if (j >= nt) {
        batch[k] = make_cuFloatComplex(0, 0);
        return;
    }
    auto v = echo[r * step + j];
    float c = codes[phase * nt + j];
    batch[k] = make_cuFloatComplex(v.x * c, v.y * c);
}
// FFT-shift each spectrum, optionally average adjacent frequency bins, and
// normalize squared voltage by template length and frequency-bin count.
__global__ void reduce_power(const cufftComplex *x, float *out, int nf, int no, int nr, int nt) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nr * no)
        return;
    int r = k / no, j = k % no, factor = nf / no;
    float s = 0;
    for (int i = 0; i < factor; i++) {
        auto v = x[r * nf + (j * factor + i + nf / 2) % nf];
        s += v.x * v.x + v.y * v.y;
    }
    out[k] = s / (nt * factor);
}
// Persistent double-precision first and second raw moments across IPPs.
// The host tracks accepted IPP counts and forms means and scatter errors.
__global__ void accumulate(const float *power, double *sum, double *square, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        double p = power[k];
        sum[k] += p;
        square[k] += p * p;
    }
}
extern "C" int pl_match_create(int device, int ne, int nt, int nf, int step, int no,
                               const float *codes, const float *gain, void **result) {
    CUDA(cudaSetDevice(device));
    auto c = new Match{};
    c->ne = ne;
    c->nt = nt;
    c->nfft = nf;
    c->step = step;
    c->nout = no;
    c->nr = (ne - nt) / step + 1;
    c->nwhite = 1;
    while (c->nwhite < ne)
        c->nwhite *= 2;
    if (nt > nf || nf % no || c->nr < 1) {
        snprintf(error_text, 256, "invalid match dimensions");
        return 3;
    }
    CUDA(cudaMalloc(&c->qi, size_t(ne) * 8));
    CUDA(cudaMalloc(&c->echo, size_t(c->nwhite) * 8));
    CUDA(cudaMalloc(&c->batch, size_t(nf) * c->nr * 8));
    CUDA(cudaMalloc(&c->out, size_t(no) * c->nr * 4));
    CUDA(cudaMalloc(&c->gain, size_t(c->nwhite) * 4));
    CUDA(cudaMalloc(&c->codes, size_t(nt) * 64 * 4));
    CUDA(cudaMalloc(&c->sum, size_t(no) * c->nr * 8));
    CUDA(cudaMalloc(&c->square, size_t(no) * c->nr * 8));
    CUDA(cudaMemset(c->sum, 0, size_t(no) * c->nr * 8));
    CUDA(cudaMemset(c->square, 0, size_t(no) * c->nr * 8));
    CUDA(cudaMemcpy(c->gain, gain, size_t(c->nwhite) * 4, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(c->codes, codes, size_t(nt) * 64 * 4, cudaMemcpyHostToDevice));
    FFT(cufftPlan1d(&c->whiteplan, c->nwhite, CUFFT_C2C, 1));
    FFT(cufftPlan1d(&c->plan, nf, CUFFT_C2C, c->nr));
    *result = c;
    return 0;
}
// Replay one IPP through whitening, sliding template multiplication, FFT,
// and power formation. Whole-record whitening has nonlocal time support.
static int execute_match(void *ptr, const int32_t *qi, int phase) {
    auto c = (Match *)ptr;
    if (phase < 0 || phase >= 64) {
        snprintf(error_text, 256, "invalid code phase");
        return 3;
    }
    CUDA(cudaMemcpy(c->qi, qi, size_t(c->ne) * 8, cudaMemcpyHostToDevice));
    unpack<<<(c->nwhite + 255) / 256, 256>>>(c->qi, c->echo, c->ne, c->nwhite);
    CUDA(cudaGetLastError());
    FFT(cufftExecC2C(c->whiteplan, c->echo, c->echo, CUFFT_FORWARD));
    apply_gain<<<(c->nwhite + 255) / 256, 256>>>(c->echo, c->gain, c->nwhite);
    CUDA(cudaGetLastError());
    FFT(cufftExecC2C(c->whiteplan, c->echo, c->echo, CUFFT_INVERSE));
    int total = c->nfft * c->nr;
    code_echo<<<(total + 255) / 256, 256>>>(c->echo, c->codes, c->batch, phase, c->nt, c->nfft,
                                            c->nr, c->step);
    CUDA(cudaGetLastError());
    FFT(cufftExecC2C(c->plan, c->batch, c->batch, CUFFT_FORWARD));
    total = c->nout * c->nr;
    reduce_power<<<(total + 255) / 256, 256>>>(c->batch, c->out, c->nfft, c->nout, c->nr, c->nt);
    CUDA(cudaGetLastError());
    return 0;
}
extern "C" int pl_match_power(void *ptr, const int32_t *qi, int phase, float *out) {
    int e = execute_match(ptr, qi, phase);
    if (e)
        return e;
    auto c = (Match *)ptr;
    CUDA(cudaMemcpy(out, c->out, size_t(c->nout) * c->nr * 4, cudaMemcpyDeviceToHost));
    return 0;
}
extern "C" int pl_match_accumulate(void *ptr, const int32_t *qi, int phase) {
    int e = execute_match(ptr, qi, phase);
    if (e)
        return e;
    auto c = (Match *)ptr;
    int n = c->nout * c->nr;
    accumulate<<<(n + 255) / 256, 256>>>(c->out, c->sum, c->square, n);
    CUDA(cudaGetLastError());
    return 0;
}
extern "C" int pl_match_results(void *ptr, double *sum, double *square) {
    auto c = (Match *)ptr;
    CUDA(cudaMemcpy(sum, c->sum, size_t(c->nout) * c->nr * 8, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(square, c->square, size_t(c->nout) * c->nr * 8, cudaMemcpyDeviceToHost));
    return 0;
}
extern "C" void pl_match_destroy(void *ptr) {
    auto c = (Match *)ptr;
    if (!c)
        return;
    cufftDestroy(c->whiteplan);
    cufftDestroy(c->plan);
    cudaFree(c->qi);
    cudaFree(c->echo);
    cudaFree(c->batch);
    cudaFree(c->out);
    cudaFree(c->gain);
    cudaFree(c->codes);
    cudaFree(c->sum);
    cudaFree(c->square);
    delete c;
}
