#pragma once
// Faster-GS+FusedShAdam: device helpers of the lazy Adam kernels, shared by lazy_adam.cu and the fused SH-rest update in
// preprocess_backward_cu (identical arithmetic and stochastic-rounding seeds in both places)
#include <cuda_bf16.h>
#include <cstdint>

namespace faster_gs::adam {

    // Faster-GS+Combo-SR: m and v can each be float32 or bfloat16 (all arithmetic in float32). With round-to-nearest,
    // bfloat16 cannot represent the per-step decay v <- 0.999 v, so a bfloat16 v is stored with unbiased stochastic
    // rounding (see Faster-GS+HalfMoments-SR); m uses round-to-nearest.
    __device__ __forceinline__ uint32_t hash32(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
        return x;
    }
    __device__ __forceinline__ void store_moment_sr(float* p, int i, float v, uint32_t) { p[i] = v; }
    __device__ __forceinline__ void store_moment_sr(__nv_bfloat16* p, int i, float v, uint32_t r) {
        p[i] = __ushort_as_bfloat16(static_cast<unsigned short>((__float_as_uint(v) + (r & 0xFFFFU)) >> 16));
    }
    inline uint32_t sr_seed(const int step, const int n_elements, const uint32_t kernel_salt) {
        return static_cast<uint32_t>(step) * 0x9E3779B1U ^ static_cast<uint32_t>(n_elements) * 0x85EBCA77U ^ kernel_salt * 0xC2B2AE3DU;
    }
    __device__ __forceinline__ float load_moment(const float* p, int i) { return p[i]; }
    __device__ __forceinline__ float load_moment(const __nv_bfloat16* p, int i) { return __bfloat162float(p[i]); }
    __device__ __forceinline__ void store_moment(float* p, int i, float v) { p[i] = v; }
    __device__ __forceinline__ void store_moment(__nv_bfloat16* p, int i, float v) { p[i] = __float2bfloat16_rn(v); }


    // zero-gradient steps beyond this window change the parameter by less than beta1^256 ~ 2e-12 of the momentum displacement
    constexpr int max_explicit_steps = 256;

    // application of k >= 1 zero-gradient Adam steps following step t0
    __device__ __forceinline__ void catch_up(
        float& param, float& moment1, float& moment2,
        const int t0, const int k,
        const float* __restrict__ f_table,
        const float* __restrict__ step_size_table,
        const float* __restrict__ bias_correction2_sqrt_rcp_table,
        const float beta1, const float beta2, const float epsilon,
        const float log2_gamma, const float log2_beta1, const float log2_beta2)
    {
        const int window = min(k, max_explicit_steps);
        // epsilon is negligible (< 1e-6 relative) if sqrt(v) stays above 1e6 * epsilon over the significant window
        const float sqrt_v_window_end = sqrtf(moment2 * exp2f(static_cast<float>(window) * log2_beta2));
        if (sqrt_v_window_end >= 1e6f * epsilon) {
            // closed form: theta -= m / sqrt(v) * sum_j f(t0 + j) gamma^j (float32 suffices: the relative error of the
            // displacement is ~1e-6, i.e., orders of magnitude below the float32 spacing of the parameter; FP64 would be
            // 64x slower on this GPU class)
            const float displacement_scale = f_table[t0] - exp2f(static_cast<float>(k) * log2_gamma) * f_table[t0 + k];
            param -= moment1 * rsqrtf(moment2) * displacement_scale;
            moment1 *= exp2f(static_cast<float>(k) * log2_beta1);
            moment2 *= exp2f(static_cast<float>(k) * log2_beta2);
        } else {
            // tiny second moment: replay the zero-gradient steps with exactly the arithmetic of adam_step_cu
            for (int j = 1; j <= window; ++j) {
                const int s = t0 + j;
                moment1 = fmaf(beta1, moment1, 0.0f);
                moment2 = fmaf(beta2, moment2, 0.0f);
                const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp_table[s] + epsilon;
                param -= step_size_table[s] * moment1 / denom;
            }
            if (k > window) {
                moment1 *= exp2f(static_cast<float>(k - window) * log2_beta1);
                moment2 *= exp2f(static_cast<float>(k - window) * log2_beta2);
            }
        }
    }


    // Faster-GS+FusedShAdam: arguments of the SH-rest Adam step that preprocess_backward_cu applies in place (exp_avg == nullptr:
    // not fused, the gradient is written out as usual)
    struct FusedShAdamArgs {
        float* param = nullptr;
        void* exp_avg = nullptr;
        void* exp_avg_sq = nullptr;
        bool m_half = false;
        bool v_half = false;
        const int* last_step = nullptr;
        const float* f_table = nullptr;
        const float* step_size_table = nullptr;
        const float* bias_correction2_sqrt_rcp_table = nullptr;
        int step = 0;
        float step_size = 0.0f, beta1 = 0.0f, beta2 = 0.0f, epsilon = 0.0f, bias_correction2_sqrt_rcp = 0.0f;
        float log2_gamma = 0.0f, log2_beta1 = 0.0f, log2_beta2 = 0.0f;
        uint32_t seed = 0;
    };

    // the regular lazy step of one element, with exactly the arithmetic and seed of lazy_adam_warp_cu (is_step)
    __device__ __forceinline__ void fused_adam_element(const FusedShAdamArgs& a, const int idx, const int gaussian, const float grad) {
        float p = a.param[idx];
        float m1 = a.m_half ? __bfloat162float(static_cast<const __nv_bfloat16*>(a.exp_avg)[idx]) : static_cast<const float*>(a.exp_avg)[idx];
        float m2 = a.v_half ? __bfloat162float(static_cast<const __nv_bfloat16*>(a.exp_avg_sq)[idx]) : static_cast<const float*>(a.exp_avg_sq)[idx];
        const int t0 = a.last_step[gaussian];
        const int k = a.step - 1 - t0;
        if (k > 0) catch_up(p, m1, m2, t0, k, a.f_table, a.step_size_table, a.bias_correction2_sqrt_rcp_table, a.beta1, a.beta2, a.epsilon, a.log2_gamma, a.log2_beta1, a.log2_beta2);
        const float grad_sq = grad * grad;
        const float moment1 = fmaf(a.beta1, m1 - grad, grad);
        const float moment2 = fmaf(a.beta2, m2 - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) * a.bias_correction2_sqrt_rcp + a.epsilon;
        a.param[idx] = p - a.step_size * moment1 / denom;
        if (a.m_half) store_moment(static_cast<__nv_bfloat16*>(a.exp_avg), idx, moment1);
        else store_moment(static_cast<float*>(a.exp_avg), idx, moment1);
        const uint32_t r = hash32(static_cast<uint32_t>(idx) ^ a.seed);
        if (a.v_half) store_moment_sr(static_cast<__nv_bfloat16*>(a.exp_avg_sq), idx, moment2, r);
        else store_moment_sr(static_cast<float*>(a.exp_avg_sq), idx, moment2, r);
    }

}
