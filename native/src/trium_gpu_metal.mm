/**
 * @file trium_gpu_metal.mm
 * @brief Native Apple Metal GPU Backend for Trium-XBT & Biology MM
 *
 * Implements Zero-Copy Unified Memory (MTLResourceStorageModeShared) compute
 * pipelines compiled at runtime via Metal Shading Language:
 *   1. xbt_gpu_probe_device
 *   2. xbt_gpu_verlet_plan_create
 *   3. xbt_gpu_verlet_plan_eval
 *   4. xbt_gpu_verlet_plan_destroy
 *   5. xbt_gpu_verlet_plan_num_pairs
 *   6. xbt_gpu_ternary_matvec_dualrail
 *   7. xbt_gpu_msm_stencil27
 *   8. xbt_gpu_md_batch_run
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#define XBT_GPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr double SQRT_PI = 1.77245385090551602729816748334114518;
constexpr int DSF_N_TAB = 512;

struct alignas(4) VerletUniforms {
    uint32_t n_atoms;
    uint32_t do_lj;
    uint32_t do_coul;
    uint32_t use_lfs;
    uint32_t has_scaled;
    uint32_t has_box;
    uint32_t is_triclinic;
    uint32_t compute_virial;
    float rcut;
    float rc2;
    float inv_rc;
    float inv_rc2;
    float coulomb_prefactor;
    float Arc;
    float Brc;
    float du;
    float inv_du;
    float L0;
    float L1;
    float L2;
    float inv_L0;
    float inv_L1;
    float inv_L2;
    float _pad0;
    float H[9];
    float Hinv[9];
};

struct alignas(4) MatvecUniforms {
    uint32_t M;
    uint32_t K_words;
    uint32_t has_WP;
    uint32_t has_WN;
    uint32_t has_xP;
    uint32_t has_xN;
};

struct alignas(4) ReduceUniforms {
    uint32_t n_atoms;
    uint32_t compute_virial;
    uint32_t has_bonded;
    uint32_t compute_ke;
};

struct alignas(4) IntegratorUniforms {
    uint32_t n_atoms;
    float kick_dt;
    float drift_dt;
    uint32_t _pad0;
};

struct Float4 {
    float x, y, z, w;
};

struct Float2 {
    float x, y;
};

static_assert(sizeof(Float4) == 16, "Float4 must be 16 bytes");
static_assert(sizeof(Float2) == 8, "Float2 must be 8 bytes");
static_assert(sizeof(VerletUniforms) == 42 * 4, "VerletUniforms layout mismatch");
static_assert(sizeof(ReduceUniforms) == 16, "ReduceUniforms layout mismatch");
static_assert(sizeof(IntegratorUniforms) == 16, "IntegratorUniforms layout mismatch");

static const char* METAL_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

struct VerletUniforms {
    uint n_atoms;
    uint do_lj;
    uint do_coul;
    uint use_lfs;
    uint has_scaled;
    uint has_box;
    uint is_triclinic;
    uint compute_virial;
    float rcut;
    float rc2;
    float inv_rc;
    float inv_rc2;
    float coulomb_prefactor;
    float Arc;
    float Brc;
    float du;
    float inv_du;
    float L0;
    float L1;
    float L2;
    float inv_L0;
    float inv_L1;
    float inv_L2;
    float _pad0;
    float H[9];
    float Hinv[9];
};

struct MatvecUniforms {
    uint M;
    uint K_words;
    uint has_WP;
    uint has_WN;
    uint has_xP;
    uint has_xN;
};

struct ReduceUniforms {
    uint n_atoms;
    uint compute_virial;
    uint has_bonded;
    uint compute_ke;
};

struct IntegratorUniforms {
    uint n_atoms;
    float kick_dt;
    float drift_dt;
    uint _pad0;
};

inline float3 mic_disp(float dx0, float dy0, float dz0, constant VerletUniforms& u) {
    float dx = dx0;
    float dy = dy0;
    float dz = dz0;
    if (u.has_box != 0u) {
        if (u.is_triclinic == 0u) {
            dx = dx0 - u.L0 * rint(dx0 * u.inv_L0);
            dy = dy0 - u.L1 * rint(dy0 * u.inv_L1);
            dz = dz0 - u.L2 * rint(dz0 * u.inv_L2);
        } else {
            const float f0 = u.Hinv[0] * dx0 + u.Hinv[3] * dy0 + u.Hinv[6] * dz0;
            const float f1 = u.Hinv[1] * dx0 + u.Hinv[4] * dy0 + u.Hinv[7] * dz0;
            const float f2 = u.Hinv[2] * dx0 + u.Hinv[5] * dy0 + u.Hinv[8] * dz0;

            const float n0 = rint(f0);
            const float n1 = rint(f1);
            const float n2 = rint(f2);

            float bx = dx0 - (u.H[0] * n0 + u.H[3] * n1 + u.H[6] * n2);
            float by = dy0 - (u.H[1] * n0 + u.H[4] * n1 + u.H[7] * n2);
            float bz = dz0 - (u.H[2] * n0 + u.H[5] * n1 + u.H[8] * n2);
            float b2 = bx * bx + by * by + bz * bz;

            for (int da = -1; da <= 1; ++da) {
                for (int db = -1; db <= 1; ++db) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (da == 0 && db == 0 && dc == 0) continue;
                        const float na = n0 + (float)da;
                        const float nb = n1 + (float)db;
                        const float nc = n2 + (float)dc;
                        const float rx = dx0 - (u.H[0] * na + u.H[3] * nb + u.H[6] * nc);
                        const float ry = dy0 - (u.H[1] * na + u.H[4] * nb + u.H[7] * nc);
                        const float rz = dz0 - (u.H[2] * na + u.H[5] * nb + u.H[8] * nc);
                        const float r2c = rx * rx + ry * ry + rz * rz;
                        if (r2c < b2) {
                            b2 = r2c;
                            bx = rx;
                            by = ry;
                            bz = rz;
                        }
                    }
                }
            }
            dx = bx;
            dy = by;
            dz = bz;
        }
    }
    return float3(dx, dy, dz);
}

constant bool FC_COMPUTE_VIRIAL [[function_constant(0)]];

kernel void lj_dsf_csr_kernel(
    device const float4*   q_f4        [[buffer(0)]],
    device const float2*   sig_eps     [[buffer(1)]],
    device const uint*     sym_row_ptr [[buffer(2)]],
    device const int*      sym_col_idx [[buffer(3)]],
    device const float*    sym_wlj     [[buffer(4)]],
    device const float*    sym_wcoul   [[buffer(5)]],
    device const float4*   dsf_table   [[buffer(6)]],
    device float*          f_out       [[buffer(7)]],
    device float*          u_out       [[buffer(8)]],
    device float*          vir_out     [[buffer(9)]],
    constant VerletUniforms& u         [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const uint i = gid >> 3u;
    const uint sub_lane = gid & 7u;
    const bool active = (i < u.n_atoms);

    uint k_begin = 0u, k_end = 0u;
    float4 qi = float4(0.0f);
    float2 sei = float2(0.0f);
    bool row_has_lj = false;
    bool row_has_coul = false;
    if (active) {
        k_begin = sym_row_ptr[i];
        k_end   = sym_row_ptr[i + 1u];
        qi  = q_f4[i];
        sei = sig_eps[i];
        row_has_lj   = (u.do_lj != 0u) && (sei.y > 0.0f);
        row_has_coul = (u.do_coul != 0u) && (qi.w != 0.0f);
    }

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float ui = 0.0f;
    float v00 = 0.0f, v10 = 0.0f, v20 = 0.0f;
    float v01 = 0.0f, v11 = 0.0f, v21 = 0.0f;
    float v02 = 0.0f, v12 = 0.0f, v22 = 0.0f;

    if (active && k_begin < k_end && (row_has_lj || row_has_coul)) {
        const float k_chg_i = u.coulomb_prefactor * qi.w;

        for (uint k = k_begin + sub_lane; k < k_end; k += 8u) {
            const uint j = (uint)sym_col_idx[k];
            const float4 qj = q_f4[j];

            const float3 disp = mic_disp(qi.x - qj.x, qi.y - qj.y, qi.z - qj.z, u);
            const float dx = disp.x;
            const float dy = disp.y;
            const float dz = disp.z;

            const float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= u.rc2 || r2 < 1e-24f) continue;

            const float w_lj   = (u.has_scaled != 0u) ? sym_wlj[k]   : 1.0f;
            const float w_coul = (u.has_scaled != 0u) ? sym_wcoul[k] : 1.0f;

            const float inv_r = rsqrt(r2);
            const float r = r2 * inv_r;
            const float inv_r2 = inv_r * inv_r;

            float u_pair = 0.0f;
            float c_total = 0.0f;

            if (row_has_lj && w_lj != 0.0f) {
                const float2 sej = sig_eps[j];
                const float eij = sei.y * sej.y;
                if (eij > 0.0f) {
                    const float sij = sei.x + sej.x;
                    if (sij > 0.0f) {
                        const float sij2 = sij * sij;
                        const float s2 = sij2 * inv_r2;
                        const float s6 = s2 * s2 * s2;
                        const float s12 = s6 * s6;

                        const float s2c = sij2 * u.inv_rc2;
                        const float s6c = s2c * s2c * s2c;
                        const float s12c = s6c * s6c;

                        float u_lj = 4.0f * eij * ((s12 - s6) - (s12c - s6c));
                        float c_lj = 24.0f * eij * (2.0f * s12 - s6) * inv_r2;

                        if (u.use_lfs != 0u) {
                            const float duc = 24.0f * eij * (s6c - 2.0f * s12c) * u.inv_rc;
                            u_lj -= duc * (r - u.rcut);
                            c_lj += duc * inv_r;
                        }

                        u_pair += w_lj * u_lj;
                        c_total += w_lj * c_lj;
                    }
                }
            }

            if (row_has_coul && w_coul != 0.0f && qj.w != 0.0f) {
                const float kq = w_coul * k_chg_i * qj.w;
                const int tidx = clamp((int)(r2 * u.inv_du), 0, 511);
                const float dt = r2 - ((float)tidx + 0.5f) * u.du;
                const float4 c = dsf_table[tidx];
                const float phi0  = fma(dt, fma(dt, fma(dt, c.w, c.z), c.y), c.x);
                const float dphi0 = fma(dt, fma(dt, 3.0f * c.w, 2.0f * c.z), c.y);
                const float erfc_over_r = inv_r - phi0;
                const float inv_r3 = inv_r2 * inv_r;
                u_pair  += kq * ((erfc_over_r - u.Brc) + u.Arc * r);
                c_total += kq * (inv_r3 + 2.0f * dphi0 - u.Arc * inv_r);
            }

            ui += u_pair;
            const float fx = c_total * dx;
            const float fy = c_total * dy;
            const float fz = c_total * dz;
            fxi += fx;
            fyi += fy;
            fzi += fz;

            if (FC_COMPUTE_VIRIAL) {
                float wx = 0.0f, wy = 0.0f, wz = 0.0f;
                if (u.is_triclinic == 0u) {
                    wx = dx * u.inv_L0;
                    wy = dy * u.inv_L1;
                    wz = dz * u.inv_L2;
                } else {
                    wx = u.Hinv[0] * dx + u.Hinv[3] * dy + u.Hinv[6] * dz;
                    wy = u.Hinv[1] * dx + u.Hinv[4] * dy + u.Hinv[7] * dz;
                    wz = u.Hinv[2] * dx + u.Hinv[5] * dy + u.Hinv[8] * dz;
                }
                const float gx = -fx;
                const float gy = -fy;
                const float gz = -fz;
                v00 += gx * wx; v10 += gy * wx; v20 += gz * wx;
                v01 += gx * wy; v11 += gy * wy; v21 += gz * wy;
                v02 += gx * wz; v12 += gy * wz; v22 += gz * wz;
            }
        }
    }

    fxi += simd_shuffle_xor(fxi, 4u);
    fxi += simd_shuffle_xor(fxi, 2u);
    fxi += simd_shuffle_xor(fxi, 1u);

    fyi += simd_shuffle_xor(fyi, 4u);
    fyi += simd_shuffle_xor(fyi, 2u);
    fyi += simd_shuffle_xor(fyi, 1u);

    fzi += simd_shuffle_xor(fzi, 4u);
    fzi += simd_shuffle_xor(fzi, 2u);
    fzi += simd_shuffle_xor(fzi, 1u);

    ui += simd_shuffle_xor(ui, 4u);
    ui += simd_shuffle_xor(ui, 2u);
    ui += simd_shuffle_xor(ui, 1u);

    if (FC_COMPUTE_VIRIAL) {
        v00 += simd_shuffle_xor(v00, 4u); v00 += simd_shuffle_xor(v00, 2u); v00 += simd_shuffle_xor(v00, 1u);
        v10 += simd_shuffle_xor(v10, 4u); v10 += simd_shuffle_xor(v10, 2u); v10 += simd_shuffle_xor(v10, 1u);
        v20 += simd_shuffle_xor(v20, 4u); v20 += simd_shuffle_xor(v20, 2u); v20 += simd_shuffle_xor(v20, 1u);
        v01 += simd_shuffle_xor(v01, 4u); v01 += simd_shuffle_xor(v01, 2u); v01 += simd_shuffle_xor(v01, 1u);
        v11 += simd_shuffle_xor(v11, 4u); v11 += simd_shuffle_xor(v11, 2u); v11 += simd_shuffle_xor(v11, 1u);
        v21 += simd_shuffle_xor(v21, 4u); v21 += simd_shuffle_xor(v21, 2u); v21 += simd_shuffle_xor(v21, 1u);
        v02 += simd_shuffle_xor(v02, 4u); v02 += simd_shuffle_xor(v02, 2u); v02 += simd_shuffle_xor(v02, 1u);
        v12 += simd_shuffle_xor(v12, 4u); v12 += simd_shuffle_xor(v12, 2u); v12 += simd_shuffle_xor(v12, 1u);
        v22 += simd_shuffle_xor(v22, 4u); v22 += simd_shuffle_xor(v22, 2u); v22 += simd_shuffle_xor(v22, 1u);
    }

    if (active && sub_lane == 0u) {
        const uint fbase = 3u * i;
        f_out[fbase + 0u] = fxi;
        f_out[fbase + 1u] = fyi;
        f_out[fbase + 2u] = fzi;
        u_out[i] = ui;

        if (FC_COMPUTE_VIRIAL) {
            const uint vb = 9u * i;
            vir_out[vb + 0u] = v00; vir_out[vb + 1u] = v10; vir_out[vb + 2u] = v20;
            vir_out[vb + 3u] = v01; vir_out[vb + 4u] = v11; vir_out[vb + 5u] = v21;
            vir_out[vb + 6u] = v02; vir_out[vb + 7u] = v12; vir_out[vb + 8u] = v22;
        }
    }
}

kernel void ternary_matvec_dualrail_kernel(
    device const uint2*      W_P     [[buffer(0)]],
    device const uint2*      W_N     [[buffer(1)]],
    device const uint2*      x_P     [[buffer(2)]],
    device const uint2*      x_N     [[buffer(3)]],
    device long*             acc_out [[buffer(4)]],
    constant MatvecUniforms& u       [[buffer(5)]],
    uint m [[thread_position_in_grid]])
{
    if (m >= u.M) return;

    long acc = 0;
    const ulong stride = (ulong)u.M;
    ulong w_idx = (ulong)m;

    for (uint k = 0u; k < u.K_words; ++k) {
        const uint2 wp = (u.has_WP != 0u) ? W_P[w_idx] : uint2(0u, 0u);
        const uint2 wn = (u.has_WN != 0u) ? W_N[w_idx] : uint2(0u, 0u);
        const uint2 xp = (u.has_xP != 0u) ? x_P[k]     : uint2(0u, 0u);
        const uint2 xn = (u.has_xN != 0u) ? x_N[k]     : uint2(0u, 0u);

        const uint2 pos = (wp & xp) | (wn & xn);
        const uint2 neg = (wp & xn) | (wn & xp);
        const uint2 pc = popcount(pos);
        const uint2 nc = popcount(neg);
        acc += (long)(pc.x + pc.y) - (long)(nc.x + nc.y);

        w_idx += stride;
    }

    acc_out[m] = acc;
}

kernel void msm_stencil27_kernel(
    device const float* Q_grid   [[buffer(0)]],
    device const float* k27      [[buffer(1)]],
    device float*       Phi_grid [[buffer(2)]],
    constant uint&      Ml       [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= Ml || gid.y >= Ml || gid.z >= Ml) return;

    const int iMl = (int)Ml;
    const int x = (int)gid.x;
    const int y = (int)gid.y;
    const int z = (int)gid.z;
    const uint Ml2 = Ml * Ml;

    float acc = 0.0f;
    uint t = 0u;
    for (int dz = -1; dz <= 1; ++dz) {
        int mz = z + dz;
        if (mz < 0) mz += iMl; else if (mz >= iMl) mz -= iMl;
        const uint z_off = (uint)mz * Ml2;
        for (int dy = -1; dy <= 1; ++dy) {
            int my = y + dy;
            if (my < 0) my += iMl; else if (my >= iMl) my -= iMl;
            const uint yz_off = z_off + (uint)my * Ml;
            for (int dx = -1; dx <= 1; ++dx) {
                int mx = x + dx;
                if (mx < 0) mx += iMl; else if (mx >= iMl) mx -= iMl;
                acc = fma(k27[t++], Q_grid[yz_off + (uint)mx], acc);
            }
        }
    }

    Phi_grid[gid.x + Ml * gid.y + Ml2 * gid.z] = acc;
}

kernel void reduce_energy_virial_kernel(
    device const float*    u_nb        [[buffer(0)]],
    device const float*    vir_out     [[buffer(1)]],
    device const float*    u_bond      [[buffer(2)]],
    device const float4*   p_f4        [[buffer(3)]],
    device float*          partial_out [[buffer(4)]],
    constant ReduceUniforms& ru        [[buffer(5)]],
    uint gid         [[thread_position_in_grid]],
    uint tg_id       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_id     [[simdgroup_index_in_threadgroup]],
    uint num_simds   [[simdgroups_per_threadgroup]],
    uint simd_size   [[threads_per_simdgroup]],
    uint total_threads [[threads_per_grid]])
{
    float su = 0.0f;
    float sb = 0.0f;
    float ske = 0.0f;
    float sv0 = 0.0f, sv1 = 0.0f, sv2 = 0.0f;
    float sv3 = 0.0f, sv4 = 0.0f, sv5 = 0.0f;
    float sv6 = 0.0f, sv7 = 0.0f, sv8 = 0.0f;

    for (uint i = gid; i < ru.n_atoms; i += total_threads) {
        su += u_nb[i];
        if (ru.has_bonded != 0u) {
            sb += u_bond[i];
        }
        if (ru.compute_ke != 0u) {
            const float4 pv = p_f4[i];
            ske += 0.5f * (pv.x * pv.x + pv.y * pv.y + pv.z * pv.z) * pv.w;
        }
        if (ru.compute_virial != 0u) {
            const uint vb = 9u * i;
            sv0 += vir_out[vb + 0u]; sv1 += vir_out[vb + 1u]; sv2 += vir_out[vb + 2u];
            sv3 += vir_out[vb + 3u]; sv4 += vir_out[vb + 4u]; sv5 += vir_out[vb + 5u];
            sv6 += vir_out[vb + 6u]; sv7 += vir_out[vb + 7u]; sv8 += vir_out[vb + 8u];
        }
    }

    su = simd_sum(su);
    if (ru.has_bonded != 0u) sb = simd_sum(sb);
    if (ru.compute_ke != 0u) ske = simd_sum(ske);
    if (ru.compute_virial != 0u) {
        sv0 = simd_sum(sv0); sv1 = simd_sum(sv1); sv2 = simd_sum(sv2);
        sv3 = simd_sum(sv3); sv4 = simd_sum(sv4); sv5 = simd_sum(sv5);
        sv6 = simd_sum(sv6); sv7 = simd_sum(sv7); sv8 = simd_sum(sv8);
    }

    threadgroup float tg_sums[32 * 12];
    if (simd_lane == 0u) {
        const uint base = simd_id * 12u;
        tg_sums[base + 0u] = su;
        tg_sums[base + 1u] = sb;
        tg_sums[base + 2u] = ske;
        if (ru.compute_virial != 0u) {
            tg_sums[base + 3u]  = sv0; tg_sums[base + 4u]  = sv1; tg_sums[base + 5u]  = sv2;
            tg_sums[base + 6u]  = sv3; tg_sums[base + 7u]  = sv4; tg_sums[base + 8u]  = sv5;
            tg_sums[base + 9u]  = sv6; tg_sums[base + 10u] = sv7; tg_sums[base + 11u] = sv8;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u) {
        float fu = 0.0f, fb = 0.0f, fke = 0.0f;
        float fv0 = 0.0f, fv1 = 0.0f, fv2 = 0.0f;
        float fv3 = 0.0f, fv4 = 0.0f, fv5 = 0.0f;
        float fv6 = 0.0f, fv7 = 0.0f, fv8 = 0.0f;

        for (uint s = simd_lane; s < num_simds; s += simd_size) {
            const uint base = s * 12u;
            fu += tg_sums[base + 0u];
            if (ru.has_bonded != 0u) fb += tg_sums[base + 1u];
            if (ru.compute_ke != 0u) fke += tg_sums[base + 2u];
            if (ru.compute_virial != 0u) {
                fv0 += tg_sums[base + 3u];  fv1 += tg_sums[base + 4u];  fv2 += tg_sums[base + 5u];
                fv3 += tg_sums[base + 6u];  fv4 += tg_sums[base + 7u];  fv5 += tg_sums[base + 8u];
                fv6 += tg_sums[base + 9u];  fv7 += tg_sums[base + 10u]; fv8 += tg_sums[base + 11u];
            }
        }

        fu = simd_sum(fu);
        if (ru.has_bonded != 0u) fb = simd_sum(fb);
        if (ru.compute_ke != 0u) fke = simd_sum(fke);
        if (ru.compute_virial != 0u) {
            fv0 = simd_sum(fv0); fv1 = simd_sum(fv1); fv2 = simd_sum(fv2);
            fv3 = simd_sum(fv3); fv4 = simd_sum(fv4); fv5 = simd_sum(fv5);
            fv6 = simd_sum(fv6); fv7 = simd_sum(fv7); fv8 = simd_sum(fv8);
        }

        if (simd_lane == 0u) {
            const uint out_base = tg_id * 12u;
            partial_out[out_base + 0u] = fu;
            partial_out[out_base + 1u] = fb;
            partial_out[out_base + 2u] = fke;
            if (ru.compute_virial != 0u) {
                partial_out[out_base + 3u]  = fv0;
                partial_out[out_base + 4u]  = fv1;
                partial_out[out_base + 5u]  = fv2;
                partial_out[out_base + 6u]  = fv3;
                partial_out[out_base + 7u]  = fv4;
                partial_out[out_base + 8u]  = fv5;
                partial_out[out_base + 9u]  = fv6;
                partial_out[out_base + 10u] = fv7;
                partial_out[out_base + 11u] = fv8;
            }
        }
    }
}

kernel void bonded_csr_kernel(
    device const float4*   q_f4          [[buffer(0)]],
    device const uint*     atom_bond_ptr [[buffer(1)]],
    device const uint*     atom_bond_idx [[buffer(2)]],
    device const uint2*    bond_atoms    [[buffer(3)]],
    device const float2*   bond_kr0      [[buffer(4)]],
    device const uint*     atom_ang_ptr  [[buffer(5)]],
    device const uint*     atom_ang_idx  [[buffer(6)]],
    device const uint4*    ang_atoms     [[buffer(7)]],
    device const float2*   ang_kt0       [[buffer(8)]],
    device float*          f_out         [[buffer(9)]],
    device float*          u_bond_out    [[buffer(10)]],
    constant VerletUniforms& u           [[buffer(11)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= u.n_atoms) return;

    const uint b_begin = atom_bond_ptr[r];
    const uint b_end   = atom_bond_ptr[r + 1u];
    const uint a_begin = atom_ang_ptr[r];
    const uint a_end   = atom_ang_ptr[r + 1u];

    if (b_begin == b_end && a_begin == a_end) {
        u_bond_out[r] = 0.0f;
        return;
    }

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float ui = 0.0f;

    for (uint kb = b_begin; kb < b_end; ++kb) {
        const uint entry = atom_bond_idx[kb];
        const uint b = entry >> 1u;
        const uint role = entry & 1u;
        const uint2 ba = bond_atoms[b];
        const float3 qi = q_f4[ba.x].xyz;
        const float3 qj = q_f4[ba.y].xyz;
        const float3 d = mic_disp(qi.x - qj.x, qi.y - qj.y, qi.z - qj.z, u);
        const float r2 = dot(d, d);
        if (r2 >= 1e-24f) {
            const float dist = sqrt(r2);
            const float2 kr0 = bond_kr0[b];
            const float dr = dist - kr0.y;
            const float c = -kr0.x * dr / dist;
            const float sgn = (role == 0u) ? 1.0f : -1.0f;
            const float sc = sgn * c;
            fxi = fma(sc, d.x, fxi);
            fyi = fma(sc, d.y, fyi);
            fzi = fma(sc, d.z, fzi);
            ui += 0.25f * kr0.x * dr * dr;
        }
    }

    for (uint ka = a_begin; ka < a_end; ++ka) {
        const uint entry = atom_ang_idx[ka];
        const uint a = entry >> 2u;
        const uint role = entry & 3u;
        const uint4 aa = ang_atoms[a];
        const float3 qi = q_f4[aa.x].xyz;
        const float3 qj = q_f4[aa.y].xyz;
        const float3 qk = q_f4[aa.z].xyz;
        const float3 u1 = mic_disp(qi.x - qj.x, qi.y - qj.y, qi.z - qj.z, u);
        const float3 u2 = mic_disp(qk.x - qj.x, qk.y - qj.y, qk.z - qj.z, u);
        const float r1_sq = dot(u1, u1);
        const float r2_sq = dot(u2, u2);
        if (r1_sq >= 1e-24f && r2_sq >= 1e-24f) {
            const float r1 = sqrt(r1_sq);
            const float r2 = sqrt(r2_sq);
            const float inv_r1 = 1.0f / r1;
            const float inv_r2 = 1.0f / r2;
            const float ct = clamp(dot(u1, u2) * (inv_r1 * inv_r2), -1.0f, 1.0f);
            const float th = acos(ct);
            const float st = sin(th);
            if (abs(st) >= 1e-7f) {
                const float2 kt0 = ang_kt0[a];
                const float dth = th - kt0.y;
                const float dU = kt0.x * dth;
                const float fac = dU / (r1 * r2 * st);
                if (role == 0u) {
                    const float3 e1 = u2 - (ct * r2 * inv_r1) * u1;
                    fxi = fma(fac, e1.x, fxi);
                    fyi = fma(fac, e1.y, fyi);
                    fzi = fma(fac, e1.z, fzi);
                } else if (role == 2u) {
                    const float3 e2 = u1 - (ct * r1 * inv_r2) * u2;
                    fxi = fma(fac, e2.x, fxi);
                    fyi = fma(fac, e2.y, fyi);
                    fzi = fma(fac, e2.z, fzi);
                } else {
                    const float3 e1 = u2 - (ct * r2 * inv_r1) * u1;
                    const float3 e2 = u1 - (ct * r1 * inv_r2) * u2;
                    fxi = fma(-fac, e1.x + e2.x, fxi);
                    fyi = fma(-fac, e1.y + e2.y, fyi);
                    fzi = fma(-fac, e1.z + e2.z, fzi);
                    ui += 0.5f * kt0.x * dth * dth;
                }
            }
        }
    }

    const uint fbase = 3u * r;
    f_out[fbase + 0u] += fxi;
    f_out[fbase + 1u] += fyi;
    f_out[fbase + 2u] += fzi;
    u_bond_out[r] = ui;
}

kernel void verlet_kick_drift_kernel(
    device float4*               q_f4 [[buffer(0)]],
    device float4*               p_f4 [[buffer(1)]],
    device const float*          f_in [[buffer(2)]],
    constant IntegratorUniforms& iu   [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= iu.n_atoms) return;
    const uint fbase = 3u * i;
    float4 pv = p_f4[i];
    pv.x = fma(iu.kick_dt, f_in[fbase + 0u], pv.x);
    pv.y = fma(iu.kick_dt, f_in[fbase + 1u], pv.y);
    pv.z = fma(iu.kick_dt, f_in[fbase + 2u], pv.z);
    p_f4[i] = pv;

    if (iu.drift_dt != 0.0f) {
        const float dt_invm = iu.drift_dt * pv.w;
        float4 qv = q_f4[i];
        qv.x = fma(dt_invm, pv.x, qv.x);
        qv.y = fma(dt_invm, pv.y, qv.y);
        qv.z = fma(dt_invm, pv.z, qv.z);
        q_f4[i] = qv;
    }
}
)";

struct MetalContext {
    std::mutex init_mtx;
    bool initialized = false;
    bool available = false;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pso_verlet = nil;
    id<MTLComputePipelineState> pso_verlet_novir = nil;
    id<MTLComputePipelineState> pso_matvec = nil;
    id<MTLComputePipelineState> pso_stencil27 = nil;
    id<MTLComputePipelineState> pso_reduce = nil;
    id<MTLComputePipelineState> pso_bonded = nil;
    id<MTLComputePipelineState> pso_verlet_kd = nil;
    id<MTLBuffer> dummy_buf = nil;

    // Reusable shared scratch buffers for matvec & stencil27
    std::mutex scratch_mtx;
    id<MTLBuffer> mv_WP = nil, mv_WN = nil, mv_xP = nil, mv_xN = nil, mv_acc = nil;
    size_t mv_W_cap = 0, mv_x_cap = 0, mv_acc_cap = 0;

    id<MTLBuffer> st_Q = nil, st_k27 = nil, st_Phi = nil;
    size_t st_grid_cap = 0;

    bool ensure_init() {
        std::lock_guard<std::mutex> lk(init_mtx);
        if (initialized) return available;
        initialized = true;

        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                available = false;
                return false;
            }
            queue = [device newCommandQueue];
            if (!queue) {
                available = false;
                return false;
            }

            NSError* error = nil;
            NSString* src = [NSString stringWithUTF8String:METAL_SHADER_SOURCE];
            library = [device newLibraryWithSource:src options:nil error:&error];
            if (!library) {
                NSLog(@"trium_gpu_metal: Metal shader compile failed: %@", error);
                available = false;
                return false;
            }

            auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [library newFunctionWithName:name];
                if (!fn) return nil;
                NSError* pso_err = nil;
                id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&pso_err];
                [fn release];
                if (!pso) {
                    NSLog(@"trium_gpu_metal: Pipeline creation failed for %@: %@", name, pso_err);
                }
                return pso;
            };

            auto make_pso_verlet = [&](bool with_virial) -> id<MTLComputePipelineState> {
                MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
                [fcv setConstantValue:&with_virial type:MTLDataTypeBool atIndex:0];
                NSError* fn_err = nil;
                id<MTLFunction> fn = [library newFunctionWithName:@"lj_dsf_csr_kernel" constantValues:fcv error:&fn_err];
                [fcv release];
                if (!fn) {
                    NSLog(@"trium_gpu_metal: Function creation failed for lj_dsf_csr_kernel: %@", fn_err);
                    return nil;
                }
                NSError* pso_err = nil;
                id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&pso_err];
                [fn release];
                if (!pso) {
                    NSLog(@"trium_gpu_metal: Pipeline creation failed for lj_dsf_csr_kernel: %@", pso_err);
                }
                return pso;
            };

            pso_verlet = make_pso_verlet(true);
            pso_verlet_novir = make_pso_verlet(false);
            pso_matvec = make_pso(@"ternary_matvec_dualrail_kernel");
            pso_stencil27 = make_pso(@"msm_stencil27_kernel");
            pso_reduce = make_pso(@"reduce_energy_virial_kernel");
            pso_bonded = make_pso(@"bonded_csr_kernel");
            pso_verlet_kd = make_pso(@"verlet_kick_drift_kernel");
            dummy_buf = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
            if (dummy_buf) {
                std::memset([dummy_buf contents], 0, 64);
            }

            available = (pso_verlet != nil && pso_verlet_novir != nil &&
                         pso_matvec != nil && pso_stencil27 != nil &&
                         pso_reduce != nil && pso_bonded != nil && pso_verlet_kd != nil &&
                         dummy_buf != nil);
            return available;
        }
    }

    static id<MTLBuffer> ensure_shared_buf(id<MTLDevice> dev, id<MTLBuffer>& buf, size_t& cap, size_t needed_bytes) {
        const size_t aligned = std::max<size_t>(64, needed_bytes);
        if (buf == nil || cap < aligned) {
            if (buf != nil) {
                [buf release];
                buf = nil;
            }
            buf = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
            cap = (buf != nil) ? aligned : 0;
        }
        return buf;
    }
};

MetalContext& get_metal_context() {
    static MetalContext ctx;
    return ctx;
}

void build_dsf_table_float4(double rcut, double alpha, Float4* out_bins) {
    const double rc2 = rcut * rcut;
    const double du = rc2 / static_cast<double>(DSF_N_TAB);
    const double two_a_sqrtpi = 2.0 * alpha / SQRT_PI;
    const double a2 = alpha * alpha;
    const double a4 = a2 * a2;

    for (int k = 0; k < DSF_N_TAB; ++k) {
        const double u = (static_cast<double>(k) + 0.5) * du;
        const double z2 = a2 * u;

        double f0, f1, f2, f3;
        if (u < 1e-14 || z2 < 0.05) {
            auto deriv_series = [&](int m) -> double {
                double s = 0.0;
                double term = 1.0;
                for (int n = 0; n < 22; ++n) {
                    s += term / static_cast<double>(2 * n + 2 * m + 1);
                    term *= -z2 / static_cast<double>(n + 1);
                }
                double sign_pow = 1.0;
                for (int p = 0; p < m; ++p) sign_pow *= -a2;
                return two_a_sqrtpi * sign_pow * s;
            };
            f0 = deriv_series(0);
            f1 = deriv_series(1);
            f2 = deriv_series(2);
            f3 = deriv_series(3);
        } else {
            const double r = std::sqrt(u);
            const double g = two_a_sqrtpi * std::exp(-z2);
            const double inv_2u = 0.5 / u;
            f0 = std::erf(alpha * r) / r;
            f1 = (g - f0) * inv_2u;
            f2 = (-a2 * g - 3.0 * f1) * inv_2u;
            f3 = (a4 * g - 5.0 * f2) * inv_2u;
        }

        out_bins[k] = Float4{
            static_cast<float>(f0),
            static_cast<float>(f1),
            static_cast<float>(f2 * 0.5),
            static_cast<float>(f3 * (1.0 / 6.0))
        };
    }
}

inline void cpu_apply_mic(
    bool has_box,
    bool is_triclinic,
    double L0, double L1, double L2,
    double inv_L0, double inv_L1, double inv_L2,
    const double* H,
    const double* Hinv,
    double dx0, double dy0, double dz0,
    double& dx, double& dy, double& dz
) noexcept {
    dx = dx0;
    dy = dy0;
    dz = dz0;
    if (!has_box) return;

    if (!is_triclinic) {
        dx = dx0 - L0 * std::nearbyint(dx0 * inv_L0);
        dy = dy0 - L1 * std::nearbyint(dy0 * inv_L1);
        dz = dz0 - L2 * std::nearbyint(dz0 * inv_L2);
    } else if (H && Hinv) {
        const double f0 = Hinv[0] * dx0 + Hinv[3] * dy0 + Hinv[6] * dz0;
        const double f1 = Hinv[1] * dx0 + Hinv[4] * dy0 + Hinv[7] * dz0;
        const double f2 = Hinv[2] * dx0 + Hinv[5] * dy0 + Hinv[8] * dz0;

        const double n0_0 = std::nearbyint(f0);
        const double n0_1 = std::nearbyint(f1);
        const double n0_2 = std::nearbyint(f2);

        double best_x = dx0 - (H[0] * n0_0 + H[3] * n0_1 + H[6] * n0_2);
        double best_y = dy0 - (H[1] * n0_0 + H[4] * n0_1 + H[7] * n0_2);
        double best_z = dz0 - (H[2] * n0_0 + H[5] * n0_1 + H[8] * n0_2);
        double best2 = best_x * best_x + best_y * best_y + best_z * best_z;

        for (int da = -1; da <= 1; ++da) {
            for (int db = -1; db <= 1; ++db) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (da == 0 && db == 0 && dc == 0) continue;
                    const double na = n0_0 + da;
                    const double nb = n0_1 + db;
                    const double nc = n0_2 + dc;
                    const double rx = dx0 - (H[0] * na + H[3] * nb + H[6] * nc);
                    const double ry = dy0 - (H[1] * na + H[4] * nb + H[7] * nc);
                    const double rz = dz0 - (H[2] * na + H[5] * nb + H[8] * nc);
                    const double r2_cand = rx * rx + ry * ry + rz * rz;
                    if (r2_cand < best2) {
                        best2 = r2_cand;
                        best_x = rx;
                        best_y = ry;
                        best_z = rz;
                    }
                }
            }
        }
        dx = best_x;
        dy = best_y;
        dz = best_z;
    }
}

constexpr size_t REDUCE_MAX_GROUPS = 64;
constexpr size_t REDUCE_STRIDE = 12;

struct UInt2 {
    uint32_t x, y;
};

struct UInt4 {
    uint32_t x, y, z, w;
};

static_assert(sizeof(UInt2) == 8, "UInt2 must be 8 bytes");
static_assert(sizeof(UInt4) == 16, "UInt4 must be 16 bytes");

struct XbtGpuVerletPlan {
    size_t n_atoms = 0;
    size_t nnz = 0;
    size_t unique_undirected_pairs = 0;

    bool do_lj = false;
    bool do_coul = false;
    bool use_lfs = false;
    bool has_scaled = false;
    bool order_is_perm = false;

    double rcut = 0.0;
    double rc2 = 0.0;
    double inv_rc = 0.0;
    double inv_rc2 = 0.0;
    double alpha_dsf = 0.0;
    double coulomb_prefactor = 0.0;
    double Arc = 0.0;
    double Brc = 0.0;
    double du = 0.0;
    double inv_du = 0.0;

    std::vector<int32_t> morton_to_atom;
    std::vector<int32_t> atom_to_morton;
    std::vector<float> chg_f;

    id<MTLBuffer> buf_q_f4 = nil;
    id<MTLBuffer> buf_sig_eps = nil;
    id<MTLBuffer> buf_row_ptr = nil;
    id<MTLBuffer> buf_col_idx = nil;
    id<MTLBuffer> buf_wlj = nil;
    id<MTLBuffer> buf_wcoul = nil;
    id<MTLBuffer> buf_dsf_table = nil;
    id<MTLBuffer> buf_f_out = nil;
    id<MTLBuffer> buf_u_out = nil;
    id<MTLBuffer> buf_vir_out = nil;
    id<MTLBuffer> buf_reduce_out = nil;

    // Lazy cached buffers for GPU-resident batched MD + bonded terms
    id<MTLBuffer> buf_p_f4 = nil;
    id<MTLBuffer> buf_u_bond = nil;
    id<MTLBuffer> buf_atom_bond_ptr = nil;
    id<MTLBuffer> buf_atom_bond_idx = nil;
    id<MTLBuffer> buf_bond_atoms = nil;
    id<MTLBuffer> buf_bond_kr0 = nil;
    size_t bond_idx_cap = 0;
    size_t bond_cap = 0;

    id<MTLBuffer> buf_atom_ang_ptr = nil;
    id<MTLBuffer> buf_atom_ang_idx = nil;
    id<MTLBuffer> buf_ang_atoms = nil;
    id<MTLBuffer> buf_ang_kt0 = nil;
    size_t ang_idx_cap = 0;
    size_t ang_cap = 0;

    bool bonded_cached = false;
    size_t cached_n_bonds = 0;
    size_t cached_n_angles = 0;
    uint64_t cached_topo_hash = 0;

    std::mutex eval_mtx;

    ~XbtGpuVerletPlan() {
        auto rel = [](id<MTLBuffer>& b) {
            if (b != nil) {
                [b release];
                b = nil;
            }
        };
        rel(buf_q_f4);
        rel(buf_sig_eps);
        rel(buf_row_ptr);
        rel(buf_col_idx);
        rel(buf_wlj);
        rel(buf_wcoul);
        rel(buf_dsf_table);
        rel(buf_f_out);
        rel(buf_u_out);
        rel(buf_vir_out);
        rel(buf_reduce_out);
        rel(buf_p_f4);
        rel(buf_u_bond);
        rel(buf_atom_bond_ptr);
        rel(buf_atom_bond_idx);
        rel(buf_bond_atoms);
        rel(buf_bond_kr0);
        rel(buf_atom_ang_ptr);
        rel(buf_atom_ang_idx);
        rel(buf_ang_atoms);
        rel(buf_ang_kt0);
    }
};

inline VerletUniforms build_verlet_uniforms(
    const XbtGpuVerletPlan& plan,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    bool need_vir
) {
    const bool has_box = (box_H != nullptr);
    const bool tri = (is_triclinic != 0) && has_box && (box_Hinv != nullptr);

    VerletUniforms u{};
    u.n_atoms = static_cast<uint32_t>(plan.n_atoms);
    u.do_lj = plan.do_lj ? 1u : 0u;
    u.do_coul = plan.do_coul ? 1u : 0u;
    u.use_lfs = plan.use_lfs ? 1u : 0u;
    u.has_scaled = plan.has_scaled ? 1u : 0u;
    u.has_box = has_box ? 1u : 0u;
    u.is_triclinic = tri ? 1u : 0u;
    u.compute_virial = (need_vir && has_box) ? 1u : 0u;
    u.rcut = static_cast<float>(plan.rcut);
    u.rc2 = static_cast<float>(plan.rc2);
    u.inv_rc = static_cast<float>(plan.inv_rc);
    u.inv_rc2 = static_cast<float>(plan.inv_rc2);
    u.coulomb_prefactor = static_cast<float>(plan.coulomb_prefactor);
    u.Arc = static_cast<float>(plan.Arc);
    u.Brc = static_cast<float>(plan.Brc);
    u.du = static_cast<float>(plan.du);
    u.inv_du = static_cast<float>(plan.inv_du);

    if (has_box && !tri) {
        const double L0 = box_H[0];
        const double L1 = box_H[4];
        const double L2 = box_H[8];
        u.L0 = static_cast<float>(L0);
        u.L1 = static_cast<float>(L1);
        u.L2 = static_cast<float>(L2);
        u.inv_L0 = (L0 != 0.0) ? static_cast<float>(1.0 / L0) : 0.0f;
        u.inv_L1 = (L1 != 0.0) ? static_cast<float>(1.0 / L1) : 0.0f;
        u.inv_L2 = (L2 != 0.0) ? static_cast<float>(1.0 / L2) : 0.0f;
    } else if (tri) {
        for (int k = 0; k < 9; ++k) {
            u.H[k] = static_cast<float>(box_H[k]);
            u.Hinv[k] = static_cast<float>(box_Hinv[k]);
        }
    }
    return u;
}

inline uint64_t hash_bonded_topology(
    size_t n_bonds, const int32_t* bond_i, const int32_t* bond_j,
    const double* bond_k, const double* bond_r0,
    size_t n_angles, const int32_t* ang_i, const int32_t* ang_j, const int32_t* ang_k,
    const double* ang_ktheta, const double* ang_theta0
) {
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    auto d2u = [](double d) -> uint64_t {
        uint64_t u = 0;
        std::memcpy(&u, &d, sizeof(uint64_t));
        return u;
    };
    mix(static_cast<uint64_t>(n_bonds));
    mix(static_cast<uint64_t>(n_angles));
    for (size_t b = 0; b < n_bonds; ++b) {
        mix(static_cast<uint64_t>(static_cast<uint32_t>(bond_i[b])));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(bond_j[b])));
        mix(d2u(bond_k[b]));
        mix(d2u(bond_r0[b]));
    }
    for (size_t a = 0; a < n_angles; ++a) {
        mix(static_cast<uint64_t>(static_cast<uint32_t>(ang_i[a])));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(ang_j[a])));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(ang_k[a])));
        mix(d2u(ang_ktheta[a]));
        mix(d2u(ang_theta0[a]));
    }
    return h;
}

} // anonymous namespace

XBT_GPU_EXPORT int32_t xbt_gpu_probe_device(
    char* name_buf,
    int32_t name_cap,
    int32_t* out_backend_id,
    int32_t* out_unified_mem,
    int32_t* out_max_threads_per_tg,
    int32_t* out_simd_width,
    uint64_t* out_vram_bytes
) {
    @autoreleasepool {
        MetalContext& ctx = get_metal_context();
        if (!ctx.ensure_init() || ctx.device == nil) {
            return 0;
        }

        if (name_buf && name_cap > 0) {
            const char* cname = [[ctx.device name] UTF8String];
            if (!cname) cname = "Apple Metal GPU";
            std::strncpy(name_buf, cname, static_cast<size_t>(name_cap - 1));
            name_buf[name_cap - 1] = '\0';
        }
        if (out_backend_id) {
            *out_backend_id = 1; // Metal
        }
        if (out_unified_mem) {
            *out_unified_mem = [ctx.device hasUnifiedMemory] ? 1 : 0;
        }
        if (out_max_threads_per_tg) {
            const NSUInteger max_tg = ctx.pso_verlet
                ? [ctx.pso_verlet maxTotalThreadsPerThreadgroup]
                : [ctx.device maxThreadsPerThreadgroup].width;
            *out_max_threads_per_tg = static_cast<int32_t>(max_tg);
        }
        if (out_simd_width) {
            const NSUInteger simd_w = ctx.pso_verlet
                ? [ctx.pso_verlet threadExecutionWidth]
                : 32;
            *out_simd_width = static_cast<int32_t>(simd_w);
        }
        if (out_vram_bytes) {
            *out_vram_bytes = static_cast<uint64_t>([ctx.device recommendedMaxWorkingSetSize]);
        }
        return 1;
    }
}

XBT_GPU_EXPORT void* xbt_gpu_verlet_plan_create(
    size_t n_atoms,
    const double* q,
    const double* sigma,
    const double* epsilon,
    const double* charge,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    size_t n_tile_pairs,
    const int32_t* tile_i,
    const int32_t* tile_j,
    const int32_t* order,
    const int32_t* tile_starts,
    const int32_t* tile_ends,
    size_t n_excl,
    const uint64_t* excl_keys,
    size_t n_scaled,
    const uint64_t* scaled_keys,
    const double* scaled_weights,
    double lj14_scale,
    double coul14_scale,
    double rcut,
    double skin,
    double alpha_dsf,
    double coulomb_prefactor,
    int use_lfs
) {
    if (!q || n_atoms == 0 || rcut <= 0.0) {
        return nullptr;
    }

    MetalContext& ctx = get_metal_context();
    if (!ctx.ensure_init() || ctx.device == nil) {
        return nullptr;
    }

    auto plan = std::make_unique<XbtGpuVerletPlan>();
    plan->n_atoms = n_atoms;
    plan->do_lj = (sigma != nullptr && epsilon != nullptr);
    plan->do_coul = (charge != nullptr && coulomb_prefactor != 0.0 && alpha_dsf > 0.0);
    plan->use_lfs = (use_lfs != 0);
    plan->rcut = rcut;
    plan->rc2 = rcut * rcut;
    plan->inv_rc = 1.0 / rcut;
    plan->inv_rc2 = 1.0 / plan->rc2;
    plan->alpha_dsf = alpha_dsf;
    plan->coulomb_prefactor = coulomb_prefactor;

    if (plan->do_coul) {
        const double two_a_sqrtpi = 2.0 * alpha_dsf / SQRT_PI;
        const double z_rc = alpha_dsf * rcut;
        const double erfc_rc = std::erfc(z_rc);
        const double exp_rc = std::exp(-alpha_dsf * alpha_dsf * plan->rc2);
        plan->Arc = erfc_rc * plan->inv_rc2 + two_a_sqrtpi * exp_rc * plan->inv_rc;
        plan->Brc = erfc_rc * plan->inv_rc + plan->Arc * rcut;
        plan->du = plan->rc2 / static_cast<double>(DSF_N_TAB);
        plan->inv_du = static_cast<double>(DSF_N_TAB) / plan->rc2;
    }

    plan->morton_to_atom.resize(n_atoms, -1);
    plan->atom_to_morton.resize(n_atoms, -1);
    plan->chg_f.resize(n_atoms, 0.0f);

    std::vector<double> morton_q(3 * n_atoms, 0.0);
    std::vector<Float2> sig_eps_host(n_atoms, Float2{0.0f, 0.0f});
    std::vector<uint8_t> seen_atom(n_atoms, 0);
    bool is_perm = (order != nullptr);

    for (size_t r = 0; r < n_atoms; ++r) {
        const int32_t atom1 = order ? order[r] : static_cast<int32_t>(r + 1);
        const int32_t atom0 = atom1 - 1;
        if (atom0 < 0 || static_cast<size_t>(atom0) >= n_atoms) {
            is_perm = false;
            continue;
        }
        if (seen_atom[static_cast<size_t>(atom0)]) {
            is_perm = false;
        } else {
            seen_atom[static_cast<size_t>(atom0)] = 1;
        }
        plan->morton_to_atom[r] = atom0;
        plan->atom_to_morton[static_cast<size_t>(atom0)] = static_cast<int32_t>(r);
        const size_t ua = static_cast<size_t>(atom0);

        morton_q[3 * r + 0] = q[3 * ua + 0];
        morton_q[3 * r + 1] = q[3 * ua + 1];
        morton_q[3 * r + 2] = q[3 * ua + 2];

        if (plan->do_lj) {
            const float hs = static_cast<float>(0.5 * sigma[ua]);
            const double eps_val = epsilon[ua];
            const float se = (eps_val > 0.0) ? static_cast<float>(std::sqrt(eps_val)) : 0.0f;
            sig_eps_host[r] = Float2{hs, se};
        }
        if (plan->do_coul) {
            plan->chg_f[r] = static_cast<float>(charge[ua]);
        }
    }
    plan->order_is_perm = is_perm;

    struct AcceptedPair {
        int32_t r;
        int32_t c;
        float w_lj;
        float w_coul;
    };
    std::vector<AcceptedPair> accepted;
    std::vector<uint32_t> row_counts(n_atoms, 0);
    bool any_scaled = false;

    if (n_tile_pairs > 0 && tile_i && tile_j && order && tile_starts && tile_ends) {
        const double eff_skin = std::max(skin, 0.02);
        const double r_list = rcut + eff_skin;
        const double r_list2 = r_list * r_list;

        const bool has_box = (box_H != nullptr);
        const bool tri = (is_triclinic != 0);
        double L0 = 0.0, L1 = 0.0, L2 = 0.0;
        double inv_L0 = 0.0, inv_L1 = 0.0, inv_L2 = 0.0;
        if (has_box && !tri) {
            L0 = box_H[0];
            L1 = box_H[4];
            L2 = box_H[8];
            inv_L0 = (L0 != 0.0) ? (1.0 / L0) : 0.0;
            inv_L1 = (L1 != 0.0) ? (1.0 / L1) : 0.0;
            inv_L2 = (L2 != 0.0) ? (1.0 / L2) : 0.0;
        }

        const bool has_excl = (n_excl > 0 && excl_keys != nullptr);
        const bool has_scaled = (n_scaled > 0 && scaled_keys != nullptr);
        const bool check_topo = (has_excl || has_scaled);

        accepted.reserve(n_atoms * 64);

        auto test_pair = [&](int32_t a_pos0, int32_t b_pos0, int32_t i1, double qxi, double qyi, double qzi) {
            const int32_t j1 = order[b_pos0];
            if (j1 < 1 || static_cast<size_t>(j1) > n_atoms || b_pos0 == a_pos0) return;

            const size_t ub = static_cast<size_t>(b_pos0);
            const double dx0 = qxi - morton_q[3 * ub + 0];
            const double dy0 = qyi - morton_q[3 * ub + 1];
            const double dz0 = qzi - morton_q[3 * ub + 2];

            double dx, dy, dz;
            cpu_apply_mic(
                has_box, tri,
                L0, L1, L2, inv_L0, inv_L1, inv_L2,
                box_H, box_Hinv,
                dx0, dy0, dz0,
                dx, dy, dz
            );
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= r_list2) return;

            const int32_t lo_pos = (a_pos0 < b_pos0) ? a_pos0 : b_pos0;
            const int32_t hi_pos = (a_pos0 < b_pos0) ? b_pos0 : a_pos0;
            float w_lj = 1.0f;
            float w_coul = 1.0f;

            if (check_topo) {
                const uint64_t lo1 = static_cast<uint64_t>(i1 < j1 ? i1 : j1);
                const uint64_t hi1 = static_cast<uint64_t>(i1 < j1 ? j1 : i1);
                const uint64_t key = (lo1 << 32) | hi1;
                if (has_excl && std::binary_search(excl_keys, excl_keys + n_excl, key)) {
                    return;
                }
                if (has_scaled) {
                    const uint64_t* it = std::lower_bound(scaled_keys, scaled_keys + n_scaled, key);
                    if (it != scaled_keys + n_scaled && *it == key) {
                        const size_t pos = static_cast<size_t>(it - scaled_keys);
                        const double w = scaled_weights ? scaled_weights[pos] : lj14_scale;
                        const double w_lj_d = (w != 1.0) ? w : lj14_scale;
                        const double w_coul_d = coul14_scale;
                        if (w_lj_d == 0.0 && w_coul_d == 0.0) return;
                        w_lj = static_cast<float>(w_lj_d);
                        w_coul = static_cast<float>(w_coul_d);
                        any_scaled = true;
                    }
                }
            }

            accepted.push_back(AcceptedPair{lo_pos, hi_pos, w_lj, w_coul});
            row_counts[static_cast<size_t>(lo_pos)]++;
            row_counts[static_cast<size_t>(hi_pos)]++;
        };

        for (size_t k = 0; k < n_tile_pairs; ++k) {
            const int32_t ti1 = tile_i[k];
            const int32_t tj1 = tile_j[k];
            const int32_t si1 = tile_starts[ti1 - 1];
            const int32_t ei1 = tile_ends[ti1 - 1];

            if (ti1 == tj1) {
                for (int32_t a = si1; a < ei1; ++a) {
                    const int32_t a_pos0 = a - 1;
                    const int32_t i1 = order[a_pos0];
                    if (i1 < 1 || static_cast<size_t>(i1) > n_atoms) continue;
                    const size_t ua = static_cast<size_t>(a_pos0);
                    const double qxi = morton_q[3 * ua + 0];
                    const double qyi = morton_q[3 * ua + 1];
                    const double qzi = morton_q[3 * ua + 2];
                    for (int32_t b = a + 1; b <= ei1; ++b) {
                        test_pair(a_pos0, b - 1, i1, qxi, qyi, qzi);
                    }
                }
            } else {
                const int32_t sj1 = tile_starts[tj1 - 1];
                const int32_t ej1 = tile_ends[tj1 - 1];
                for (int32_t a = si1; a <= ei1; ++a) {
                    const int32_t a_pos0 = a - 1;
                    const int32_t i1 = order[a_pos0];
                    if (i1 < 1 || static_cast<size_t>(i1) > n_atoms) continue;
                    const size_t ua = static_cast<size_t>(a_pos0);
                    const double qxi = morton_q[3 * ua + 0];
                    const double qyi = morton_q[3 * ua + 1];
                    const double qzi = morton_q[3 * ua + 2];
                    for (int32_t b = sj1; b <= ej1; ++b) {
                        test_pair(a_pos0, b - 1, i1, qxi, qyi, qzi);
                    }
                }
            }
        }
    }

    plan->unique_undirected_pairs = accepted.size();
    plan->nnz = accepted.size() * 2;
    plan->has_scaled = any_scaled;

    @autoreleasepool {
        id<MTLDevice> dev = ctx.device;
        plan->buf_q_f4      = [dev newBufferWithLength:n_atoms * sizeof(Float4) options:MTLResourceStorageModeShared];
        plan->buf_sig_eps   = [dev newBufferWithLength:n_atoms * sizeof(Float2) options:MTLResourceStorageModeShared];
        plan->buf_row_ptr   = [dev newBufferWithLength:(n_atoms + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        plan->buf_dsf_table = [dev newBufferWithLength:DSF_N_TAB * sizeof(Float4) options:MTLResourceStorageModeShared];
        plan->buf_f_out      = [dev newBufferWithLength:3 * n_atoms * sizeof(float) options:MTLResourceStorageModeShared];
        plan->buf_u_out      = [dev newBufferWithLength:n_atoms * sizeof(float) options:MTLResourceStorageModeShared];
        plan->buf_vir_out    = [dev newBufferWithLength:9 * n_atoms * sizeof(float) options:MTLResourceStorageModeShared];
        plan->buf_reduce_out = [dev newBufferWithLength:REDUCE_MAX_GROUPS * REDUCE_STRIDE * sizeof(float) options:MTLResourceStorageModeShared];

        const size_t nnz_alloc = std::max<size_t>(1, plan->nnz);
        plan->buf_col_idx = [dev newBufferWithLength:nnz_alloc * sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (plan->has_scaled) {
            plan->buf_wlj   = [dev newBufferWithLength:nnz_alloc * sizeof(float) options:MTLResourceStorageModeShared];
            plan->buf_wcoul = [dev newBufferWithLength:nnz_alloc * sizeof(float) options:MTLResourceStorageModeShared];
        } else {
            plan->buf_wlj   = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
            plan->buf_wcoul = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        }

        if (!plan->buf_q_f4 || !plan->buf_sig_eps || !plan->buf_row_ptr || !plan->buf_dsf_table ||
            !plan->buf_f_out || !plan->buf_u_out || !plan->buf_vir_out || !plan->buf_reduce_out ||
            !plan->buf_col_idx || !plan->buf_wlj || !plan->buf_wcoul) {
            return nullptr;
        }

        std::memcpy([plan->buf_sig_eps contents], sig_eps_host.data(), n_atoms * sizeof(Float2));

        if (plan->do_coul) {
            build_dsf_table_float4(rcut, alpha_dsf, static_cast<Float4*>([plan->buf_dsf_table contents]));
        } else {
            std::memset([plan->buf_dsf_table contents], 0, DSF_N_TAB * sizeof(Float4));
        }

        uint32_t* sym_row_ptr = static_cast<uint32_t*>([plan->buf_row_ptr contents]);
        sym_row_ptr[0] = 0;
        for (size_t r = 0; r < n_atoms; ++r) {
            sym_row_ptr[r + 1] = sym_row_ptr[r] + row_counts[r];
        }

        if (plan->nnz > 0) {
            int32_t* sym_col_idx = static_cast<int32_t*>([plan->buf_col_idx contents]);
            float* sym_wlj = plan->has_scaled ? static_cast<float*>([plan->buf_wlj contents]) : nullptr;
            float* sym_wcoul = plan->has_scaled ? static_cast<float*>([plan->buf_wcoul contents]) : nullptr;

            std::vector<uint32_t> write_pos(sym_row_ptr, sym_row_ptr + n_atoms);
            for (const AcceptedPair& ap : accepted) {
                const uint32_t p1 = write_pos[static_cast<size_t>(ap.r)]++;
                const uint32_t p2 = write_pos[static_cast<size_t>(ap.c)]++;
                sym_col_idx[p1] = ap.c;
                sym_col_idx[p2] = ap.r;
                if (plan->has_scaled) {
                    sym_wlj[p1] = ap.w_lj;
                    sym_wlj[p2] = ap.w_lj;
                    sym_wcoul[p1] = ap.w_coul;
                    sym_wcoul[p2] = ap.w_coul;
                }
            }

            // Sort each row's neighbors in ascending column index for coalesced reads
            if (!plan->has_scaled) {
                for (size_t r = 0; r < n_atoms; ++r) {
                    const uint32_t b = sym_row_ptr[r];
                    const uint32_t e = sym_row_ptr[r + 1];
                    if (e - b > 1) {
                        std::sort(sym_col_idx + b, sym_col_idx + e);
                    }
                }
            } else {
                struct NeighborEntry {
                    int32_t col;
                    float wlj;
                    float wcoul;
                };
                std::vector<NeighborEntry> row_tmp;
                for (size_t r = 0; r < n_atoms; ++r) {
                    const uint32_t b = sym_row_ptr[r];
                    const uint32_t e = sym_row_ptr[r + 1];
                    const uint32_t deg = e - b;
                    if (deg > 1) {
                        row_tmp.resize(deg);
                        for (uint32_t t = 0; t < deg; ++t) {
                            row_tmp[t] = NeighborEntry{sym_col_idx[b + t], sym_wlj[b + t], sym_wcoul[b + t]};
                        }
                        std::sort(row_tmp.begin(), row_tmp.end(), [](const NeighborEntry& a, const NeighborEntry& b) {
                            return a.col < b.col;
                        });
                        for (uint32_t t = 0; t < deg; ++t) {
                            sym_col_idx[b + t] = row_tmp[t].col;
                            sym_wlj[b + t] = row_tmp[t].wlj;
                            sym_wcoul[b + t] = row_tmp[t].wcoul;
                        }
                    }
                }
            }
        }
    }

    return plan.release();
}

XBT_GPU_EXPORT double xbt_gpu_verlet_plan_eval(
    void* gpu_plan_ptr,
    const double* q,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    int compute_virial,
    double* out_forces,
    double* out_virial9
) {
    if (!gpu_plan_ptr) return 0.0;
    XbtGpuVerletPlan& plan = *static_cast<XbtGpuVerletPlan*>(gpu_plan_ptr);
    const size_t n_atoms = plan.n_atoms;

    if (out_forces && n_atoms > 0) {
        std::memset(out_forces, 0, 3 * n_atoms * sizeof(double));
    }
    if (out_virial9) {
        std::memset(out_virial9, 0, 9 * sizeof(double));
    }

    if (!q || n_atoms == 0 || plan.nnz == 0) {
        return 0.0;
    }

    MetalContext& ctx = get_metal_context();
    if (!ctx.ensure_init()) return 0.0;

    std::lock_guard<std::mutex> lock(plan.eval_mtx);

    // Pack q + charge into unified shared float4 buffer in Morton order
    Float4* q_f4 = static_cast<Float4*>([plan.buf_q_f4 contents]);
    const int32_t* m2a = plan.morton_to_atom.data();
    const float* chg_f = plan.chg_f.data();
    for (size_t r = 0; r < n_atoms; ++r) {
        const int32_t atom = m2a[r];
        if (atom >= 0 && static_cast<size_t>(atom) < n_atoms) {
            const size_t ua = static_cast<size_t>(atom);
            q_f4[r] = Float4{
                static_cast<float>(q[3 * ua + 0]),
                static_cast<float>(q[3 * ua + 1]),
                static_cast<float>(q[3 * ua + 2]),
                chg_f[r]
            };
        } else {
            q_f4[r] = Float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    const bool has_box = (box_H != nullptr);
    const bool need_vir = (compute_virial != 0) && (out_virial9 != nullptr) && has_box;
    const VerletUniforms u = build_verlet_uniforms(plan, box_H, box_Hinv, is_triclinic, need_vir);

    id<MTLComputePipelineState> pso_nb = need_vir ? ctx.pso_verlet : ctx.pso_verlet_novir;
    const NSUInteger max_tg = [pso_nb maxTotalThreadsPerThreadgroup];
    const NSUInteger tg_w = std::min<NSUInteger>(max_tg, 256);
    const NSUInteger red_max_tg = [ctx.pso_reduce maxTotalThreadsPerThreadgroup];
    const NSUInteger red_tg_w = std::min<NSUInteger>(red_max_tg, 256);
    const NSUInteger num_groups = std::min<NSUInteger>(REDUCE_MAX_GROUPS, std::max<NSUInteger>(1, (n_atoms + red_tg_w - 1) / red_tg_w));

    ReduceUniforms ru{};
    ru.n_atoms = static_cast<uint32_t>(n_atoms);
    ru.compute_virial = need_vir ? 1u : 0u;
    ru.has_bonded = 0u;
    ru.compute_ke = 0u;

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:pso_nb];
        [enc setBuffer:plan.buf_q_f4      offset:0 atIndex:0];
        [enc setBuffer:plan.buf_sig_eps   offset:0 atIndex:1];
        [enc setBuffer:plan.buf_row_ptr   offset:0 atIndex:2];
        [enc setBuffer:plan.buf_col_idx   offset:0 atIndex:3];
        [enc setBuffer:plan.buf_wlj       offset:0 atIndex:4];
        [enc setBuffer:plan.buf_wcoul     offset:0 atIndex:5];
        [enc setBuffer:plan.buf_dsf_table offset:0 atIndex:6];
        [enc setBuffer:plan.buf_f_out     offset:0 atIndex:7];
        [enc setBuffer:plan.buf_u_out     offset:0 atIndex:8];
        [enc setBuffer:plan.buf_vir_out   offset:0 atIndex:9];
        [enc setBytes:&u length:sizeof(VerletUniforms) atIndex:10];

        const MTLSize tgSize = MTLSizeMake(tg_w, 1, 1);
        const MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(n_atoms) * 8u, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

        [enc setComputePipelineState:ctx.pso_reduce];
        [enc setBuffer:plan.buf_u_out      offset:0 atIndex:0];
        [enc setBuffer:plan.buf_vir_out    offset:0 atIndex:1];
        [enc setBuffer:ctx.dummy_buf       offset:0 atIndex:2];
        [enc setBuffer:ctx.dummy_buf       offset:0 atIndex:3];
        [enc setBuffer:plan.buf_reduce_out offset:0 atIndex:4];
        [enc setBytes:&ru length:sizeof(ReduceUniforms) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(red_tg_w, 1, 1)];

        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    const float* f_out = static_cast<const float*>([plan.buf_f_out contents]);
    if (out_forces) {
        for (size_t r = 0; r < n_atoms; ++r) {
            const int32_t atom = m2a[r];
            if (atom >= 0 && static_cast<size_t>(atom) < n_atoms) {
                const size_t ua = static_cast<size_t>(atom);
                out_forces[3 * ua + 0] += static_cast<double>(f_out[3 * r + 0]);
                out_forces[3 * ua + 1] += static_cast<double>(f_out[3 * r + 1]);
                out_forces[3 * ua + 2] += static_cast<double>(f_out[3 * r + 2]);
            }
        }
    }

    const float* partial = static_cast<const float*>([plan.buf_reduce_out contents]);
    double sum_u = 0.0;
    for (NSUInteger g = 0; g < num_groups; ++g) {
        sum_u += static_cast<double>(partial[g * REDUCE_STRIDE + 0]);
    }

    if (need_vir) {
        double v_acc[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (NSUInteger g = 0; g < num_groups; ++g) {
            const size_t base = g * REDUCE_STRIDE + 3;
            for (int v = 0; v < 9; ++v) {
                v_acc[v] += static_cast<double>(partial[base + v]);
            }
        }
        for (int v = 0; v < 9; ++v) {
            out_virial9[v] = 0.5 * v_acc[v];
        }
    }

    return 0.5 * sum_u;
}

XBT_GPU_EXPORT double xbt_gpu_md_batch_run(
    void* gpu_plan_ptr,
    double* q,
    double* p,
    const double* inv_mass,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    size_t n_bonds, const int32_t* bond_i, const int32_t* bond_j, const double* bond_k, const double* bond_r0,
    size_t n_angles, const int32_t* ang_i, const int32_t* ang_j, const int32_t* ang_k, const double* ang_ktheta, const double* ang_theta0,
    double dt,
    int n_steps,
    double* out_forces,
    double* out_pe,
    double* out_ke
) {
    if (out_pe) *out_pe = 0.0;
    if (out_ke) *out_ke = 0.0;
    if (!gpu_plan_ptr || !q || !p || !inv_mass || n_steps < 0) {
        return 0.0;
    }

    XbtGpuVerletPlan& plan = *static_cast<XbtGpuVerletPlan*>(gpu_plan_ptr);
    const size_t n_atoms = plan.n_atoms;
    if (n_atoms == 0) {
        return 0.0;
    }

    MetalContext& ctx = get_metal_context();
    if (!ctx.ensure_init()) return 0.0;

    std::lock_guard<std::mutex> lock(plan.eval_mtx);

    // Ensure lazy per-plan batch buffers
    @autoreleasepool {
        id<MTLDevice> dev = ctx.device;
        if (!plan.buf_p_f4) {
            plan.buf_p_f4 = [dev newBufferWithLength:n_atoms * sizeof(Float4) options:MTLResourceStorageModeShared];
        }
        if (!plan.buf_u_bond) {
            plan.buf_u_bond = [dev newBufferWithLength:n_atoms * sizeof(float) options:MTLResourceStorageModeShared];
        }
        if (!plan.buf_atom_bond_ptr) {
            plan.buf_atom_bond_ptr = [dev newBufferWithLength:(n_atoms + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        }
        if (!plan.buf_atom_ang_ptr) {
            plan.buf_atom_ang_ptr = [dev newBufferWithLength:(n_atoms + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        }
    }
    if (!plan.buf_p_f4 || !plan.buf_u_bond || !plan.buf_atom_bond_ptr || !plan.buf_atom_ang_ptr) {
        return 0.0;
    }

    // Pack q + charge and p + inv_mass into Morton-ordered float4 buffers
    Float4* q_f4 = static_cast<Float4*>([plan.buf_q_f4 contents]);
    Float4* p_f4 = static_cast<Float4*>([plan.buf_p_f4 contents]);
    const int32_t* m2a = plan.morton_to_atom.data();
    const int32_t* a2m = plan.atom_to_morton.data();
    const float* chg_f = plan.chg_f.data();
    for (size_t r = 0; r < n_atoms; ++r) {
        const int32_t atom = m2a[r];
        if (atom >= 0 && static_cast<size_t>(atom) < n_atoms) {
            const size_t ua = static_cast<size_t>(atom);
            q_f4[r] = Float4{
                static_cast<float>(q[3 * ua + 0]),
                static_cast<float>(q[3 * ua + 1]),
                static_cast<float>(q[3 * ua + 2]),
                chg_f[r]
            };
            p_f4[r] = Float4{
                static_cast<float>(p[3 * ua + 0]),
                static_cast<float>(p[3 * ua + 1]),
                static_cast<float>(p[3 * ua + 2]),
                static_cast<float>(inv_mass[ua])
            };
        } else {
            q_f4[r] = Float4{0.0f, 0.0f, 0.0f, 0.0f};
            p_f4[r] = Float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    const size_t eff_n_bonds = (n_bonds > 0 && bond_i && bond_j && bond_k && bond_r0) ? n_bonds : 0;
    const size_t eff_n_angles = (n_angles > 0 && ang_i && ang_j && ang_k && ang_ktheta && ang_theta0) ? n_angles : 0;
    const bool has_bonded = (eff_n_bonds > 0 || eff_n_angles > 0);

    if (has_bonded) {
        const uint64_t topo_hash = hash_bonded_topology(
            eff_n_bonds, bond_i, bond_j, bond_k, bond_r0,
            eff_n_angles, ang_i, ang_j, ang_k, ang_ktheta, ang_theta0
        );

        if (!plan.bonded_cached ||
            plan.cached_n_bonds != eff_n_bonds ||
            plan.cached_n_angles != eff_n_angles ||
            plan.cached_topo_hash != topo_hash) {

            uint32_t* atom_bond_ptr = static_cast<uint32_t*>([plan.buf_atom_bond_ptr contents]);
            if (eff_n_bonds == 0) {
                std::memset(atom_bond_ptr, 0, (n_atoms + 1) * sizeof(uint32_t));
            } else {
                @autoreleasepool {
                    MetalContext::ensure_shared_buf(ctx.device, plan.buf_bond_atoms, plan.bond_cap, eff_n_bonds * sizeof(UInt2));
                    MetalContext::ensure_shared_buf(ctx.device, plan.buf_bond_kr0, plan.bond_cap, eff_n_bonds * sizeof(Float2));
                    MetalContext::ensure_shared_buf(ctx.device, plan.buf_atom_bond_idx, plan.bond_idx_cap, 2 * eff_n_bonds * sizeof(uint32_t));
                }
                UInt2* b_atoms = static_cast<UInt2*>([plan.buf_bond_atoms contents]);
                Float2* b_kr0 = static_cast<Float2*>([plan.buf_bond_kr0 contents]);
                uint32_t* b_idx = static_cast<uint32_t*>([plan.buf_atom_bond_idx contents]);

                std::vector<uint32_t> counts(n_atoms, 0);
                for (size_t b = 0; b < eff_n_bonds; ++b) {
                    const int32_t ai = bond_i[b];
                    const int32_t aj = bond_j[b];
                    if (ai >= 0 && static_cast<size_t>(ai) < n_atoms &&
                        aj >= 0 && static_cast<size_t>(aj) < n_atoms) {
                        const int32_t ri = a2m[ai];
                        const int32_t rj = a2m[aj];
                        if (ri >= 0 && rj >= 0) {
                            b_atoms[b] = UInt2{static_cast<uint32_t>(ri), static_cast<uint32_t>(rj)};
                            b_kr0[b] = Float2{static_cast<float>(bond_k[b]), static_cast<float>(bond_r0[b])};
                            counts[static_cast<size_t>(ri)]++;
                            counts[static_cast<size_t>(rj)]++;
                            continue;
                        }
                    }
                    b_atoms[b] = UInt2{0u, 0u};
                    b_kr0[b] = Float2{0.0f, 0.0f};
                }

                atom_bond_ptr[0] = 0;
                for (size_t r = 0; r < n_atoms; ++r) {
                    atom_bond_ptr[r + 1] = atom_bond_ptr[r] + counts[r];
                }
                std::vector<uint32_t> write_pos(atom_bond_ptr, atom_bond_ptr + n_atoms);
                for (size_t b = 0; b < eff_n_bonds; ++b) {
                    const int32_t ai = bond_i[b];
                    const int32_t aj = bond_j[b];
                    if (ai >= 0 && static_cast<size_t>(ai) < n_atoms &&
                        aj >= 0 && static_cast<size_t>(aj) < n_atoms) {
                        const int32_t ri = a2m[ai];
                        const int32_t rj = a2m[aj];
                        if (ri >= 0 && rj >= 0) {
                            b_idx[write_pos[static_cast<size_t>(ri)]++] = (static_cast<uint32_t>(b) << 1u) | 0u;
                            b_idx[write_pos[static_cast<size_t>(rj)]++] = (static_cast<uint32_t>(b) << 1u) | 1u;
                        }
                    }
                }
            }

            uint32_t* atom_ang_ptr = static_cast<uint32_t*>([plan.buf_atom_ang_ptr contents]);
            if (eff_n_angles == 0) {
                std::memset(atom_ang_ptr, 0, (n_atoms + 1) * sizeof(uint32_t));
            } else {
                @autoreleasepool {
                    MetalContext::ensure_shared_buf(ctx.device, plan.buf_ang_atoms, plan.ang_cap, eff_n_angles * sizeof(UInt4));
                    MetalContext::ensure_shared_buf(ctx.device, plan.buf_ang_kt0, plan.ang_cap, eff_n_angles * sizeof(Float2));
                    MetalContext::ensure_shared_buf(ctx.device, plan.buf_atom_ang_idx, plan.ang_idx_cap, 3 * eff_n_angles * sizeof(uint32_t));
                }
                UInt4* a_atoms = static_cast<UInt4*>([plan.buf_ang_atoms contents]);
                Float2* a_kt0 = static_cast<Float2*>([plan.buf_ang_kt0 contents]);
                uint32_t* a_idx = static_cast<uint32_t*>([plan.buf_atom_ang_idx contents]);

                std::vector<uint32_t> counts(n_atoms, 0);
                for (size_t a = 0; a < eff_n_angles; ++a) {
                    const int32_t ai = ang_i[a];
                    const int32_t aj = ang_j[a];
                    const int32_t ak = ang_k[a];
                    if (ai >= 0 && static_cast<size_t>(ai) < n_atoms &&
                        aj >= 0 && static_cast<size_t>(aj) < n_atoms &&
                        ak >= 0 && static_cast<size_t>(ak) < n_atoms) {
                        const int32_t ri = a2m[ai];
                        const int32_t rj = a2m[aj];
                        const int32_t rk = a2m[ak];
                        if (ri >= 0 && rj >= 0 && rk >= 0) {
                            a_atoms[a] = UInt4{static_cast<uint32_t>(ri), static_cast<uint32_t>(rj), static_cast<uint32_t>(rk), 0u};
                            a_kt0[a] = Float2{static_cast<float>(ang_ktheta[a]), static_cast<float>(ang_theta0[a])};
                            counts[static_cast<size_t>(ri)]++;
                            counts[static_cast<size_t>(rj)]++;
                            counts[static_cast<size_t>(rk)]++;
                            continue;
                        }
                    }
                    a_atoms[a] = UInt4{0u, 0u, 0u, 0u};
                    a_kt0[a] = Float2{0.0f, 0.0f};
                }

                atom_ang_ptr[0] = 0;
                for (size_t r = 0; r < n_atoms; ++r) {
                    atom_ang_ptr[r + 1] = atom_ang_ptr[r] + counts[r];
                }
                std::vector<uint32_t> write_pos(atom_ang_ptr, atom_ang_ptr + n_atoms);
                for (size_t a = 0; a < eff_n_angles; ++a) {
                    const int32_t ai = ang_i[a];
                    const int32_t aj = ang_j[a];
                    const int32_t ak = ang_k[a];
                    if (ai >= 0 && static_cast<size_t>(ai) < n_atoms &&
                        aj >= 0 && static_cast<size_t>(aj) < n_atoms &&
                        ak >= 0 && static_cast<size_t>(ak) < n_atoms) {
                        const int32_t ri = a2m[ai];
                        const int32_t rj = a2m[aj];
                        const int32_t rk = a2m[ak];
                        if (ri >= 0 && rj >= 0 && rk >= 0) {
                            a_idx[write_pos[static_cast<size_t>(ri)]++] = (static_cast<uint32_t>(a) << 2u) | 0u;
                            a_idx[write_pos[static_cast<size_t>(rj)]++] = (static_cast<uint32_t>(a) << 2u) | 1u;
                            a_idx[write_pos[static_cast<size_t>(rk)]++] = (static_cast<uint32_t>(a) << 2u) | 2u;
                        }
                    }
                }
            }

            plan.bonded_cached = true;
            plan.cached_n_bonds = eff_n_bonds;
            plan.cached_n_angles = eff_n_angles;
            plan.cached_topo_hash = topo_hash;
        }
    }

    const VerletUniforms u = build_verlet_uniforms(plan, box_H, box_Hinv, is_triclinic, false);
    const NSUInteger tg_nb = std::min<NSUInteger>([ctx.pso_verlet_novir maxTotalThreadsPerThreadgroup], 256);
    const NSUInteger tg_kd = std::min<NSUInteger>([ctx.pso_verlet_kd maxTotalThreadsPerThreadgroup], 256);
    const NSUInteger tg_bnd = std::min<NSUInteger>([ctx.pso_bonded maxTotalThreadsPerThreadgroup], 256);
    const NSUInteger red_tg_w = std::min<NSUInteger>([ctx.pso_reduce maxTotalThreadsPerThreadgroup], 256);
    const NSUInteger num_groups = std::min<NSUInteger>(REDUCE_MAX_GROUPS, std::max<NSUInteger>(1, (n_atoms + red_tg_w - 1) / red_tg_w));

    const MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(n_atoms), 1, 1);
    const MTLSize gridSizeNb = MTLSizeMake(static_cast<NSUInteger>(n_atoms) * 8u, 1, 1);
    const MTLSize tgSizeNb = MTLSizeMake(tg_nb, 1, 1);
    const MTLSize tgSizeKd = MTLSizeMake(tg_kd, 1, 1);
    const MTLSize tgSizeBnd = MTLSizeMake(tg_bnd, 1, 1);

    id<MTLBuffer> buf_b_idx   = (eff_n_bonds > 0 && plan.buf_atom_bond_idx) ? plan.buf_atom_bond_idx : ctx.dummy_buf;
    id<MTLBuffer> buf_b_atoms = (eff_n_bonds > 0 && plan.buf_bond_atoms)    ? plan.buf_bond_atoms    : ctx.dummy_buf;
    id<MTLBuffer> buf_b_kr0   = (eff_n_bonds > 0 && plan.buf_bond_kr0)      ? plan.buf_bond_kr0      : ctx.dummy_buf;
    id<MTLBuffer> buf_a_idx   = (eff_n_angles > 0 && plan.buf_atom_ang_idx) ? plan.buf_atom_ang_idx  : ctx.dummy_buf;
    id<MTLBuffer> buf_a_atoms = (eff_n_angles > 0 && plan.buf_ang_atoms)    ? plan.buf_ang_atoms     : ctx.dummy_buf;
    id<MTLBuffer> buf_a_kt0   = (eff_n_angles > 0 && plan.buf_ang_kt0)      ? plan.buf_ang_kt0       : ctx.dummy_buf;

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        auto encode_forces = [&]() {
            [enc setComputePipelineState:ctx.pso_verlet_novir];
            [enc setBuffer:plan.buf_q_f4      offset:0 atIndex:0];
            [enc setBuffer:plan.buf_sig_eps   offset:0 atIndex:1];
            [enc setBuffer:plan.buf_row_ptr   offset:0 atIndex:2];
            [enc setBuffer:plan.buf_col_idx   offset:0 atIndex:3];
            [enc setBuffer:plan.buf_wlj       offset:0 atIndex:4];
            [enc setBuffer:plan.buf_wcoul     offset:0 atIndex:5];
            [enc setBuffer:plan.buf_dsf_table offset:0 atIndex:6];
            [enc setBuffer:plan.buf_f_out     offset:0 atIndex:7];
            [enc setBuffer:plan.buf_u_out     offset:0 atIndex:8];
            [enc setBuffer:plan.buf_vir_out   offset:0 atIndex:9];
            [enc setBytes:&u length:sizeof(VerletUniforms) atIndex:10];
            [enc dispatchThreads:gridSizeNb threadsPerThreadgroup:tgSizeNb];

            if (has_bonded) {
                [enc setComputePipelineState:ctx.pso_bonded];
                [enc setBuffer:plan.buf_q_f4          offset:0 atIndex:0];
                [enc setBuffer:plan.buf_atom_bond_ptr offset:0 atIndex:1];
                [enc setBuffer:buf_b_idx              offset:0 atIndex:2];
                [enc setBuffer:buf_b_atoms            offset:0 atIndex:3];
                [enc setBuffer:buf_b_kr0              offset:0 atIndex:4];
                [enc setBuffer:plan.buf_atom_ang_ptr  offset:0 atIndex:5];
                [enc setBuffer:buf_a_idx              offset:0 atIndex:6];
                [enc setBuffer:buf_a_atoms            offset:0 atIndex:7];
                [enc setBuffer:buf_a_kt0              offset:0 atIndex:8];
                [enc setBuffer:plan.buf_f_out         offset:0 atIndex:9];
                [enc setBuffer:plan.buf_u_bond        offset:0 atIndex:10];
                [enc setBytes:&u length:sizeof(VerletUniforms) atIndex:11];
                [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSizeBnd];
            }
        };

        auto encode_kick_drift = [&](float kick_dt, float drift_dt) {
            IntegratorUniforms iu{};
            iu.n_atoms = static_cast<uint32_t>(n_atoms);
            iu.kick_dt = kick_dt;
            iu.drift_dt = drift_dt;
            iu._pad0 = 0u;
            [enc setComputePipelineState:ctx.pso_verlet_kd];
            [enc setBuffer:plan.buf_q_f4  offset:0 atIndex:0];
            [enc setBuffer:plan.buf_p_f4  offset:0 atIndex:1];
            [enc setBuffer:plan.buf_f_out offset:0 atIndex:2];
            [enc setBytes:&iu length:sizeof(IntegratorUniforms) atIndex:3];
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSizeKd];
        };

        // Initial force evaluation at q(0)
        encode_forces();

        constexpr int STEPS_PER_CMDBUF = 100;
        if (n_steps > 0) {
            for (int step = 0; step < n_steps; ++step) {
                if (step > 0 && (step % STEPS_PER_CMDBUF) == 0) {
                    [enc endEncoding];
                    [cmdBuf commit];
                    cmdBuf = [ctx.queue commandBuffer];
                    enc = [cmdBuf computeCommandEncoder];
                }
                const float kick_dt = (step == 0)
                    ? static_cast<float>(0.5 * dt)
                    : static_cast<float>(dt);
                encode_kick_drift(kick_dt, static_cast<float>(dt));
                encode_forces();
            }
            encode_kick_drift(static_cast<float>(0.5 * dt), 0.0f);
        }

        ReduceUniforms ru{};
        ru.n_atoms = static_cast<uint32_t>(n_atoms);
        ru.compute_virial = 0u;
        ru.has_bonded = has_bonded ? 1u : 0u;
        ru.compute_ke = 1u;

        [enc setComputePipelineState:ctx.pso_reduce];
        [enc setBuffer:plan.buf_u_out      offset:0 atIndex:0];
        [enc setBuffer:plan.buf_vir_out    offset:0 atIndex:1];
        [enc setBuffer:plan.buf_u_bond     offset:0 atIndex:2];
        [enc setBuffer:plan.buf_p_f4       offset:0 atIndex:3];
        [enc setBuffer:plan.buf_reduce_out offset:0 atIndex:4];
        [enc setBytes:&ru length:sizeof(ReduceUniforms) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(red_tg_w, 1, 1)];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    // Unpack final q, p, and optional final forces in original atom order
    const Float4* q_out_f4 = static_cast<const Float4*>([plan.buf_q_f4 contents]);
    const Float4* p_out_f4 = static_cast<const Float4*>([plan.buf_p_f4 contents]);
    const float* f_out = static_cast<const float*>([plan.buf_f_out contents]);

    for (size_t r = 0; r < n_atoms; ++r) {
        const int32_t atom = m2a[r];
        if (atom >= 0 && static_cast<size_t>(atom) < n_atoms) {
            const size_t ua = static_cast<size_t>(atom);
            q[3 * ua + 0] = static_cast<double>(q_out_f4[r].x);
            q[3 * ua + 1] = static_cast<double>(q_out_f4[r].y);
            q[3 * ua + 2] = static_cast<double>(q_out_f4[r].z);
            p[3 * ua + 0] = static_cast<double>(p_out_f4[r].x);
            p[3 * ua + 1] = static_cast<double>(p_out_f4[r].y);
            p[3 * ua + 2] = static_cast<double>(p_out_f4[r].z);
            if (out_forces) {
                out_forces[3 * ua + 0] = static_cast<double>(f_out[3 * r + 0]);
                out_forces[3 * ua + 1] = static_cast<double>(f_out[3 * r + 1]);
                out_forces[3 * ua + 2] = static_cast<double>(f_out[3 * r + 2]);
            }
        }
    }

    const float* partial = static_cast<const float*>([plan.buf_reduce_out contents]);
    double sum_u_nb = 0.0;
    double sum_u_bond = 0.0;
    double sum_ke = 0.0;
    for (NSUInteger g = 0; g < num_groups; ++g) {
        const size_t base = g * REDUCE_STRIDE;
        sum_u_nb   += static_cast<double>(partial[base + 0]);
        sum_u_bond += static_cast<double>(partial[base + 1]);
        sum_ke     += static_cast<double>(partial[base + 2]);
    }

    const double pe = 0.5 * sum_u_nb + sum_u_bond;
    if (out_pe) *out_pe = pe;
    if (out_ke) *out_ke = sum_ke;
    return pe + sum_ke;
}

XBT_GPU_EXPORT void xbt_gpu_verlet_plan_destroy(void* gpu_plan_ptr) {
    if (!gpu_plan_ptr) return;
    delete static_cast<XbtGpuVerletPlan*>(gpu_plan_ptr);
}

XBT_GPU_EXPORT size_t xbt_gpu_verlet_plan_num_pairs(void* gpu_plan_ptr) {
    if (!gpu_plan_ptr) return 0;
    const XbtGpuVerletPlan* plan = static_cast<const XbtGpuVerletPlan*>(gpu_plan_ptr);
    return plan->unique_undirected_pairs;
}

XBT_GPU_EXPORT void xbt_gpu_ternary_matvec_dualrail(
    size_t M,
    size_t K_words,
    const uint64_t* W_P,
    const uint64_t* W_N,
    const double* row_scales,
    const uint64_t* x_P,
    const uint64_t* x_N,
    double x_scale,
    double* y_out
) {
    if (!y_out || M == 0) return;
    if (K_words == 0 || (!W_P && !W_N) || (!x_P && !x_N)) {
        std::memset(y_out, 0, M * sizeof(double));
        return;
    }

    MetalContext& ctx = get_metal_context();
    if (!ctx.ensure_init()) return;

    std::lock_guard<std::mutex> lk(ctx.scratch_mtx);

    const size_t W_bytes = M * K_words * sizeof(uint64_t);
    const size_t x_bytes = K_words * sizeof(uint64_t);
    const size_t acc_bytes = M * sizeof(int64_t);

    id<MTLBuffer> buf_WP = W_P ? MetalContext::ensure_shared_buf(ctx.device, ctx.mv_WP, ctx.mv_W_cap, W_bytes) : ctx.dummy_buf;
    id<MTLBuffer> buf_WN = W_N ? MetalContext::ensure_shared_buf(ctx.device, ctx.mv_WN, ctx.mv_W_cap, W_bytes) : ctx.dummy_buf;
    id<MTLBuffer> buf_xP = x_P ? MetalContext::ensure_shared_buf(ctx.device, ctx.mv_xP, ctx.mv_x_cap, x_bytes) : ctx.dummy_buf;
    id<MTLBuffer> buf_xN = x_N ? MetalContext::ensure_shared_buf(ctx.device, ctx.mv_xN, ctx.mv_x_cap, x_bytes) : ctx.dummy_buf;
    id<MTLBuffer> buf_acc = MetalContext::ensure_shared_buf(ctx.device, ctx.mv_acc, ctx.mv_acc_cap, acc_bytes);

    if (W_P) std::memcpy([buf_WP contents], W_P, W_bytes);
    if (W_N) std::memcpy([buf_WN contents], W_N, W_bytes);
    if (x_P) std::memcpy([buf_xP contents], x_P, x_bytes);
    if (x_N) std::memcpy([buf_xN contents], x_N, x_bytes);

    MatvecUniforms u{};
    u.M = static_cast<uint32_t>(M);
    u.K_words = static_cast<uint32_t>(K_words);
    u.has_WP = W_P ? 1u : 0u;
    u.has_WN = W_N ? 1u : 0u;
    u.has_xP = x_P ? 1u : 0u;
    u.has_xN = x_N ? 1u : 0u;

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:ctx.pso_matvec];
        [enc setBuffer:buf_WP  offset:0 atIndex:0];
        [enc setBuffer:buf_WN  offset:0 atIndex:1];
        [enc setBuffer:buf_xP  offset:0 atIndex:2];
        [enc setBuffer:buf_xN  offset:0 atIndex:3];
        [enc setBuffer:buf_acc offset:0 atIndex:4];
        [enc setBytes:&u length:sizeof(MatvecUniforms) atIndex:5];

        const NSUInteger max_tg = [ctx.pso_matvec maxTotalThreadsPerThreadgroup];
        const NSUInteger tg_w = std::min<NSUInteger>(max_tg, 256);
        const MTLSize tgSize = MTLSizeMake(tg_w, 1, 1);
        const MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    const int64_t* acc_ptr = static_cast<const int64_t*>([buf_acc contents]);
    for (size_t m = 0; m < M; ++m) {
        const double r_scale = row_scales ? row_scales[m] : 1.0;
        y_out[m] = r_scale * x_scale * static_cast<double>(acc_ptr[m]);
    }
}

XBT_GPU_EXPORT double xbt_gpu_msm_stencil27(
    size_t Ml,
    const double* Q_grid,
    const double* stencil_k27,
    double prefactor,
    double* out_Phi_grid
) {
    if (Ml == 0 || !Q_grid || !stencil_k27 || !out_Phi_grid) {
        return 0.0;
    }

    MetalContext& ctx = get_metal_context();
    if (!ctx.ensure_init()) return 0.0;

    std::lock_guard<std::mutex> lk(ctx.scratch_mtx);

    const size_t n_pts = Ml * Ml * Ml;
    const size_t grid_bytes = n_pts * sizeof(float);
    id<MTLBuffer> buf_Q = MetalContext::ensure_shared_buf(ctx.device, ctx.st_Q, ctx.st_grid_cap, grid_bytes);
    id<MTLBuffer> buf_Phi = MetalContext::ensure_shared_buf(ctx.device, ctx.st_Phi, ctx.st_grid_cap, grid_bytes);
    if (!ctx.st_k27) {
        ctx.st_k27 = [ctx.device newBufferWithLength:27 * sizeof(float) options:MTLResourceStorageModeShared];
    }

    float* q_f = static_cast<float*>([buf_Q contents]);
    for (size_t i = 0; i < n_pts; ++i) {
        q_f[i] = static_cast<float>(Q_grid[i]);
    }

    float* k27_f = static_cast<float*>([ctx.st_k27 contents]);
    for (int t = 0; t < 27; ++t) {
        k27_f[t] = static_cast<float>(stencil_k27[t]);
    }

    const uint32_t u_Ml = static_cast<uint32_t>(Ml);

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:ctx.pso_stencil27];
        [enc setBuffer:buf_Q      offset:0 atIndex:0];
        [enc setBuffer:ctx.st_k27 offset:0 atIndex:1];
        [enc setBuffer:buf_Phi    offset:0 atIndex:2];
        [enc setBytes:&u_Ml length:sizeof(uint32_t) atIndex:3];

        const MTLSize tgSize = MTLSizeMake(4, 4, 4);
        const MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(Ml),
                                             static_cast<NSUInteger>(Ml),
                                             static_cast<NSUInteger>(Ml));

        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    const float* phi_f = static_cast<const float*>([buf_Phi contents]);
    double sum_qp = 0.0;
    for (size_t i = 0; i < n_pts; ++i) {
        const double pval = static_cast<double>(phi_f[i]);
        out_Phi_grid[i] = pval;
        sum_qp += Q_grid[i] * pval;
    }

    return 0.5 * prefactor * sum_qp;
}
