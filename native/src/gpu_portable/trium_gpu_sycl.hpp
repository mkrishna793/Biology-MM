#pragma once
// =============================================================================
// Trium-XBT Universal SYCL 2020 Portable GPU Backend Header
// =============================================================================
// Targets Intel Arc / Data Center GPU (DPC++), AMD ROCm, and NVIDIA CUDA
// (AdaptiveCpp / oneAPI) via standard SYCL 2020 (<sycl/sycl.hpp>) and USM.
//
// Features:
//   - Device introspection: max_work_group_size, sub_group_sizes,
//     local_mem_size, global_mem_size -> auto-tuned nd_range parameters
//   - eval_lj_dsf_csr: Full Symmetric CSR Verlet nonbonded (NoBox,
//     OrthorhombicBox, TriclinicBox), LFS LJ + 4096-bin midpoint degree-3 DSF
//   - ternary_matvec_dualrail: Dual-rail bit-sliced sycl::popcount MatVec
//   - msm_stencil27: 3D periodic 27-point balanced ternary MSM grid stencil
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(__has_include)
#  if __has_include(<sycl/sycl.hpp>)
#    define TRIUM_HAS_SYCL 1
#    include <sycl/sycl.hpp>
#  else
#    define TRIUM_HAS_SYCL 0
#  endif
#else
#  define TRIUM_HAS_SYCL 0
#endif

namespace trium::gpu {

struct SyclDeviceProfile {
    std::string device_name = "unavailable";
    std::size_t max_work_group_size = 64;
    std::size_t preferred_sub_group_size = 32;
    std::size_t local_mem_size = 0;
    std::size_t global_mem_size = 0;
    std::size_t tuned_wg_1d = 64;
    std::array<std::size_t, 3> tuned_wg_3d{4, 4, 4};
    bool supports_usm_device = false;
    bool supports_usm_shared = false;
};

struct LjDsfCsrParams {
    std::uint32_t n_atoms = 0;
    std::uint32_t n_edges = 0;
    std::uint32_t box_mode = 0;   // 0 = NoBox, 1 = OrthorhombicBox, 2 = TriclinicBox
    bool do_lj = true;
    bool do_coul = true;
    bool use_lfs = true;
    bool need_virial = false;
    float rc2 = 0.0f;
    float rcut = 0.0f;
    float inv_rc = 0.0f;
    float inv_rc2 = 0.0f;
    float du = 0.0f;
    float inv_du = 0.0f;
    float coulomb_prefactor = 0.0f;
    float Arc = 0.0f;
    float Brc = 0.0f;
    std::array<float, 3> box_L{0.0f, 0.0f, 0.0f};
    std::array<float, 3> box_invL{0.0f, 0.0f, 0.0f};
    std::array<float, 9> box_H{0.0f};
    std::array<float, 9> box_Hinv{0.0f};
};

struct Float4 {
    float x, y, z, w;
};

struct Float2 {
    float x, y;
};

#if TRIUM_HAS_SYCL

class TriumSyclEngine {
public:
    TriumSyclEngine() : q_(sycl::default_selector_v) {
        probe_device();
    }

    template <typename Selector>
    explicit TriumSyclEngine(const Selector& selector) : q_(selector) {
        probe_device();
    }

    explicit TriumSyclEngine(sycl::queue q) : q_(std::move(q)) {
        probe_device();
    }

    static constexpr bool is_compiled_with_sycl() noexcept { return true; }

    const SyclDeviceProfile& profile() const noexcept { return profile_; }
    sycl::queue& queue() noexcept { return q_; }
    const sycl::queue& queue() const noexcept { return q_; }

    // Allocate USM device or shared buffer depending on device capabilities
    template <typename T>
    T* alloc_usm(std::size_t count, bool prefer_shared = false) {
        if (prefer_shared && profile_.supports_usm_shared) {
            return sycl::malloc_shared<T>(count, q_);
        }
        if (profile_.supports_usm_device) {
            return sycl::malloc_device<T>(count, q_);
        }
        return sycl::malloc_shared<T>(count, q_);
    }

    void free_usm(void* ptr) {
        if (ptr) sycl::free(ptr, q_);
    }

    // -------------------------------------------------------------------------
    // Kernel 1: Full Symmetric CSR LJ + DSF Coulomb (NoBox / Ortho / Triclinic)
    // -------------------------------------------------------------------------
    void eval_lj_dsf_csr(
        const LjDsfCsrParams& p,
        const Float4* h_q_f4,
        const Float2* h_sig_eps,
        const std::uint32_t* h_sym_row_ptr,
        const std::uint32_t* h_sym_col_idx,
        const float* h_sym_wlj,
        const float* h_sym_wcoul,
        const Float4* h_dsf_table,
        Float4* h_out_forces,
        float* h_out_u_half,
        float* h_out_vir9
    ) {
        if (p.n_atoms == 0) return;
        const std::size_t safe_edges = std::max<std::size_t>(1, p.n_edges);
        const bool use_shared = !profile_.supports_usm_device && profile_.supports_usm_shared;

        auto* d_q = alloc_usm<Float4>(p.n_atoms, use_shared);
        auto* d_se = alloc_usm<Float2>(p.n_atoms, use_shared);
        auto* d_rp = alloc_usm<std::uint32_t>(p.n_atoms + 1, use_shared);
        auto* d_ci = alloc_usm<std::uint32_t>(safe_edges, use_shared);
        auto* d_wlj = alloc_usm<float>(safe_edges, use_shared);
        auto* d_wcoul = alloc_usm<float>(safe_edges, use_shared);
        auto* d_tbl = alloc_usm<Float4>(4096, use_shared);
        auto* d_f = alloc_usm<Float4>(p.n_atoms, use_shared);
        auto* d_uh = alloc_usm<float>(p.n_atoms, use_shared);
        const std::size_t vir_count = p.need_virial ? (9 * std::size_t(p.n_atoms)) : 1;
        auto* d_vir = alloc_usm<float>(vir_count, use_shared);

        q_.memcpy(d_q, h_q_f4, sizeof(Float4) * p.n_atoms);
        q_.memcpy(d_se, h_sig_eps, sizeof(Float2) * p.n_atoms);
        q_.memcpy(d_rp, h_sym_row_ptr, sizeof(std::uint32_t) * (p.n_atoms + 1));
        if (p.n_edges > 0) {
            q_.memcpy(d_ci, h_sym_col_idx, sizeof(std::uint32_t) * p.n_edges);
            q_.memcpy(d_wlj, h_sym_wlj, sizeof(float) * p.n_edges);
            q_.memcpy(d_wcoul, h_sym_wcoul, sizeof(float) * p.n_edges);
        }
        q_.memcpy(d_tbl, h_dsf_table, sizeof(Float4) * 4096);
        q_.wait();

        const std::size_t wg = profile_.tuned_wg_1d;
        const std::size_t global_n = ((std::size_t(p.n_atoms) + wg - 1) / wg) * wg;

        q_.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_n), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                const std::uint32_t i = static_cast<std::uint32_t>(item.get_global_id(0));
                if (i >= p.n_atoms) return;

                const Float4 qi4 = d_q[i];
                const Float2 sei = d_se[i];
                const std::uint32_t e0 = d_rp[i];
                const std::uint32_t e1 = d_rp[i + 1];

                float fx_acc = 0.0f, fy_acc = 0.0f, fz_acc = 0.0f;
                float u_half = 0.0f;
                float vir[9] = {0.0f};
                const bool do_vir = p.need_virial && (p.box_mode != 0u);

                for (std::uint32_t e = e0; e < e1; ++e) {
                    const std::uint32_t j = d_ci[e];
                    const Float4 qj4 = d_q[j];
                    const float dx0 = qi4.x - qj4.x;
                    const float dy0 = qi4.y - qj4.y;
                    const float dz0 = qi4.z - qj4.z;

                    float dx = dx0, dy = dy0, dz = dz0;
                    float wx = 0.0f, wy = 0.0f, wz = 0.0f;
                    if (p.box_mode == 1u) {
                        dx = dx0 - p.box_L[0] * sycl::round(dx0 * p.box_invL[0]);
                        dy = dy0 - p.box_L[1] * sycl::round(dy0 * p.box_invL[1]);
                        dz = dz0 - p.box_L[2] * sycl::round(dz0 * p.box_invL[2]);
                        if (do_vir) {
                            wx = dx * p.box_invL[0];
                            wy = dy * p.box_invL[1];
                            wz = dz * p.box_invL[2];
                        }
                    } else if (p.box_mode == 2u) {
                        const float* H = p.box_H.data();
                        const float* Hinv = p.box_Hinv.data();
                        const float f0 = Hinv[0] * dx0 + Hinv[3] * dy0 + Hinv[6] * dz0;
                        const float f1 = Hinv[1] * dx0 + Hinv[4] * dy0 + Hinv[7] * dz0;
                        const float f2 = Hinv[2] * dx0 + Hinv[5] * dy0 + Hinv[8] * dz0;
                        const float n0_0 = sycl::round(f0);
                        const float n0_1 = sycl::round(f1);
                        const float n0_2 = sycl::round(f2);
                        float bx = dx0 - (H[0] * n0_0 + H[3] * n0_1 + H[6] * n0_2);
                        float by = dy0 - (H[1] * n0_0 + H[4] * n0_1 + H[7] * n0_2);
                        float bz = dz0 - (H[2] * n0_0 + H[5] * n0_1 + H[8] * n0_2);
                        float best2 = bx * bx + by * by + bz * bz;
                        for (int dc = -1; dc <= 1; ++dc) {
                            for (int db = -1; db <= 1; ++db) {
                                for (int da = -1; da <= 1; ++da) {
                                    if (da == 0 && db == 0 && dc == 0) continue;
                                    const float na = n0_0 + float(da);
                                    const float nb = n0_1 + float(db);
                                    const float nc = n0_2 + float(dc);
                                    const float rx = dx0 - (H[0] * na + H[3] * nb + H[6] * nc);
                                    const float ry = dy0 - (H[1] * na + H[4] * nb + H[7] * nc);
                                    const float rz = dz0 - (H[2] * na + H[5] * nb + H[8] * nc);
                                    const float r2c = rx * rx + ry * ry + rz * rz;
                                    if (r2c < best2) {
                                        best2 = r2c;
                                        bx = rx; by = ry; bz = rz;
                                    }
                                }
                            }
                        }
                        dx = bx; dy = by; dz = bz;
                        if (do_vir) {
                            wx = Hinv[0] * dx + Hinv[3] * dy + Hinv[6] * dz;
                            wy = Hinv[1] * dx + Hinv[4] * dy + Hinv[7] * dz;
                            wz = Hinv[2] * dx + Hinv[5] * dy + Hinv[8] * dz;
                        }
                    }

                    const float r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= p.rc2 || r2 < 1e-12f) continue;

                    const float dist = sycl::sqrt(r2);
                    const float inv_r = 1.0f / dist;
                    const float inv_r2 = 1.0f / r2;
                    float u_pair = 0.0f;
                    float c_total = 0.0f;

                    const float w_lj = d_wlj[e];
                    if (p.do_lj && w_lj != 0.0f) {
                        const Float2 sej = d_se[j];
                        const float eij = sei.y * sej.y;
                        const float sij = sei.x + sej.x;
                        if (eij > 0.0f && sij > 0.0f) {
                            const float sij2 = sij * sij;
                            const float s2 = sij2 * inv_r2;
                            const float s6 = s2 * s2 * s2;
                            const float s12 = s6 * s6;
                            const float s2c = sij2 * p.inv_rc2;
                            const float s6c = s2c * s2c * s2c;
                            const float s12c = s6c * s6c;
                            float u_lj = 4.0f * eij * ((s12 - s6) - (s12c - s6c));
                            float c_lj = 24.0f * eij * (2.0f * s12 - s6) * inv_r2;
                            if (p.use_lfs) {
                                const float duc = 24.0f * eij * (s6c - 2.0f * s12c) * p.inv_rc;
                                u_lj -= duc * (dist - p.rcut);
                                c_lj += duc * inv_r;
                            }
                            u_pair += w_lj * u_lj;
                            c_total += w_lj * c_lj;
                        }
                    }

                    const float w_coul = d_wcoul[e];
                    if (p.do_coul && w_coul != 0.0f) {
                        const float qij = qi4.w * qj4.w;
                        if (qij != 0.0f) {
                            const float kq = w_coul * p.coulomb_prefactor * qij;
                            int k = static_cast<int>(r2 * p.inv_du);
                            k = sycl::clamp(k, 0, 4095);
                            const float dt = r2 - (float(k) + 0.5f) * p.du;
                            const Float4 b = d_tbl[k];
                            const float phi0 = b.x + dt * (b.y + dt * (b.z + dt * b.w));
                            const float dphi0 = b.y + dt * (2.0f * b.z + dt * 3.0f * b.w);
                            u_pair += kq * ((inv_r - phi0) - p.Brc + p.Arc * dist);
                            c_total += kq * (inv_r2 * inv_r + 2.0f * dphi0 - p.Arc * inv_r);
                        }
                    }

                    u_half += 0.5f * u_pair;
                    const float fx = c_total * dx;
                    const float fy = c_total * dy;
                    const float fz = c_total * dz;
                    fx_acc += fx;
                    fy_acc += fy;
                    fz_acc += fz;

                    if (do_vir) {
                        const float gx = -0.5f * fx;
                        const float gy = -0.5f * fy;
                        const float gz = -0.5f * fz;
                        vir[0] += gx * wx; vir[1] += gy * wx; vir[2] += gz * wx;
                        vir[3] += gx * wy; vir[4] += gy * wy; vir[5] += gz * wy;
                        vir[6] += gx * wz; vir[7] += gy * wz; vir[8] += gz * wz;
                    }
                }

                d_f[i] = Float4{fx_acc, fy_acc, fz_acc, 0.0f};
                d_uh[i] = u_half;
                if (p.need_virial) {
                    const std::size_t vb = 9 * std::size_t(i);
                    for (int m = 0; m < 9; ++m) d_vir[vb + m] = vir[m];
                }
            });

        q_.wait();
        q_.memcpy(h_out_forces, d_f, sizeof(Float4) * p.n_atoms);
        q_.memcpy(h_out_u_half, d_uh, sizeof(float) * p.n_atoms);
        if (p.need_virial && h_out_vir9) {
            q_.memcpy(h_out_vir9, d_vir, sizeof(float) * 9 * p.n_atoms);
        }
        q_.wait();

        free_usm(d_q); free_usm(d_se); free_usm(d_rp); free_usm(d_ci);
        free_usm(d_wlj); free_usm(d_wcoul); free_usm(d_tbl);
        free_usm(d_f); free_usm(d_uh); free_usm(d_vir);
    }

    // -------------------------------------------------------------------------
    // Kernel 2: Dual-Rail Bit-Sliced POPCNT Ternary Matrix-Vector Product
    // -------------------------------------------------------------------------
    void ternary_matvec_dualrail(
        std::size_t M,
        std::size_t K_words,
        const std::uint64_t* h_W_P,
        const std::uint64_t* h_W_N,
        const float* h_row_scales,
        const std::uint64_t* h_x_P,
        const std::uint64_t* h_x_N,
        float x_scale,
        float* h_y_out
    ) {
        if (M == 0) return;
        const std::size_t w_count = std::max<std::size_t>(1, M * K_words);
        const std::size_t k_count = std::max<std::size_t>(1, K_words);
        const bool use_shared = !profile_.supports_usm_device && profile_.supports_usm_shared;

        auto* d_WP = alloc_usm<std::uint64_t>(w_count, use_shared);
        auto* d_WN = alloc_usm<std::uint64_t>(w_count, use_shared);
        auto* d_xP = alloc_usm<std::uint64_t>(k_count, use_shared);
        auto* d_xN = alloc_usm<std::uint64_t>(k_count, use_shared);
        auto* d_rs = h_row_scales ? alloc_usm<float>(M, use_shared) : nullptr;
        auto* d_y = alloc_usm<float>(M, use_shared);

        if (K_words > 0) {
            q_.memcpy(d_WP, h_W_P, sizeof(std::uint64_t) * w_count);
            q_.memcpy(d_WN, h_W_N, sizeof(std::uint64_t) * w_count);
            q_.memcpy(d_xP, h_x_P, sizeof(std::uint64_t) * k_count);
            q_.memcpy(d_xN, h_x_N, sizeof(std::uint64_t) * k_count);
        }
        if (d_rs) q_.memcpy(d_rs, h_row_scales, sizeof(float) * M);
        q_.wait();

        const std::size_t wg = profile_.tuned_wg_1d;
        const std::size_t global_m = ((M + wg - 1) / wg) * wg;

        q_.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_m), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                const std::size_t m = item.get_global_id(0);
                if (m >= M) return;

                std::int64_t acc = 0;
                for (std::size_t k = 0; k < K_words; ++k) {
                    const std::size_t idx = m + M * k;
                    const std::uint64_t wp = d_WP[idx];
                    const std::uint64_t wn = d_WN[idx];
                    const std::uint64_t xp = d_xP[k];
                    const std::uint64_t xn = d_xN[k];
                    const std::uint64_t p_prod = (wp & xp) | (wn & xn);
                    const std::uint64_t n_prod = (wp & xn) | (wn & xp);
                    acc += static_cast<std::int64_t>(sycl::popcount(p_prod)) -
                           static_cast<std::int64_t>(sycl::popcount(n_prod));
                }
                const float rs = d_rs ? d_rs[m] : 1.0f;
                d_y[m] = rs * x_scale * static_cast<float>(acc);
            });

        q_.wait();
        q_.memcpy(h_y_out, d_y, sizeof(float) * M).wait();

        free_usm(d_WP); free_usm(d_WN);
        free_usm(d_xP); free_usm(d_xN);
        free_usm(d_rs); free_usm(d_y);
    }

    // -------------------------------------------------------------------------
    // Kernel 3: Periodic 27-Point Balanced Ternary MSM Stencil on Ml^3 Grid
    // -------------------------------------------------------------------------
    void msm_stencil27(
        std::size_t Ml,
        const float* h_q_in,
        const float* h_w27,
        bool accumulate,
        float* h_phi_out
    ) {
        if (Ml == 0) return;
        const std::size_t n_grid = Ml * Ml * Ml;
        const bool use_shared = !profile_.supports_usm_device && profile_.supports_usm_shared;

        auto* d_qin = alloc_usm<float>(n_grid, use_shared);
        auto* d_w27 = alloc_usm<float>(27, use_shared);
        auto* d_phi = alloc_usm<float>(n_grid, use_shared);

        q_.memcpy(d_qin, h_q_in, sizeof(float) * n_grid);
        q_.memcpy(d_w27, h_w27, sizeof(float) * 27);
        if (accumulate) {
            q_.memcpy(d_phi, h_phi_out, sizeof(float) * n_grid);
        }
        q_.wait();

        const auto& wg3 = profile_.tuned_wg_3d;
        const sycl::range<3> local_r(wg3[0], wg3[1], wg3[2]);
        const sycl::range<3> global_r(
            ((Ml + wg3[0] - 1) / wg3[0]) * wg3[0],
            ((Ml + wg3[1] - 1) / wg3[1]) * wg3[1],
            ((Ml + wg3[2] - 1) / wg3[2]) * wg3[2]
        );

        q_.parallel_for(
            sycl::nd_range<3>(global_r, local_r),
            [=](sycl::nd_item<3> item) {
                const std::size_t gx = item.get_global_id(0);
                const std::size_t gy = item.get_global_id(1);
                const std::size_t gz = item.get_global_id(2);
                if (gx >= Ml || gy >= Ml || gz >= Ml) return;

                const std::size_t mx[3] = {(gx + Ml - 1) % Ml, gx, (gx + 1) % Ml};
                const std::size_t my[3] = {(gy + Ml - 1) % Ml, gy, (gy + 1) % Ml};
                const std::size_t mz[3] = {(gz + Ml - 1) % Ml, gz, (gz + 1) % Ml};
                const std::size_t s2 = Ml;
                const std::size_t s3 = Ml * Ml;

                float acc = 0.0f;
                for (std::size_t dz = 0; dz < 3; ++dz) {
                    const std::size_t bz = mz[dz] * s3;
                    const std::size_t wz = dz * 9;
                    for (std::size_t dy = 0; dy < 3; ++dy) {
                        const std::size_t byz = bz + my[dy] * s2;
                        const std::size_t wyz = wz + dy * 3;
                        acc += d_w27[wyz + 0] * d_qin[byz + mx[0]];
                        acc += d_w27[wyz + 1] * d_qin[byz + mx[1]];
                        acc += d_w27[wyz + 2] * d_qin[byz + mx[2]];
                    }
                }
                const std::size_t out_idx = gx + s2 * gy + s3 * gz;
                if (accumulate) {
                    d_phi[out_idx] += acc;
                } else {
                    d_phi[out_idx] = acc;
                }
            });

        q_.wait();
        q_.memcpy(h_phi_out, d_phi, sizeof(float) * n_grid).wait();

        free_usm(d_qin);
        free_usm(d_w27);
        free_usm(d_phi);
    }

private:
    void probe_device() {
        const auto dev = q_.get_device();
        profile_.device_name = dev.get_info<sycl::info::device::name>();
        profile_.max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
        profile_.local_mem_size = dev.get_info<sycl::info::device::local_mem_size>();
        profile_.global_mem_size = dev.get_info<sycl::info::device::global_mem_size>();

        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        if (!sg_sizes.empty()) {
            profile_.preferred_sub_group_size = *std::min_element(sg_sizes.begin(), sg_sizes.end());
        }
        profile_.supports_usm_device = dev.has(sycl::aspect::usm_device_allocations);
        profile_.supports_usm_shared = dev.has(sycl::aspect::usm_shared_allocations);

        // Auto-tune nd_range local sizes against vendor hardware limits
        std::size_t wg1d = std::clamp<std::size_t>(profile_.preferred_sub_group_size * 2, 32, 256);
        wg1d = std::min(wg1d, profile_.max_work_group_size);
        profile_.tuned_wg_1d = std::max<std::size_t>(1, wg1d);

        if (profile_.max_work_group_size >= 64) {
            profile_.tuned_wg_3d = {4, 4, 4};
        } else if (profile_.max_work_group_size >= 8) {
            profile_.tuned_wg_3d = {2, 2, 2};
        } else {
            profile_.tuned_wg_3d = {1, 1, 1};
        }
    }

    sycl::queue q_;
    SyclDeviceProfile profile_{};
};

#else // !TRIUM_HAS_SYCL — Host C++20 Fallback Stub for Non-SYCL Toolchains

class TriumSyclEngine {
public:
    TriumSyclEngine() = default;
    static constexpr bool is_compiled_with_sycl() noexcept { return false; }
    const SyclDeviceProfile& profile() const noexcept { return profile_; }

    void eval_lj_dsf_csr(
        const LjDsfCsrParams&, const Float4*, const Float2*,
        const std::uint32_t*, const std::uint32_t*, const float*, const float*,
        const Float4*, Float4*, float*, float*
    ) {}

    void ternary_matvec_dualrail(
        std::size_t, std::size_t, const std::uint64_t*, const std::uint64_t*,
        const float*, const std::uint64_t*, const std::uint64_t*, float, float*
    ) {}

    void msm_stencil27(std::size_t, const float*, const float*, bool, float*) {}

private:
    SyclDeviceProfile profile_{};
};

#endif // TRIUM_HAS_SYCL

} // namespace trium::gpu
