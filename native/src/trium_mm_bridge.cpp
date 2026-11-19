/**
 * @file trium_mm_bridge.cpp
 * @brief Native C++20 / ARM64 NEON Molecular Mechanics Kernel Bridge for Balanced Ternary & Biology MM
 * @author Biology-MM Contributors
 *
 * Implements high-throughput C ABI bridge functions connecting Trium-XBT balanced ternary
 * primitives to Julia / Python molecular mechanics engines:
 *  1. xbt_mm_version() -> int64_t
 *  2. xbt_mm_abi_version() -> int64_t
 *  3. xbt_exp_radix3(double x) -> double
 *  4. xbt_tanh_radix3(double x) -> double
 *  5. xbt_rsqrt_fast(double x) -> double
 *  6. xbt_erfc_fast(double x) -> double
 *  7. xbt_csa_accumulate_array(...) -> int
 *  8. xbt_symplectic_kdk_step_int64(...) -> int
 *  9. xbt_eval_lj_dsf_pairs(...) -> double
 * 10. xbt_eval_lj_dsf_pairs_mt(...) -> double
 * 11. xbt_eval_lj_dsf_tiles_mt(...) -> double
 * 12. xbt_schedule_disjoint_tile_orbits(...) -> int32_t
 * 13. xbt_ternary_matvec_dualrail(...) -> void
 * 14. xbt_ternary_matvec_multiplane(...) -> void
 * 15. xbt_verlet_plan_create(...) -> void*
 * 16. xbt_verlet_plan_eval_mt(...) -> double
 * 17. xbt_verlet_plan_destroy(void*) -> void
 * 18. xbt_verlet_plan_num_pairs(void*) -> size_t
 *
 * Zero external Kokkos/OpenMP runtime dependency; compiles cleanly with standard C++20.
 */

#include "ternary_xbt_foundation.hpp"
#include "ternary_carry_free.hpp"
#include "ternary_nonlinear_engine.hpp"
#include "symplectic_ternary_engine.hpp"
#include "ternary_dense_algebra.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(_WIN32)
#define XBT_EXPORT extern "C" __declspec(dllexport)
#else
#define XBT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr double SQRT_PI = 1.772453850905516027298167483341145182797549456122387;
constexpr double FIXED_POINT_SCALE = 1099511627776.0; // 2^40 for deterministic Int64 force accumulation

inline int resolve_thread_count(int n_threads) {
    if (n_threads <= 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        const int t = hw > 0 ? static_cast<int>(hw) : 1;
        return std::min(std::max(1, t), 8);
    }
    return n_threads;
}

inline void zero_output_buffers(
    size_t n_atoms,
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9
) {
    if (out_forces && n_atoms > 0) {
        std::memset(out_forces, 0, 3 * n_atoms * sizeof(double));
    }
    if (out_forces_fixed && n_atoms > 0) {
        std::memset(out_forces_fixed, 0, 3 * n_atoms * sizeof(int64_t));
    }
    if (out_virial9) {
        std::memset(out_virial9, 0, 9 * sizeof(double));
    }
}

// =============================================================================
// Persistent Worker Pool (0 thread ctor/dtor and 0 heap allocations per call)
// =============================================================================

struct alignas(64) WorkerScratch {
    double u = 0.0;
    double vir[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> f;
    std::vector<int64_t> fx;
};

class XbtPersistentPool {
public:
    static constexpr int MAX_TOTAL_THREADS = 8;

    static XbtPersistentPool& instance() {
        static XbtPersistentPool pool;
        return pool;
    }

    int max_threads() const noexcept {
        return static_cast<int>(max_workers_) + 1;
    }

    WorkerScratch& scratch(int worker_id) noexcept {
        return scratch_[static_cast<size_t>(worker_id)];
    }

    void prepare_scratch(int T, size_t f_len, size_t fx_len, bool need_vir) {
        for (int t = 0; t < T; ++t) {
            WorkerScratch& s = scratch_[static_cast<size_t>(t)];
            s.u = 0.0;
            if (need_vir) {
                std::memset(s.vir, 0, sizeof(s.vir));
            }
            if (f_len > 0) {
                if (s.f.size() < f_len) {
                    s.f.resize(f_len);
                }
                std::fill_n(s.f.data(), f_len, 0.0);
            }
            if (fx_len > 0) {
                if (s.fx.size() < fx_len) {
                    s.fx.resize(fx_len);
                }
                std::fill_n(s.fx.data(), fx_len, int64_t{0});
            }
        }
    }

    // Dispatches task(worker_id, n_threads) across worker_id in [0, T).
    // Calling thread runs worker_id == 0. Callers passing a stack-context lambda
    // [&ctx] fit inside std::function SBO with zero heap allocations.
    void run(int n_threads, const std::function<void(int worker_id, int n_threads)>& task) {
        const int T = std::min(std::max(1, n_threads), max_threads());
        if (T <= 1 || max_workers_ == 0) {
            task(0, 1);
            return;
        }

        std::lock_guard<std::mutex> dispatch_lock(dispatch_mtx_);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            task_ = task;
            active_threads_ = T;
            pending_workers_ = T - 1;
            ++epoch_;
        }
        cv_work_.notify_all();

        task_(0, T);

        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_done_.wait(lk, [this]() { return pending_workers_ == 0; });
            task_ = nullptr;
        }
    }

    ~XbtPersistentPool() {
        shutdown();
    }

private:
    XbtPersistentPool() {
        const unsigned int hw = std::thread::hardware_concurrency();
        const unsigned int total = std::min(std::max(1u, hw), static_cast<unsigned int>(MAX_TOTAL_THREADS));
        max_workers_ = total - 1u;
        workers_.reserve(max_workers_);
        for (unsigned int w = 1; w <= max_workers_; ++w) {
            workers_.emplace_back([this, w]() { worker_loop(static_cast<int>(w)); });
        }
    }

    XbtPersistentPool(const XbtPersistentPool&) = delete;
    XbtPersistentPool& operator=(const XbtPersistentPool&) = delete;

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (stop_) return;
            stop_ = true;
            ++epoch_;
        }
        cv_work_.notify_all();
        for (auto& th : workers_) {
            if (th.joinable()) {
                th.join();
            }
        }
        workers_.clear();
    }

    void worker_loop(int worker_id) {
        uint64_t seen_epoch = 0;
        for (;;) {
            int T = 1;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_work_.wait(lk, [this, seen_epoch]() {
                    return stop_ || epoch_ != seen_epoch;
                });
                if (stop_) return;
                seen_epoch = epoch_;
                T = active_threads_;
            }

            if (worker_id < T) {
                task_(worker_id, T);
                std::lock_guard<std::mutex> lk(mtx_);
                if (--pending_workers_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    unsigned int max_workers_ = 0;
    std::array<WorkerScratch, MAX_TOTAL_THREADS> scratch_{};
    std::vector<std::thread> workers_;
    std::mutex dispatch_mtx_;
    std::mutex mtx_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::function<void(int, int)> task_;
    uint64_t epoch_ = 0;
    int active_threads_ = 1;
    int pending_workers_ = 0;
    bool stop_ = false;
};

// =============================================================================
// 4096-Bin Midpoint Degree-4 Analytic DSF Taylor Table (160 KB L2-resident)
// =============================================================================

struct DsfAnalyticTable {
    static constexpr int N_TAB = 4096;

    struct Bin {
        double c0, c1, c2, c3, c4;
    };
    struct FBin {
        float c0, c1, c2, c3, c4;
    };

    double rcut = 0.0;
    double alpha = 0.0;
    double rc2 = 0.0;
    double du = 0.0;
    double inv_du = 0.0;
    std::array<Bin, N_TAB> bins{};
    std::array<FBin, N_TAB> fbins{};

    void init(double rcut_in, double alpha_in) {
        rcut = rcut_in;
        alpha = alpha_in;
        rc2 = rcut * rcut;
        du = rc2 / static_cast<double>(N_TAB);
        inv_du = static_cast<double>(N_TAB) / rc2;

        const double two_a_sqrtpi = 2.0 * alpha / SQRT_PI;
        const double a2 = alpha * alpha;
        const double a4 = a2 * a2;
        const double a6 = a4 * a2;

        for (int k = 0; k < N_TAB; ++k) {
            const double u = (static_cast<double>(k) + 0.5) * du;
            const double z2 = a2 * u;

            double f0, f1, f2, f3, f4;
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
                f4 = deriv_series(4);
            } else {
                const double r = std::sqrt(u);
                const double g = two_a_sqrtpi * std::exp(-z2);
                const double inv_2u = 0.5 / u;
                f0 = std::erf(alpha * r) / r;
                f1 = (g - f0) * inv_2u;
                f2 = (-a2 * g - 3.0 * f1) * inv_2u;
                f3 = (a4 * g - 5.0 * f2) * inv_2u;
                f4 = (-a6 * g - 7.0 * f3) * inv_2u;
            }

            const double c0 = f0;
            const double c1 = f1;
            const double c2 = f2 * 0.5;
            const double c3 = f3 * (1.0 / 6.0);
            const double c4 = f4 * (1.0 / 24.0);

            bins[static_cast<size_t>(k)] = Bin{c0, c1, c2, c3, c4};
            fbins[static_cast<size_t>(k)] = FBin{
                static_cast<float>(c0),
                static_cast<float>(c1),
                static_cast<float>(c2),
                static_cast<float>(c3),
                static_cast<float>(c4)
            };
        }
    }

    inline void lookup(double r2, double& phi0, double& dphi0) const noexcept {
        int k = static_cast<int>(r2 * inv_du);
        if (k < 0) k = 0;
        if (k >= N_TAB) k = N_TAB - 1;
        const double dt = r2 - (static_cast<double>(k) + 0.5) * du;
        const Bin& b = bins[static_cast<size_t>(k)];
        phi0 = b.c0 + dt * (b.c1 + dt * (b.c2 + dt * (b.c3 + dt * b.c4)));
        dphi0 = b.c1 + dt * (2.0 * b.c2 + dt * (3.0 * b.c3 + dt * 4.0 * b.c4));
    }
};

class DsfTableCache {
public:
    static std::shared_ptr<const DsfAnalyticTable> get(double rcut, double alpha_dsf) {
        static DsfTableCache cache;
        return cache.get_impl(rcut, alpha_dsf);
    }

private:
    std::shared_ptr<const DsfAnalyticTable> get_impl(double rcut, double alpha_dsf) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (latest_ && latest_->rcut == rcut && latest_->alpha == alpha_dsf) {
            return latest_;
        }
        auto tbl = std::make_shared<DsfAnalyticTable>();
        tbl->init(rcut, alpha_dsf);
        latest_ = tbl;
        return tbl;
    }

    std::mutex mtx_;
    std::shared_ptr<const DsfAnalyticTable> latest_;
};

struct DsfParams {
    double rc2;
    double rcut;
    double inv_rc;
    double inv_rc2;
    double alpha_dsf;
    double coulomb_prefactor;
    double two_alpha_sqrt_pi;
    double Arc;
    double Brc;
    double L0, L1, L2;
    double inv_L0, inv_L1, inv_L2;
    bool do_coul;
    bool use_lfs;
    bool use_fast_math;
    bool has_box;
    bool is_triclinic;
    const double* box_H;
    const double* box_Hinv;
    const DsfAnalyticTable* dsf_table;
};

} // anonymous namespace

XBT_EXPORT int64_t xbt_mm_version() {
    return 202611;
}

XBT_EXPORT int64_t xbt_mm_abi_version() {
    return 2;
}

XBT_EXPORT double xbt_exp_radix3(double x) {
    return bt::nonlinear::xbt_exp_fast(x);
}

XBT_EXPORT double xbt_tanh_radix3(double x) {
    return bt::nonlinear::xbt_tanh_fast(x);
}

XBT_EXPORT double xbt_rsqrt_fast(double x) {
    return bt::nonlinear::xbt_rsqrt_householder(x);
}

/**
 * @brief High-speed erfc(x) using Abramowitz & Stegun (7.1.26) rational polynomial
 * approximation combined with Radix-3 range-reduced exponential bt::nonlinear::xbt_exp_fast(-x*x).
 * Accuracy: absolute error < 1.4e-7 over the full real axis, erfc(0.0) == 1.0, erfc(-x) = 2 - erfc(x),
 * and exact 0.0 for large positive inputs (x >= 28.0).
 */
XBT_EXPORT double xbt_erfc_fast(double x) {
    if (std::isnan(x)) return x;
    if (x == 0.0) return 1.0;
    if (x < 0.0) return 2.0 - xbt_erfc_fast(-x);
    if (x >= 28.0) return 0.0;

    // Abramowitz & Stegun 7.1.26 minimax rational polynomial (error < 1.5e-7)
    constexpr double p  = 0.3275911;
    constexpr double a1 = 0.254829592;
    constexpr double a2 = -0.284496736;
    constexpr double a3 = 1.421413741;
    constexpr double a4 = -1.453152027;
    constexpr double a5 = 1.061405429;

    const double t = 1.0 / (1.0 + p * x);
    const double poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    return poly * bt::nonlinear::xbt_exp_fast(-x * x);
}

/**
 * @brief Accumulates an array of 64-bit integer terms into a Balanced Ternary Carry-Save (S, C) pair,
 * and resolves into a canonical int64_t sum using bounded ripple / Kogge-Stone assimilation.
 */
XBT_EXPORT int xbt_csa_accumulate_array(
    const int64_t* terms,
    size_t n_terms,
    uint64_t* out_S_P,
    uint64_t* out_S_N,
    uint64_t* out_C_P,
    uint64_t* out_C_N,
    int64_t* out_resolved
) {
    bt::TritWord64 S{0, 0};
    bt::TritWord64 C{0, 0};
    bool ovf = false;

    if (terms && n_terms > 0) {
        for (size_t k = 0; k < n_terms; ++k) {
            bt::TritWord64 X = bt::Codec::from_int64(terms[k], 64);
            bt::CarryFreeStep step = bt::carry_free_add(S, C, X);
            S = step.S;
            C = step.C;
            if (step.overflow) ovf = true;
        }
    }

    bool resolve_ovf = false;
    bt::TritWord64 resolved = bt::resolve_bounded_ripple(S, C, &resolve_ovf);
    if (resolve_ovf) ovf = true;

    constexpr uint64_t high_mask = ~((1ULL << 39) - 1ULL);
    if ((resolved.P & high_mask) != 0 || (resolved.N & high_mask) != 0) {
        ovf = true;
    }

    if (out_S_P) *out_S_P = S.P;
    if (out_S_N) *out_S_N = S.N;
    if (out_C_P) *out_C_P = C.P;
    if (out_C_N) *out_C_N = C.N;
    if (out_resolved) {
        *out_resolved = bt::Codec::to_int64(resolved, 39);
    }

    return ovf ? 1 : 0;
}

/**
 * @brief Symplectic time-reversible leapfrog phase step across n_dof coordinates in discrete PhaseSpace (Q, P).
 */
XBT_EXPORT int xbt_symplectic_kdk_step_int64(
    int64_t* q_int,
    int64_t* p_int,
    size_t n_dof,
    int direction
) {
    if (!q_int || !p_int) return -1;

    for (size_t i = 0; i < n_dof; ++i) {
        bt::symplectic::PhaseState st(
            bt::Codec::from_int64(q_int[i], 39),
            bt::Codec::from_int64(p_int[i], 39)
        );
        bt::symplectic::PhaseState out_st = (direction >= 0)
            ? bt::symplectic::SymplecticLeapfrogEngine::forward_step(st, st.P)
            : bt::symplectic::SymplecticLeapfrogEngine::reverse_step(st, st.P);

        q_int[i] = bt::Codec::to_int64(out_st.Q, 39);
        p_int[i] = bt::Codec::to_int64(out_st.P, 39);
    }
    return 0;
}

namespace {

inline DsfParams init_dsf_params(
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    const double* charge,
    double rcut,
    double alpha_dsf,
    double coulomb_prefactor,
    int use_lfs,
    int use_fast_math,
    const DsfAnalyticTable* dsf_table
) {
    DsfParams p{};
    p.rcut = rcut;
    p.rc2 = rcut * rcut;
    p.inv_rc = (rcut > 0.0) ? (1.0 / rcut) : 0.0;
    p.inv_rc2 = (p.rc2 > 0.0) ? (1.0 / p.rc2) : 0.0;
    p.alpha_dsf = alpha_dsf;
    p.coulomb_prefactor = coulomb_prefactor;
    p.use_lfs = (use_lfs != 0);
    p.use_fast_math = (use_fast_math != 0);
    p.has_box = (box_H != nullptr);
    p.is_triclinic = (is_triclinic != 0);
    p.box_H = box_H;
    p.box_Hinv = box_Hinv;
    p.do_coul = (charge != nullptr && coulomb_prefactor != 0.0 && alpha_dsf > 0.0);
    p.dsf_table = dsf_table;

    if (p.do_coul) {
        p.two_alpha_sqrt_pi = 2.0 * alpha_dsf / SQRT_PI;
        const double z_rc = alpha_dsf * rcut;
        const double erfc_rc = p.use_fast_math ? xbt_erfc_fast(z_rc) : std::erfc(z_rc);
        const double exp_rc = p.use_fast_math
            ? bt::nonlinear::xbt_exp_fast(-alpha_dsf * alpha_dsf * p.rc2)
            : std::exp(-alpha_dsf * alpha_dsf * p.rc2);
        p.Arc = erfc_rc * p.inv_rc2 + p.two_alpha_sqrt_pi * exp_rc * p.inv_rc;
        p.Brc = erfc_rc * p.inv_rc + p.Arc * rcut;
    }

    if (box_H && !p.is_triclinic) {
        p.L0 = box_H[0];
        p.L1 = box_H[4];
        p.L2 = box_H[8];
        p.inv_L0 = (p.L0 != 0.0) ? (1.0 / p.L0) : 0.0;
        p.inv_L1 = (p.L1 != 0.0) ? (1.0 / p.L1) : 0.0;
        p.inv_L2 = (p.L2 != 0.0) ? (1.0 / p.L2) : 0.0;
    }
    return p;
}

inline void apply_mic(
    bool has_box,
    bool is_triclinic,
    double L0, double L1, double L2,
    double inv_L0, double inv_L1, double inv_L2,
    const double* H,
    const double* Hinv,
    double dx0, double dy0, double dz0,
    bool need_virial,
    double& dx, double& dy, double& dz,
    double& wx, double& wy, double& wz
) noexcept {
    dx = dx0;
    dy = dy0;
    dz = dz0;
    wx = 0.0;
    wy = 0.0;
    wz = 0.0;

    if (!has_box) return;

    if (!is_triclinic) {
        dx = dx0 - L0 * std::nearbyint(dx0 * inv_L0);
        dy = dy0 - L1 * std::nearbyint(dy0 * inv_L1);
        dz = dz0 - L2 * std::nearbyint(dz0 * inv_L2);
        if (need_virial) {
            wx = dx * inv_L0;
            wy = dy * inv_L1;
            wz = dz * inv_L2;
        }
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
        if (need_virial) {
            wx = Hinv[0] * dx + Hinv[3] * dy + Hinv[6] * dz;
            wy = Hinv[1] * dx + Hinv[4] * dy + Hinv[7] * dz;
            wz = Hinv[2] * dx + Hinv[5] * dy + Hinv[8] * dz;
        }
    }
}

inline void eval_pair_post_mic(
    const DsfParams& p,
    double dx, double dy, double dz, double r2,
    bool do_lj, double sig_i, double sig_j, double eps_i, double eps_j,
    double chg_i, double chg_j,
    double w_lj, double w_coul,
    double& out_u,
    double& out_fx, double& out_fy, double& out_fz
) noexcept {
    double r = 0.0;
    double inv_r = 0.0;
    if (p.use_fast_math) {
        inv_r = bt::nonlinear::xbt_rsqrt_householder(r2);
        r = r2 * inv_r;
    } else {
        r = std::sqrt(r2);
        inv_r = 1.0 / r;
    }

    double u_pair_total = 0.0;
    double c_total = 0.0;

    // --- Lennard-Jones (12-6) Contribution ---
    if (do_lj && w_lj != 0.0) {
        if (eps_i > 0.0 && eps_j > 0.0 && (sig_i > 0.0 || sig_j > 0.0)) {
            const double eij = std::sqrt(eps_i * eps_j);
            const double sij = 0.5 * (sig_i + sig_j);
            const double sij2 = sij * sij;

            const double inv_r2 = inv_r * inv_r;
            const double s2 = sij2 * inv_r2;
            const double s6 = s2 * s2 * s2;
            const double s12 = s6 * s6;

            const double s2c = sij2 * p.inv_rc2;
            const double s6c = s2c * s2c * s2c;
            const double s12c = s6c * s6c;

            double u_lj = 4.0 * eij * ((s12 - s6) - (s12c - s6c));
            double dudr_lj = 24.0 * eij * (s6 - 2.0 * s12) * inv_r;

            if (p.use_lfs) {
                const double duc = 24.0 * eij * (s6c - 2.0 * s12c) * p.inv_rc;
                u_lj -= duc * (r - p.rcut);
                dudr_lj -= duc;
            }

            u_pair_total += w_lj * u_lj;
            c_total += -(w_lj * dudr_lj) * inv_r;
        }
    }

    // --- DSF Coulomb Electrostatic Contribution ---
    if (p.do_coul && w_coul != 0.0) {
        const double qiqj = chg_i * chg_j;
        if (qiqj != 0.0) {
            const double kq = w_coul * p.coulomb_prefactor * qiqj;
            if (!p.use_fast_math && p.dsf_table != nullptr) {
                double phi0 = 0.0, dphi0 = 0.0;
                p.dsf_table->lookup(r2, phi0, dphi0);
                const double erfc_over_r = inv_r - phi0;
                const double inv_r3 = inv_r * inv_r * inv_r;
                u_pair_total += kq * (erfc_over_r - p.Brc + p.Arc * r);
                c_total += kq * (inv_r3 + 2.0 * dphi0 - p.Arc * inv_r);
            } else {
                const double z = p.alpha_dsf * r;
                const double er = p.use_fast_math ? xbt_erfc_fast(z) : std::erfc(z);
                const double exp_z = p.use_fast_math ? bt::nonlinear::xbt_exp_fast(-z * z) : std::exp(-z * z);
                const double inv_r2 = inv_r * inv_r;
                u_pair_total += kq * (er * inv_r - p.Brc + p.Arc * r);
                c_total += kq * (er * inv_r2 + p.two_alpha_sqrt_pi * exp_z * inv_r - p.Arc) * inv_r;
            }
        }
    }

    out_u = u_pair_total;
    out_fx = c_total * dx;
    out_fy = c_total * dy;
    out_fz = c_total * dz;
}

inline void accum_virial9(
    double* vir,
    double fx, double fy, double fz,
    double wx, double wy, double wz
) noexcept {
    const double gx = -fx;
    const double gy = -fy;
    const double gz = -fz;
    vir[0 + 3 * 0] += gx * wx;
    vir[1 + 3 * 0] += gy * wx;
    vir[2 + 3 * 0] += gz * wx;

    vir[0 + 3 * 1] += gx * wy;
    vir[1 + 3 * 1] += gy * wy;
    vir[2 + 3 * 1] += gz * wy;

    vir[0 + 3 * 2] += gx * wz;
    vir[1 + 3 * 2] += gy * wz;
    vir[2 + 3 * 2] += gz * wz;
}

inline double eval_pairs_range(
    const DsfParams& p,
    size_t n_atoms,
    const double* q,
    const double* sigma,
    const double* epsilon,
    const double* charge,
    size_t start_k,
    size_t end_k,
    const int32_t* pair_i,
    const int32_t* pair_j,
    const double* pair_scale_lj,
    const double* pair_scale_coul,
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9
) {
    const bool do_lj = (sigma != nullptr && epsilon != nullptr);
    const bool need_virial = (out_virial9 != nullptr && p.has_box);
    double u_loc = 0.0;
    double v[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (size_t k = start_k; k < end_k; ++k) {
        const int32_t i = pair_i[k];
        const int32_t j = pair_j[k];
        if (i < 0 || j < 0 || static_cast<size_t>(i) >= n_atoms || static_cast<size_t>(j) >= n_atoms || i == j) {
            continue;
        }

        const double w_lj = pair_scale_lj ? pair_scale_lj[k] : 1.0;
        const double w_coul = pair_scale_coul ? pair_scale_coul[k] : 1.0;
        if (w_lj == 0.0 && w_coul == 0.0) continue;

        const size_t ui = static_cast<size_t>(i);
        const size_t uj = static_cast<size_t>(j);
        const double dx0 = q[3 * ui + 0] - q[3 * uj + 0];
        const double dy0 = q[3 * ui + 1] - q[3 * uj + 1];
        const double dz0 = q[3 * ui + 2] - q[3 * uj + 2];

        double dx, dy, dz, wx, wy, wz;
        apply_mic(
            p.has_box, p.is_triclinic,
            p.L0, p.L1, p.L2, p.inv_L0, p.inv_L1, p.inv_L2,
            p.box_H, p.box_Hinv,
            dx0, dy0, dz0, need_virial,
            dx, dy, dz, wx, wy, wz
        );

        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= p.rc2 || r2 < 1e-24) continue;

        double u_pair = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
        eval_pair_post_mic(
            p, dx, dy, dz, r2,
            do_lj,
            do_lj ? sigma[ui] : 0.0, do_lj ? sigma[uj] : 0.0,
            do_lj ? epsilon[ui] : 0.0, do_lj ? epsilon[uj] : 0.0,
            charge ? charge[ui] : 0.0, charge ? charge[uj] : 0.0,
            w_lj, w_coul,
            u_pair, fx, fy, fz
        );

        u_loc += u_pair;

        if (out_forces) {
            out_forces[3 * ui + 0] += fx;
            out_forces[3 * ui + 1] += fy;
            out_forces[3 * ui + 2] += fz;
            out_forces[3 * uj + 0] -= fx;
            out_forces[3 * uj + 1] -= fy;
            out_forces[3 * uj + 2] -= fz;
        }

        if (out_forces_fixed) {
            const int64_t fx_int = std::llround(fx * FIXED_POINT_SCALE);
            const int64_t fy_int = std::llround(fy * FIXED_POINT_SCALE);
            const int64_t fz_int = std::llround(fz * FIXED_POINT_SCALE);
            out_forces_fixed[3 * ui + 0] += fx_int;
            out_forces_fixed[3 * ui + 1] += fy_int;
            out_forces_fixed[3 * ui + 2] += fz_int;
            out_forces_fixed[3 * uj + 0] -= fx_int;
            out_forces_fixed[3 * uj + 1] -= fy_int;
            out_forces_fixed[3 * uj + 2] -= fz_int;
        }

        if (need_virial) {
            accum_virial9(v, fx, fy, fz, wx, wy, wz);
        }
    }

    if (need_virial) {
        for (int idx = 0; idx < 9; ++idx) {
            out_virial9[idx] += v[idx];
        }
    }
    return u_loc;
}

inline double eval_tiles_range(
    const DsfParams& p,
    size_t n_atoms,
    const double* q,
    const double* sigma,
    const double* epsilon,
    const double* charge,
    size_t start_k,
    size_t end_k,
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
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9
) {
    const bool do_lj = (sigma != nullptr && epsilon != nullptr);
    const bool do_forces = (out_forces != nullptr);
    const bool do_fixed = (out_forces_fixed != nullptr);
    const bool need_virial = (out_virial9 != nullptr && p.has_box);
    const bool has_excl = (n_excl > 0 && excl_keys != nullptr);
    const bool has_scaled = (n_scaled > 0 && scaled_keys != nullptr);
    const bool check_topo = (has_excl || has_scaled);

    double u_loc = 0.0;
    double v[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    auto eval_ij = [&](
        size_t ui, int32_t i1,
        double qxi, double qyi, double qzi,
        double sig_i, double eps_i, double chg_i,
        int32_t b_pos,
        double& fxi, double& fyi, double& fzi,
        int64_t& fxi_fix, int64_t& fyi_fix, int64_t& fzi_fix
    ) {
        const int32_t j1 = order[b_pos - 1];
        const int32_t j = j1 - 1;
        if (j < 0 || static_cast<size_t>(j) >= n_atoms || static_cast<size_t>(j) == ui) return;

        const size_t uj = static_cast<size_t>(j);
        const double dx0 = qxi - q[3 * uj + 0];
        const double dy0 = qyi - q[3 * uj + 1];
        const double dz0 = qzi - q[3 * uj + 2];

        // Compute MIC and r2 FIRST before topology binary searches (skips 85-99% of searches)
        double dx, dy, dz, wx, wy, wz;
        apply_mic(
            p.has_box, p.is_triclinic,
            p.L0, p.L1, p.L2, p.inv_L0, p.inv_L1, p.inv_L2,
            p.box_H, p.box_Hinv,
            dx0, dy0, dz0, need_virial,
            dx, dy, dz, wx, wy, wz
        );

        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= p.rc2 || r2 < 1e-24) return;

        double w_lj = 1.0;
        double w_coul = 1.0;
        if (check_topo) {
            const uint64_t lo = static_cast<uint64_t>(i1 < j1 ? i1 : j1);
            const uint64_t hi = static_cast<uint64_t>(i1 < j1 ? j1 : i1);
            const uint64_t key = (lo << 32) | hi;
            if (has_excl && std::binary_search(excl_keys, excl_keys + n_excl, key)) {
                return;
            }
            if (has_scaled) {
                const uint64_t* it = std::lower_bound(scaled_keys, scaled_keys + n_scaled, key);
                if (it != scaled_keys + n_scaled && *it == key) {
                    const size_t pos = static_cast<size_t>(it - scaled_keys);
                    const double w = scaled_weights ? scaled_weights[pos] : lj14_scale;
                    w_lj = (w != 1.0) ? w : lj14_scale;
                    w_coul = coul14_scale;
                    if (w_lj == 0.0 && w_coul == 0.0) return;
                }
            }
        }

        double u_pair = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
        eval_pair_post_mic(
            p, dx, dy, dz, r2,
            do_lj, sig_i, do_lj ? sigma[uj] : 0.0,
            eps_i, do_lj ? epsilon[uj] : 0.0,
            chg_i, charge ? charge[uj] : 0.0,
            w_lj, w_coul,
            u_pair, fx, fy, fz
        );

        u_loc += u_pair;

        if (do_forces) {
            fxi += fx;
            fyi += fy;
            fzi += fz;
            out_forces[3 * uj + 0] -= fx;
            out_forces[3 * uj + 1] -= fy;
            out_forces[3 * uj + 2] -= fz;
        }

        if (do_fixed) {
            const int64_t fx_int = std::llround(fx * FIXED_POINT_SCALE);
            const int64_t fy_int = std::llround(fy * FIXED_POINT_SCALE);
            const int64_t fz_int = std::llround(fz * FIXED_POINT_SCALE);
            fxi_fix += fx_int;
            fyi_fix += fy_int;
            fzi_fix += fz_int;
            out_forces_fixed[3 * uj + 0] -= fx_int;
            out_forces_fixed[3 * uj + 1] -= fy_int;
            out_forces_fixed[3 * uj + 2] -= fz_int;
        }

        if (need_virial) {
            accum_virial9(v, fx, fy, fz, wx, wy, wz);
        }
    };

    for (size_t k = start_k; k < end_k; ++k) {
        const int32_t ti1 = tile_i[k];
        const int32_t tj1 = tile_j[k];
        const int32_t si1 = tile_starts[ti1 - 1];
        const int32_t ei1 = tile_ends[ti1 - 1];

        if (ti1 == tj1) {
            for (int32_t a = si1; a < ei1; ++a) {
                const int32_t i1 = order[a - 1];
                const int32_t i = i1 - 1;
                if (i < 0 || static_cast<size_t>(i) >= n_atoms) continue;
                const size_t ui = static_cast<size_t>(i);
                const double qxi = q[3 * ui + 0];
                const double qyi = q[3 * ui + 1];
                const double qzi = q[3 * ui + 2];
                const double sig_i = do_lj ? sigma[ui] : 0.0;
                const double eps_i = do_lj ? epsilon[ui] : 0.0;
                const double chg_i = charge ? charge[ui] : 0.0;
                double fxi = 0.0, fyi = 0.0, fzi = 0.0;
                int64_t fxi_fix = 0, fyi_fix = 0, fzi_fix = 0;

                for (int32_t b = a + 1; b <= ei1; ++b) {
                    eval_ij(ui, i1, qxi, qyi, qzi, sig_i, eps_i, chg_i, b,
                            fxi, fyi, fzi, fxi_fix, fyi_fix, fzi_fix);
                }

                if (do_forces) {
                    out_forces[3 * ui + 0] += fxi;
                    out_forces[3 * ui + 1] += fyi;
                    out_forces[3 * ui + 2] += fzi;
                }
                if (do_fixed) {
                    out_forces_fixed[3 * ui + 0] += fxi_fix;
                    out_forces_fixed[3 * ui + 1] += fyi_fix;
                    out_forces_fixed[3 * ui + 2] += fzi_fix;
                }
            }
        } else {
            const int32_t sj1 = tile_starts[tj1 - 1];
            const int32_t ej1 = tile_ends[tj1 - 1];
            for (int32_t a = si1; a <= ei1; ++a) {
                const int32_t i1 = order[a - 1];
                const int32_t i = i1 - 1;
                if (i < 0 || static_cast<size_t>(i) >= n_atoms) continue;
                const size_t ui = static_cast<size_t>(i);
                const double qxi = q[3 * ui + 0];
                const double qyi = q[3 * ui + 1];
                const double qzi = q[3 * ui + 2];
                const double sig_i = do_lj ? sigma[ui] : 0.0;
                const double eps_i = do_lj ? epsilon[ui] : 0.0;
                const double chg_i = charge ? charge[ui] : 0.0;
                double fxi = 0.0, fyi = 0.0, fzi = 0.0;
                int64_t fxi_fix = 0, fyi_fix = 0, fzi_fix = 0;

                for (int32_t b = sj1; b <= ej1; ++b) {
                    eval_ij(ui, i1, qxi, qyi, qzi, sig_i, eps_i, chg_i, b,
                            fxi, fyi, fzi, fxi_fix, fyi_fix, fzi_fix);
                }

                if (do_forces) {
                    out_forces[3 * ui + 0] += fxi;
                    out_forces[3 * ui + 1] += fyi;
                    out_forces[3 * ui + 2] += fzi;
                }
                if (do_fixed) {
                    out_forces_fixed[3 * ui + 0] += fxi_fix;
                    out_forces_fixed[3 * ui + 1] += fyi_fix;
                    out_forces_fixed[3 * ui + 2] += fzi_fix;
                }
            }
        }
    }

    if (need_virial) {
        for (int idx = 0; idx < 9; ++idx) {
            out_virial9[idx] += v[idx];
        }
    }
    return u_loc;
}

inline double reduce_pool_scratch(
    XbtPersistentPool& pool,
    int T,
    size_t n_atoms,
    bool has_box,
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9
) {
    double total_U = 0.0;
    const size_t f_len = 3 * n_atoms;
    for (int t = 0; t < T; ++t) {
        const WorkerScratch& acc = pool.scratch(t);
        total_U += acc.u;
        if (out_forces) {
            const double* tf = acc.f.data();
            for (size_t idx = 0; idx < f_len; ++idx) {
                out_forces[idx] += tf[idx];
            }
        }
        if (out_forces_fixed) {
            const int64_t* tfx = acc.fx.data();
            for (size_t idx = 0; idx < f_len; ++idx) {
                out_forces_fixed[idx] += tfx[idx];
            }
        }
        if (out_virial9 && has_box) {
            for (int v = 0; v < 9; ++v) {
                out_virial9[v] += acc.vir[v];
            }
        }
    }
    return total_U;
}

inline int64_t ternary_dot_strided_4way(
    size_t stride,
    size_t n_words,
    const uint64_t* wp_base,
    const uint64_t* wn_base,
    const uint64_t* xp_base,
    const uint64_t* xn_base
) {
    if ((!wp_base && !wn_base) || (!xp_base && !xn_base) || n_words == 0) {
        return 0;
    }

    int64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    size_t w = 0;

    if (wp_base && wn_base && xp_base && xn_base) {
        const size_t s2 = stride * 2;
        const size_t s3 = stride * 3;
        const size_t s4 = stride * 4;
        const uint64_t* wp_ptr = wp_base;
        const uint64_t* wn_ptr = wn_base;

        for (; w + 4 <= n_words; w += 4) {
            const uint64_t xp0 = xp_base[w + 0], xn0 = xn_base[w + 0];
            const uint64_t xp1 = xp_base[w + 1], xn1 = xn_base[w + 1];
            const uint64_t xp2 = xp_base[w + 2], xn2 = xn_base[w + 2];
            const uint64_t xp3 = xp_base[w + 3], xn3 = xn_base[w + 3];

            const uint64_t wp0 = wp_ptr[0],      wn0 = wn_ptr[0];
            const uint64_t wp1 = wp_ptr[stride], wn1 = wn_ptr[stride];
            const uint64_t wp2 = wp_ptr[s2],     wn2 = wn_ptr[s2];
            const uint64_t wp3 = wp_ptr[s3],     wn3 = wn_ptr[s3];

            const uint64_t pos0 = (wp0 & xp0) | (wn0 & xn0);
            const uint64_t neg0 = (wp0 & xn0) | (wn0 & xp0);
            const uint64_t pos1 = (wp1 & xp1) | (wn1 & xn1);
            const uint64_t neg1 = (wp1 & xn1) | (wn1 & xp1);
            const uint64_t pos2 = (wp2 & xp2) | (wn2 & xn2);
            const uint64_t neg2 = (wp2 & xn2) | (wn2 & xp2);
            const uint64_t pos3 = (wp3 & xp3) | (wn3 & xn3);
            const uint64_t neg3 = (wp3 & xn3) | (wn3 & xp3);

            acc0 += static_cast<int64_t>(__builtin_popcountll(pos0)) - static_cast<int64_t>(__builtin_popcountll(neg0));
            acc1 += static_cast<int64_t>(__builtin_popcountll(pos1)) - static_cast<int64_t>(__builtin_popcountll(neg1));
            acc2 += static_cast<int64_t>(__builtin_popcountll(pos2)) - static_cast<int64_t>(__builtin_popcountll(neg2));
            acc3 += static_cast<int64_t>(__builtin_popcountll(pos3)) - static_cast<int64_t>(__builtin_popcountll(neg3));

            wp_ptr += s4;
            wn_ptr += s4;
        }
        int64_t acc = acc0 + acc1 + acc2 + acc3;
        for (; w < n_words; ++w) {
            const uint64_t wp = *wp_ptr;
            const uint64_t wn = *wn_ptr;
            const uint64_t xp = xp_base[w];
            const uint64_t xn = xn_base[w];
            const uint64_t pos_bits = (wp & xp) | (wn & xn);
            const uint64_t neg_bits = (wp & xn) | (wn & xp);
            acc += static_cast<int64_t>(__builtin_popcountll(pos_bits)) - static_cast<int64_t>(__builtin_popcountll(neg_bits));
            wp_ptr += stride;
            wn_ptr += stride;
        }
        return acc;
    }

    int64_t acc = 0;
    for (; w < n_words; ++w) {
        const uint64_t wp = wp_base ? wp_base[stride * w] : 0ULL;
        const uint64_t wn = wn_base ? wn_base[stride * w] : 0ULL;
        const uint64_t xp = xp_base ? xp_base[w] : 0ULL;
        const uint64_t xn = xn_base ? xn_base[w] : 0ULL;
        const uint64_t pos_bits = (wp & xp) | (wn & xn);
        const uint64_t neg_bits = (wp & xn) | (wn & xp);
        acc += static_cast<int64_t>(__builtin_popcountll(pos_bits)) - static_cast<int64_t>(__builtin_popcountll(neg_bits));
    }
    return acc;
}

} // anonymous namespace

/**
 * @brief Multi-threaded combined Lennard-Jones (12-6) and Damped-Shifted-Force (DSF) Coulomb pair
 * evaluation with persistent worker pool and deterministic fixed-order thread reduction.
 */
XBT_EXPORT double xbt_eval_lj_dsf_pairs_mt(
    size_t n_atoms,
    const double* q,
    const double* sigma,
    const double* epsilon,
    const double* charge,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    size_t n_pairs,
    const int32_t* pair_i,
    const int32_t* pair_j,
    const double* pair_scale_lj,
    const double* pair_scale_coul,
    double rcut,
    double alpha_dsf,
    double coulomb_prefactor,
    int use_lfs,
    int use_fast_math,
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9,
    int n_threads
) {
    zero_output_buffers(n_atoms, out_forces, out_forces_fixed, out_virial9);

    if (!q || n_atoms == 0 || n_pairs == 0 || !pair_i || !pair_j || rcut <= 0.0) {
        return 0.0;
    }

    const bool need_table = (use_fast_math == 0 && charge != nullptr && coulomb_prefactor != 0.0 && alpha_dsf > 0.0);
    std::shared_ptr<const DsfAnalyticTable> tbl_holder;
    if (need_table) {
        tbl_holder = DsfTableCache::get(rcut, alpha_dsf);
    }

    const DsfParams p = init_dsf_params(
        box_H, box_Hinv, is_triclinic, charge,
        rcut, alpha_dsf, coulomb_prefactor, use_lfs, use_fast_math,
        tbl_holder.get()
    );

    XbtPersistentPool& pool = XbtPersistentPool::instance();
    int T = std::min(resolve_thread_count(n_threads), pool.max_threads());
    if (n_pairs < 8192 || T <= 1) {
        return eval_pairs_range(
            p, n_atoms, q, sigma, epsilon, charge,
            0, n_pairs, pair_i, pair_j, pair_scale_lj, pair_scale_coul,
            out_forces, out_forces_fixed, out_virial9
        );
    }

    if (static_cast<size_t>(T) > n_pairs) {
        T = static_cast<int>(n_pairs);
    }

    const size_t f_len = out_forces ? 3 * n_atoms : 0;
    const size_t fx_len = out_forces_fixed ? 3 * n_atoms : 0;
    const bool need_vir = (out_virial9 != nullptr && p.has_box);
    pool.prepare_scratch(T, f_len, fx_len, need_vir);

    struct PairCtx {
        XbtPersistentPool* pool;
        const DsfParams* p;
        size_t n_atoms;
        const double* q;
        const double* sigma;
        const double* epsilon;
        const double* charge;
        size_t n_pairs;
        const int32_t* pair_i;
        const int32_t* pair_j;
        const double* pair_scale_lj;
        const double* pair_scale_coul;
        size_t f_len;
        size_t fx_len;
        bool need_vir;
    } ctx{
        &pool, &p, n_atoms, q, sigma, epsilon, charge,
        n_pairs, pair_i, pair_j, pair_scale_lj, pair_scale_coul,
        f_len, fx_len, need_vir
    };

    pool.run(T, [&ctx](int t, int n_t) {
        const size_t start_k = (ctx.n_pairs * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
        const size_t end_k = (ctx.n_pairs * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
        WorkerScratch& acc = ctx.pool->scratch(t);
        acc.u = eval_pairs_range(
            *ctx.p, ctx.n_atoms, ctx.q, ctx.sigma, ctx.epsilon, ctx.charge,
            start_k, end_k, ctx.pair_i, ctx.pair_j, ctx.pair_scale_lj, ctx.pair_scale_coul,
            ctx.f_len > 0 ? acc.f.data() : nullptr,
            ctx.fx_len > 0 ? acc.fx.data() : nullptr,
            ctx.need_vir ? acc.vir : nullptr
        );
    });

    return reduce_pool_scratch(pool, T, n_atoms, p.has_box, out_forces, out_forces_fixed, out_virial9);
}

/**
 * @brief Evaluates combined Lennard-Jones (12-6) and Damped-Shifted-Force (DSF) Coulomb pair
 * interactions (forwards to xbt_eval_lj_dsf_pairs_mt with automatic thread selection).
 */
XBT_EXPORT double xbt_eval_lj_dsf_pairs(
    size_t n_atoms,
    const double* q,
    const double* sigma,
    const double* epsilon,
    const double* charge,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    size_t n_pairs,
    const int32_t* pair_i,
    const int32_t* pair_j,
    const double* pair_scale_lj,
    const double* pair_scale_coul,
    double rcut,
    double alpha_dsf,
    double coulomb_prefactor,
    int use_lfs,
    int use_fast_math,
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9
) {
    return xbt_eval_lj_dsf_pairs_mt(
        n_atoms, q, sigma, epsilon, charge, box_H, box_Hinv, is_triclinic,
        n_pairs, pair_i, pair_j, pair_scale_lj, pair_scale_coul,
        rcut, alpha_dsf, coulomb_prefactor, use_lfs, use_fast_math,
        out_forces, out_forces_fixed, out_virial9, 0
    );
}

/**
 * @brief High-speed multi-threaded 32-atom tile kernel matching Julia's Neighbors.TilePairCache
 * and StreamedLJ.PairTopology (1-based tile/atom indices, sorted uint64 topology keys).
 */
XBT_EXPORT double xbt_eval_lj_dsf_tiles_mt(
    size_t n_atoms, const double* q, const double* sigma, const double* epsilon, const double* charge,
    const double* box_H, const double* box_Hinv, int is_triclinic,
    size_t n_tile_pairs, const int32_t* tile_i, const int32_t* tile_j,
    const int32_t* order, const int32_t* tile_starts, const int32_t* tile_ends,
    size_t n_excl, const uint64_t* excl_keys,
    size_t n_scaled, const uint64_t* scaled_keys, const double* scaled_weights,
    double lj14_scale, double coul14_scale,
    double rcut, double alpha_dsf, double coulomb_prefactor,
    int use_lfs, int use_fast_math, int n_threads,
    double* out_forces, int64_t* out_forces_fixed, double* out_virial9
) {
    zero_output_buffers(n_atoms, out_forces, out_forces_fixed, out_virial9);

    if (!q || n_atoms == 0 || n_tile_pairs == 0 || !tile_i || !tile_j ||
        !order || !tile_starts || !tile_ends || rcut <= 0.0) {
        return 0.0;
    }

    const bool need_table = (use_fast_math == 0 && charge != nullptr && coulomb_prefactor != 0.0 && alpha_dsf > 0.0);
    std::shared_ptr<const DsfAnalyticTable> tbl_holder;
    if (need_table) {
        tbl_holder = DsfTableCache::get(rcut, alpha_dsf);
    }

    const DsfParams p = init_dsf_params(
        box_H, box_Hinv, is_triclinic, charge,
        rcut, alpha_dsf, coulomb_prefactor, use_lfs, use_fast_math,
        tbl_holder.get()
    );

    XbtPersistentPool& pool = XbtPersistentPool::instance();
    int T = std::min(resolve_thread_count(n_threads), pool.max_threads());
    if (n_tile_pairs < 128 || T <= 1) {
        return eval_tiles_range(
            p, n_atoms, q, sigma, epsilon, charge,
            0, n_tile_pairs, tile_i, tile_j, order, tile_starts, tile_ends,
            n_excl, excl_keys, n_scaled, scaled_keys, scaled_weights,
            lj14_scale, coul14_scale,
            out_forces, out_forces_fixed, out_virial9
        );
    }

    if (static_cast<size_t>(T) > n_tile_pairs) {
        T = static_cast<int>(n_tile_pairs);
    }

    const size_t f_len = out_forces ? 3 * n_atoms : 0;
    const size_t fx_len = out_forces_fixed ? 3 * n_atoms : 0;
    const bool need_vir = (out_virial9 != nullptr && p.has_box);
    pool.prepare_scratch(T, f_len, fx_len, need_vir);

    struct TileCtx {
        XbtPersistentPool* pool;
        const DsfParams* p;
        size_t n_atoms;
        const double* q;
        const double* sigma;
        const double* epsilon;
        const double* charge;
        size_t n_tile_pairs;
        const int32_t* tile_i;
        const int32_t* tile_j;
        const int32_t* order;
        const int32_t* tile_starts;
        const int32_t* tile_ends;
        size_t n_excl;
        const uint64_t* excl_keys;
        size_t n_scaled;
        const uint64_t* scaled_keys;
        const double* scaled_weights;
        double lj14_scale;
        double coul14_scale;
        size_t f_len;
        size_t fx_len;
        bool need_vir;
    } ctx{
        &pool, &p, n_atoms, q, sigma, epsilon, charge,
        n_tile_pairs, tile_i, tile_j, order, tile_starts, tile_ends,
        n_excl, excl_keys, n_scaled, scaled_keys, scaled_weights,
        lj14_scale, coul14_scale,
        f_len, fx_len, need_vir
    };

    pool.run(T, [&ctx](int t, int n_t) {
        const size_t start_k = (ctx.n_tile_pairs * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
        const size_t end_k = (ctx.n_tile_pairs * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
        WorkerScratch& acc = ctx.pool->scratch(t);
        acc.u = eval_tiles_range(
            *ctx.p, ctx.n_atoms, ctx.q, ctx.sigma, ctx.epsilon, ctx.charge,
            start_k, end_k, ctx.tile_i, ctx.tile_j, ctx.order, ctx.tile_starts, ctx.tile_ends,
            ctx.n_excl, ctx.excl_keys, ctx.n_scaled, ctx.scaled_keys, ctx.scaled_weights,
            ctx.lj14_scale, ctx.coul14_scale,
            ctx.f_len > 0 ? acc.f.data() : nullptr,
            ctx.fx_len > 0 ? acc.fx.data() : nullptr,
            ctx.need_vir ? acc.vir : nullptr
        );
    });

    return reduce_pool_scratch(pool, T, n_atoms, p.has_box, out_forces, out_forces_fixed, out_virial9);
}

/**
 * @brief Difference-Set Conflict-Free Tile Orbit Scheduler (GSI-R-0004 / GSI-R-0012).
 */
XBT_EXPORT int32_t xbt_schedule_disjoint_tile_orbits(
    size_t n_tiles,
    size_t n_tile_pairs,
    const int32_t* tile_i,
    const int32_t* tile_j,
    int32_t* out_orbit_id
) {
    if (n_tile_pairs == 0) return 0;
    if (n_tiles == 0 || !tile_i || !tile_j || !out_orbit_id) return -1;

    size_t words_per_tile = 4;
    std::vector<uint64_t> tile_mask(n_tiles * words_per_tile, 0ULL);
    int32_t n_orbits = 0;

    for (size_t k = 0; k < n_tile_pairs; ++k) {
        const int32_t ti = tile_i[k];
        const int32_t tj = tile_j[k];
        if (ti < 1 || tj < 1 || static_cast<size_t>(ti) > n_tiles || static_cast<size_t>(tj) > n_tiles) {
            return -1;
        }
        const size_t u = static_cast<size_t>(ti - 1);
        const size_t v = static_cast<size_t>(tj - 1);

        uint64_t* mask_u = &tile_mask[u * words_per_tile];
        uint64_t* mask_v = &tile_mask[v * words_per_tile];

        int32_t assigned = -1;
        for (size_t w = 0; w < words_per_tile; ++w) {
            const uint64_t avail = ~(mask_u[w] | mask_v[w]);
            if (avail != 0ULL) {
                const int bit = __builtin_ctzll(avail);
                const uint64_t bit_mask = (1ULL << bit);
                mask_u[w] |= bit_mask;
                mask_v[w] |= bit_mask;
                assigned = static_cast<int32_t>(w * 64 + static_cast<size_t>(bit));
                break;
            }
        }

        if (assigned < 0) {
            const size_t old_w = words_per_tile;
            const size_t new_w = old_w * 2;
            std::vector<uint64_t> grown(n_tiles * new_w, 0ULL);
            for (size_t t = 0; t < n_tiles; ++t) {
                std::memcpy(&grown[t * new_w], &tile_mask[t * old_w], old_w * sizeof(uint64_t));
            }
            tile_mask.swap(grown);
            words_per_tile = new_w;

            tile_mask[u * words_per_tile + old_w] |= 1ULL;
            tile_mask[v * words_per_tile + old_w] |= 1ULL;
            assigned = static_cast<int32_t>(old_w * 64);
        }

        out_orbit_id[k] = assigned;
        if (assigned + 1 > n_orbits) {
            n_orbits = assigned + 1;
        }
    }

    return n_orbits;
}

/**
 * @brief Bit-sliced Dual-Rail Balanced Ternary Matrix-Vector product using 4-word unrolled hardware POPCNT (ARM64 NEON).
 */
XBT_EXPORT void xbt_ternary_matvec_dualrail(
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

    for (size_t m = 0; m < M; ++m) {
        const int64_t acc = ternary_dot_strided_4way(
            M, K_words,
            W_P ? W_P + m : nullptr,
            W_N ? W_N + m : nullptr,
            x_P, x_N
        );
        const double r_scale = row_scales ? row_scales[m] : 1.0;
        y_out[m] = r_scale * x_scale * static_cast<double>(acc);
    }
}

/**
 * @brief Multi-plane Bit-sliced Balanced Ternary Matrix-Vector product (GSI-R-0017).
 */
XBT_EXPORT void xbt_ternary_matvec_multiplane(
    size_t M,
    size_t n_words,
    size_t num_planes,
    size_t kx,
    const uint64_t* W_P,
    const uint64_t* W_N,
    const double* row_scales,
    const uint64_t* x_P,
    const uint64_t* x_N,
    const double* x_scales,
    double* y_out
) {
    if (!y_out || M == 0) return;
    if (n_words == 0 || num_planes == 0 || kx == 0 || (!W_P && !W_N) || (!x_P && !x_N)) {
        std::memset(y_out, 0, M * sizeof(double));
        return;
    }

    auto compute_rows = [&](size_t m_start, size_t m_end) {
        const size_t plane_stride = M * n_words;
        for (size_t m = m_start; m < m_end; ++m) {
            double sum_m = 0.0;
            for (size_t p = 0; p < num_planes; ++p) {
                const double r_scale = row_scales ? row_scales[m + M * p] : 1.0;
                if (r_scale == 0.0) continue;
                const uint64_t* wp_plane = W_P ? (W_P + m + plane_stride * p) : nullptr;
                const uint64_t* wn_plane = W_N ? (W_N + m + plane_stride * p) : nullptr;

                for (size_t k = 0; k < kx; ++k) {
                    const double x_sc = x_scales ? x_scales[k] : 1.0;
                    if (x_sc == 0.0) continue;
                    const uint64_t* xp_col = x_P ? (x_P + n_words * k) : nullptr;
                    const uint64_t* xn_col = x_N ? (x_N + n_words * k) : nullptr;
                    const int64_t acc = ternary_dot_strided_4way(
                        M, n_words, wp_plane, wn_plane, xp_col, xn_col
                    );
                    sum_m += r_scale * x_sc * static_cast<double>(acc);
                }
            }
            y_out[m] = sum_m;
        }
    };

    const size_t work = M * n_words * num_planes * kx;
    if (M >= 64 && work >= 65536) {
        XbtPersistentPool& pool = XbtPersistentPool::instance();
        const int T = std::min({resolve_thread_count(0), pool.max_threads(), static_cast<int>(M)});
        if (T > 1) {
            struct MatvecCtx {
                decltype(compute_rows)* fn;
                size_t M;
            } ctx{&compute_rows, M};

            pool.run(T, [&ctx](int t, int n_t) {
                const size_t m_start = (ctx.M * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
                const size_t m_end = (ctx.M * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
                (*ctx.fn)(m_start, m_end);
            });
            return;
        }
    }

    compute_rows(0, M);
}

// =============================================================================
// Compiled CSR Verlet Plan Engine (Exact Degree-4 DSF & ARM64 NEON SIMD)
// =============================================================================

namespace {

struct VerletScaledPair {
    int32_t i; // Morton row index
    int32_t j; // Morton col index
    double w_lj;
    double w_coul;
};

struct XbtVerletPlan {
    static constexpr uint8_t FLAG_HAS_LJ = 1u;
    static constexpr uint8_t FLAG_HAS_COUL = 2u;

    size_t n_atoms = 0;
    bool do_lj = false;
    bool do_coul = false;
    bool use_lfs = false;
    bool order_is_perm = false;

    double rcut = 0.0;
    double rc2 = 0.0;
    double inv_rc = 0.0;
    double inv_rc2 = 0.0;
    double alpha_dsf = 0.0;
    double coulomb_prefactor = 0.0;
    double two_alpha_sqrt_pi = 0.0;
    double Arc = 0.0;
    double Brc = 0.0;

    DsfAnalyticTable dsf_table;

    // Morton-ordered atom metadata & parameters (0 .. n_atoms - 1)
    std::vector<int32_t> morton_to_atom;
    std::vector<double> half_sig;
    std::vector<double> sqrt_eps;
    std::vector<double> chg;
    std::vector<float> half_sig_f;
    std::vector<float> sqrt_eps_f;
    std::vector<float> chg_f;
    std::vector<uint8_t> row_flags;

    // Compiled CSR normal pairs (w_lj == 1.0 && w_coul == 1.0)
    std::vector<size_t> csr_row_ptr;
    std::vector<int32_t> csr_col_idx;

    // Compact scaled (1-4) pairs
    std::vector<VerletScaledPair> scaled_pairs;

    // Pre-allocated Morton-ordered coordinate workspace for zero-alloc eval
    std::vector<double> morton_q;
    std::mutex eval_mtx;
};

inline void eval_verlet_csr_exact_row_range(
    const XbtVerletPlan& plan,
    const double* morton_q,
    bool has_box,
    bool is_triclinic,
    double L0, double L1, double L2,
    double inv_L0, double inv_L1, double inv_L2,
    const double* box_H,
    const double* box_Hinv,
    size_t row_start,
    size_t row_end,
    bool do_forces,
    bool do_fixed,
    bool need_virial,
    double* f_buf,
    int64_t* fx_buf,
    double* vir,
    double& u_loc
) noexcept {
    const double rc2 = plan.rc2;
    const double rcut = plan.rcut;
    const double inv_rc = plan.inv_rc;
    const double inv_rc2 = plan.inv_rc2;
    const bool use_lfs = plan.use_lfs;
    const bool do_lj_global = plan.do_lj;
    const bool do_coul_global = plan.do_coul;
    const bool need_r = (do_coul_global || use_lfs);
    const double coul_pref = plan.coulomb_prefactor;
    const double Arc = plan.Arc;
    const double Brc = plan.Brc;

    const size_t* row_ptr = plan.csr_row_ptr.data();
    const int32_t* col_idx = plan.csr_col_idx.data();
    const double* half_sig = plan.half_sig.data();
    const double* sqrt_eps = plan.sqrt_eps.data();
    const double* chg = plan.chg.data();
    const uint8_t* row_flags = plan.row_flags.data();
    const DsfAnalyticTable& tbl = plan.dsf_table;

    for (size_t r = row_start; r < row_end; ++r) {
        const size_t k_begin = row_ptr[r];
        const size_t k_end = row_ptr[r + 1];
        if (k_begin == k_end) continue;

        const uint8_t rflags = row_flags[r];
        if (rflags == 0) continue;

        const bool row_has_lj = (rflags & XbtVerletPlan::FLAG_HAS_LJ) != 0;
        const bool row_has_coul = (rflags & XbtVerletPlan::FLAG_HAS_COUL) != 0;

        const double qxi = morton_q[3 * r + 0];
        const double qyi = morton_q[3 * r + 1];
        const double qzi = morton_q[3 * r + 2];
        const double hsig_i = row_has_lj ? half_sig[r] : 0.0;
        const double seps_i = row_has_lj ? sqrt_eps[r] : 0.0;
        const double k_chg_i = row_has_coul ? (coul_pref * chg[r]) : 0.0;

        double fxi = 0.0, fyi = 0.0, fzi = 0.0;
        int64_t fxi_fix = 0, fyi_fix = 0, fzi_fix = 0;

        for (size_t k = k_begin; k < k_end; ++k) {
            const size_t c = static_cast<size_t>(col_idx[k]);
            const double dx0 = qxi - morton_q[3 * c + 0];
            const double dy0 = qyi - morton_q[3 * c + 1];
            const double dz0 = qzi - morton_q[3 * c + 2];

            double dx, dy, dz, wx, wy, wz;
            apply_mic(
                has_box, is_triclinic,
                L0, L1, L2, inv_L0, inv_L1, inv_L2,
                box_H, box_Hinv,
                dx0, dy0, dz0, need_virial,
                dx, dy, dz, wx, wy, wz
            );

            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2 || r2 < 1e-24) continue;

            double r = 0.0;
            double inv_r = 0.0;
            double inv_r2 = 0.0;
            if (need_r) {
                r = std::sqrt(r2);
                inv_r = 1.0 / r;
                inv_r2 = inv_r * inv_r;
            } else {
                inv_r2 = 1.0 / r2;
            }

            double u_pair = 0.0;
            double c_total = 0.0;

            if (do_lj_global && row_has_lj) {
                const double seps_j = sqrt_eps[c];
                if (seps_j > 0.0) {
                    const double sij = hsig_i + half_sig[c];
                    if (sij > 0.0) {
                        const double eij = seps_i * seps_j;
                        const double sij2 = sij * sij;
                        const double s2 = sij2 * inv_r2;
                        const double s6 = s2 * s2 * s2;
                        const double s12 = s6 * s6;

                        const double s2c = sij2 * inv_rc2;
                        const double s6c = s2c * s2c * s2c;
                        const double s12c = s6c * s6c;

                        double u_lj = 4.0 * eij * ((s12 - s6) - (s12c - s6c));
                        double c_lj = 24.0 * eij * (2.0 * s12 - s6) * inv_r2;

                        if (use_lfs) {
                            const double duc = 24.0 * eij * (s6c - 2.0 * s12c) * inv_rc;
                            u_lj -= duc * (r - rcut);
                            c_lj += duc * inv_r;
                        }

                        u_pair += u_lj;
                        c_total += c_lj;
                    }
                }
            }

            if (do_coul_global && row_has_coul) {
                const double chg_j = chg[c];
                if (chg_j != 0.0) {
                    const double kq = k_chg_i * chg_j;
                    double phi0 = 0.0, dphi0 = 0.0;
                    tbl.lookup(r2, phi0, dphi0);
                    const double erfc_over_r = inv_r - phi0;
                    const double inv_r3 = inv_r2 * inv_r;
                    u_pair += kq * (erfc_over_r - Brc + Arc * r);
                    c_total += kq * (inv_r3 + 2.0 * dphi0 - Arc * inv_r);
                }
            }

            u_loc += u_pair;
            const double fx = c_total * dx;
            const double fy = c_total * dy;
            const double fz = c_total * dz;

            if (do_forces) {
                fxi += fx;
                fyi += fy;
                fzi += fz;
                f_buf[3 * c + 0] -= fx;
                f_buf[3 * c + 1] -= fy;
                f_buf[3 * c + 2] -= fz;
            }
            if (do_fixed) {
                const int64_t fx_int = std::llround(fx * FIXED_POINT_SCALE);
                const int64_t fy_int = std::llround(fy * FIXED_POINT_SCALE);
                const int64_t fz_int = std::llround(fz * FIXED_POINT_SCALE);
                fxi_fix += fx_int;
                fyi_fix += fy_int;
                fzi_fix += fz_int;
                fx_buf[3 * c + 0] -= fx_int;
                fx_buf[3 * c + 1] -= fy_int;
                fx_buf[3 * c + 2] -= fz_int;
            }
            if (need_virial) {
                accum_virial9(vir, fx, fy, fz, wx, wy, wz);
            }
        }

        if (do_forces) {
            f_buf[3 * r + 0] += fxi;
            f_buf[3 * r + 1] += fyi;
            f_buf[3 * r + 2] += fzi;
        }
        if (do_fixed) {
            fx_buf[3 * r + 0] += fxi_fix;
            fx_buf[3 * r + 1] += fyi_fix;
            fx_buf[3 * r + 2] += fzi_fix;
        }
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
inline void eval_verlet_csr_neon_row_range(
    const XbtVerletPlan& plan,
    const double* morton_q,
    bool has_box,
    double L0, double L1, double L2,
    double inv_L0, double inv_L1, double inv_L2,
    size_t row_start,
    size_t row_end,
    bool do_forces,
    bool do_fixed,
    bool need_virial,
    double* f_buf,
    int64_t* fx_buf,
    double* vir,
    double& u_loc
) noexcept {
    const float rc2_f = static_cast<float>(plan.rc2);
    const float rcut_f = static_cast<float>(plan.rcut);
    const float inv_rc_f = static_cast<float>(plan.inv_rc);
    const float inv_rc2_f = static_cast<float>(plan.inv_rc2);
    const float Arc_f = static_cast<float>(plan.Arc);
    const float Brc_f = static_cast<float>(plan.Brc);
    const float du_f = static_cast<float>(plan.dsf_table.du);
    const float inv_du_f = static_cast<float>(plan.dsf_table.inv_du);
    const float L0_f = static_cast<float>(L0);
    const float L1_f = static_cast<float>(L1);
    const float L2_f = static_cast<float>(L2);
    const float inv_L0_f = static_cast<float>(inv_L0);
    const float inv_L1_f = static_cast<float>(inv_L1);
    const float inv_L2_f = static_cast<float>(inv_L2);

    const float32x4_t vrc2 = vdupq_n_f32(rc2_f);
    const float32x4_t vmin_r2 = vdupq_n_f32(1e-24f);
    const float32x4_t vrcut = vdupq_n_f32(rcut_f);
    const float32x4_t vinv_rc = vdupq_n_f32(inv_rc_f);
    const float32x4_t vinv_rc2 = vdupq_n_f32(inv_rc2_f);
    const float32x4_t vArc = vdupq_n_f32(Arc_f);
    const float32x4_t vBrc = vdupq_n_f32(Brc_f);
    const float32x4_t vdu = vdupq_n_f32(du_f);
    const float32x4_t vinv_du = vdupq_n_f32(inv_du_f);
    const float32x4_t vL0 = vdupq_n_f32(L0_f);
    const float32x4_t vL1 = vdupq_n_f32(L1_f);
    const float32x4_t vL2 = vdupq_n_f32(L2_f);
    const float32x4_t vinv_L0 = vdupq_n_f32(inv_L0_f);
    const float32x4_t vinv_L1 = vdupq_n_f32(inv_L1_f);
    const float32x4_t vinv_L2 = vdupq_n_f32(inv_L2_f);
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_two = vdupq_n_f32(2.0f);
    const float32x4_t v_three = vdupq_n_f32(3.0f);
    const float32x4_t v_four = vdupq_n_f32(4.0f);
    const float32x4_t v_tf = vdupq_n_f32(24.0f);
    const float32x4_t v_half = vdupq_n_f32(0.5f);
    const int32x4_t v_zero_i = vdupq_n_s32(0);
    const int32x4_t v_max_k = vdupq_n_s32(DsfAnalyticTable::N_TAB - 1);

    const bool use_lfs = plan.use_lfs;
    const bool do_lj_global = plan.do_lj;
    const bool do_coul_global = plan.do_coul;
    const double coul_pref = plan.coulomb_prefactor;

    const size_t* row_ptr = plan.csr_row_ptr.data();
    const int32_t* col_idx = plan.csr_col_idx.data();
    const float* half_sig_f = plan.half_sig_f.data();
    const float* sqrt_eps_f = plan.sqrt_eps_f.data();
    const float* chg_f = plan.chg_f.data();
    const double* half_sig = plan.half_sig.data();
    const double* sqrt_eps = plan.sqrt_eps.data();
    const double* chg = plan.chg.data();
    const uint8_t* row_flags = plan.row_flags.data();
    const DsfAnalyticTable& tbl = plan.dsf_table;
    const DsfAnalyticTable::FBin* fbins = tbl.fbins.data();

    for (size_t r = row_start; r < row_end; ++r) {
        const size_t k_begin = row_ptr[r];
        const size_t k_end = row_ptr[r + 1];
        if (k_begin == k_end) continue;

        const uint8_t rflags = row_flags[r];
        if (rflags == 0) continue;

        const bool row_has_lj = (rflags & XbtVerletPlan::FLAG_HAS_LJ) != 0;
        const bool row_has_coul = (rflags & XbtVerletPlan::FLAG_HAS_COUL) != 0;

        const double qxi = morton_q[3 * r + 0];
        const double qyi = morton_q[3 * r + 1];
        const double qzi = morton_q[3 * r + 2];

        const float hsig_i_f = row_has_lj ? half_sig_f[r] : 0.0f;
        const float seps_i_f = row_has_lj ? sqrt_eps_f[r] : 0.0f;
        const float k_chg_i_f = row_has_coul ? static_cast<float>(coul_pref * chg[r]) : 0.0f;

        const float32x4_t v_hsig_i = vdupq_n_f32(hsig_i_f);
        const float32x4_t v_seps_i = vdupq_n_f32(seps_i_f);
        const float32x4_t v_kchg_i = vdupq_n_f32(k_chg_i_f);

        double fxi = 0.0, fyi = 0.0, fzi = 0.0;
        int64_t fxi_fix = 0, fyi_fix = 0, fzi_fix = 0;

        size_t k = k_begin;
        for (; k + 4 <= k_end; k += 4) {
            const size_t c0 = static_cast<size_t>(col_idx[k + 0]);
            const size_t c1 = static_cast<size_t>(col_idx[k + 1]);
            const size_t c2 = static_cast<size_t>(col_idx[k + 2]);
            const size_t c3 = static_cast<size_t>(col_idx[k + 3]);

            const double* q0 = morton_q + 3 * c0;
            const double* q1 = morton_q + 3 * c1;
            const double* q2 = morton_q + 3 * c2;
            const double* q3 = morton_q + 3 * c3;

            const float dx0_a[4] = {
                static_cast<float>(qxi - q0[0]), static_cast<float>(qxi - q1[0]),
                static_cast<float>(qxi - q2[0]), static_cast<float>(qxi - q3[0])
            };
            const float dy0_a[4] = {
                static_cast<float>(qyi - q0[1]), static_cast<float>(qyi - q1[1]),
                static_cast<float>(qyi - q2[1]), static_cast<float>(qyi - q3[1])
            };
            const float dz0_a[4] = {
                static_cast<float>(qzi - q0[2]), static_cast<float>(qzi - q1[2]),
                static_cast<float>(qzi - q2[2]), static_cast<float>(qzi - q3[2])
            };

            float32x4_t dx = vld1q_f32(dx0_a);
            float32x4_t dy = vld1q_f32(dy0_a);
            float32x4_t dz = vld1q_f32(dz0_a);

            if (has_box) {
                dx = vfmsq_f32(dx, vL0, vrndnq_f32(vmulq_f32(dx, vinv_L0)));
                dy = vfmsq_f32(dy, vL1, vrndnq_f32(vmulq_f32(dy, vinv_L1)));
                dz = vfmsq_f32(dz, vL2, vrndnq_f32(vmulq_f32(dz, vinv_L2)));
            }

            float32x4_t r2 = vmulq_f32(dx, dx);
            r2 = vfmaq_f32(r2, dy, dy);
            r2 = vfmaq_f32(r2, dz, dz);

            const uint32x4_t mask = vandq_u32(vcltq_f32(r2, vrc2), vcgeq_f32(r2, vmin_r2));
            if (vmaxvq_u32(mask) == 0) continue;

            const float32x4_t safe_r2 = vbslq_f32(mask, r2, v_one);

            // vrsqrteq_f32 + 2 Newton-Raphson iterations -> full ~23-bit FP32 accuracy
            float32x4_t inv_r = vrsqrteq_f32(safe_r2);
            inv_r = vmulq_f32(inv_r, vrsqrtsq_f32(vmulq_f32(safe_r2, inv_r), inv_r));
            inv_r = vmulq_f32(inv_r, vrsqrtsq_f32(vmulq_f32(safe_r2, inv_r), inv_r));
            const float32x4_t r_vec = vmulq_f32(safe_r2, inv_r);
            const float32x4_t inv_r2 = vmulq_f32(inv_r, inv_r);

            float32x4_t u_vec = vdupq_n_f32(0.0f);
            float32x4_t c_vec = vdupq_n_f32(0.0f);

            if (do_lj_global && row_has_lj) {
                const float se_a[4] = { sqrt_eps_f[c0], sqrt_eps_f[c1], sqrt_eps_f[c2], sqrt_eps_f[c3] };
                const float32x4_t eij = vmulq_f32(v_seps_i, vld1q_f32(se_a));
                if (vmaxvq_f32(eij) > 0.0f) {
                    const float hs_a[4] = { half_sig_f[c0], half_sig_f[c1], half_sig_f[c2], half_sig_f[c3] };
                    const float32x4_t sij = vaddq_f32(v_hsig_i, vld1q_f32(hs_a));
                    const float32x4_t sij2 = vmulq_f32(sij, sij);
                    const float32x4_t s2 = vmulq_f32(sij2, inv_r2);
                    const float32x4_t s6 = vmulq_f32(vmulq_f32(s2, s2), s2);
                    const float32x4_t s12 = vmulq_f32(s6, s6);

                    const float32x4_t s2c = vmulq_f32(sij2, vinv_rc2);
                    const float32x4_t s6c = vmulq_f32(vmulq_f32(s2c, s2c), s2c);
                    const float32x4_t s12c = vmulq_f32(s6c, s6c);

                    float32x4_t u_lj = vmulq_f32(
                        vmulq_f32(v_four, eij),
                        vsubq_f32(vsubq_f32(s12, s6), vsubq_f32(s12c, s6c))
                    );
                    const float32x4_t tf_eij = vmulq_f32(v_tf, eij);
                    float32x4_t c_lj = vmulq_f32(
                        vmulq_f32(tf_eij, vsubq_f32(vmulq_f32(v_two, s12), s6)),
                        inv_r2
                    );

                    if (use_lfs) {
                        const float32x4_t duc = vmulq_f32(
                            vmulq_f32(tf_eij, vsubq_f32(s6c, vmulq_f32(v_two, s12c))),
                            vinv_rc
                        );
                        u_lj = vfmsq_f32(u_lj, duc, vsubq_f32(r_vec, vrcut));
                        c_lj = vfmaq_f32(c_lj, duc, inv_r);
                    }

                    u_vec = vaddq_f32(u_vec, u_lj);
                    c_vec = vaddq_f32(c_vec, c_lj);
                }
            }

            if (do_coul_global && row_has_coul) {
                const float chg_a[4] = { chg_f[c0], chg_f[c1], chg_f[c2], chg_f[c3] };
                const float32x4_t qij = vmulq_f32(v_kchg_i, vld1q_f32(chg_a));
                if (vmaxvq_f32(vabsq_f32(qij)) > 0.0f) {
                    int32x4_t k_idx = vcvtq_s32_f32(vmulq_f32(safe_r2, vinv_du));
                    k_idx = vminq_s32(vmaxq_s32(k_idx, v_zero_i), v_max_k);
                    const float32x4_t umid = vmulq_f32(vaddq_f32(vcvtq_f32_s32(k_idx), v_half), vdu);
                    const float32x4_t dt = vsubq_f32(safe_r2, umid);

                    const int32_t k0 = vgetq_lane_s32(k_idx, 0);
                    const int32_t k1 = vgetq_lane_s32(k_idx, 1);
                    const int32_t k2 = vgetq_lane_s32(k_idx, 2);
                    const int32_t k3 = vgetq_lane_s32(k_idx, 3);

                    const float32x4_t r0 = vld1q_f32(&fbins[k0].c0);
                    const float32x4_t r1 = vld1q_f32(&fbins[k1].c0);
                    const float32x4_t r2_row = vld1q_f32(&fbins[k2].c0);
                    const float32x4_t r3_row = vld1q_f32(&fbins[k3].c0);

                    const float32x4x2_t t01 = vtrnq_f32(r0, r1);
                    const float32x4x2_t t23 = vtrnq_f32(r2_row, r3_row);
                    const float32x4_t vc0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
                    const float32x4_t vc1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
                    const float32x4_t vc2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
                    const float32x4_t vc3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
                    const float c4_a[4] = { fbins[k0].c4, fbins[k1].c4, fbins[k2].c4, fbins[k3].c4 };
                    const float32x4_t vc4 = vld1q_f32(c4_a);

                    float32x4_t phi0 = vfmaq_f32(vc3, dt, vc4);
                    phi0 = vfmaq_f32(vc2, dt, phi0);
                    phi0 = vfmaq_f32(vc1, dt, phi0);
                    phi0 = vfmaq_f32(vc0, dt, phi0);

                    float32x4_t dphi0 = vfmaq_f32(vmulq_f32(v_three, vc3), dt, vmulq_f32(v_four, vc4));
                    dphi0 = vfmaq_f32(vmulq_f32(v_two, vc2), dt, dphi0);
                    dphi0 = vfmaq_f32(vc1, dt, dphi0);

                    const float32x4_t erfc_over_r = vsubq_f32(inv_r, phi0);
                    const float32x4_t u_coul = vmulq_f32(qij, vfmaq_f32(vsubq_f32(erfc_over_r, vBrc), vArc, r_vec));
                    const float32x4_t inv_r3 = vmulq_f32(inv_r2, inv_r);
                    const float32x4_t c_coul = vmulq_f32(
                        qij,
                        vfmsq_f32(vfmaq_f32(inv_r3, v_two, dphi0), vArc, inv_r)
                    );

                    u_vec = vaddq_f32(u_vec, u_coul);
                    c_vec = vaddq_f32(c_vec, c_coul);
                }
            }

            u_vec = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(u_vec), mask));
            c_vec = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(c_vec), mask));

            const float32x4_t fx_vec = vmulq_f32(c_vec, dx);
            const float32x4_t fy_vec = vmulq_f32(c_vec, dy);
            const float32x4_t fz_vec = vmulq_f32(c_vec, dz);

            // Widen to float64 before horizontal sum so large totals retain mantissa precision
            const float64x2_t u_lo = vcvt_f64_f32(vget_low_f32(u_vec));
            const float64x2_t u_hi = vcvt_high_f64_f32(u_vec);
            u_loc += vaddvq_f64(vaddq_f64(u_lo, u_hi));

            if (do_forces || do_fixed || need_virial) {
                uint32_t m_arr[4];
                float fx_arr[4], fy_arr[4], fz_arr[4];
                vst1q_u32(m_arr, mask);
                vst1q_f32(fx_arr, fx_vec);
                vst1q_f32(fy_arr, fy_vec);
                vst1q_f32(fz_arr, fz_vec);

                float dx_arr[4], dy_arr[4], dz_arr[4];
                if (need_virial) {
                    vst1q_f32(dx_arr, dx);
                    vst1q_f32(dy_arr, dy);
                    vst1q_f32(dz_arr, dz);
                }

                const size_t cols4[4] = { c0, c1, c2, c3 };
                for (int lane = 0; lane < 4; ++lane) {
                    if (m_arr[lane] == 0) continue;
                    const double fx = static_cast<double>(fx_arr[lane]);
                    const double fy = static_cast<double>(fy_arr[lane]);
                    const double fz = static_cast<double>(fz_arr[lane]);
                    const size_t c = cols4[lane];

                    if (do_forces) {
                        fxi += fx;
                        fyi += fy;
                        fzi += fz;
                        f_buf[3 * c + 0] -= fx;
                        f_buf[3 * c + 1] -= fy;
                        f_buf[3 * c + 2] -= fz;
                    }
                    if (do_fixed) {
                        const int64_t fx_int = std::llround(fx * FIXED_POINT_SCALE);
                        const int64_t fy_int = std::llround(fy * FIXED_POINT_SCALE);
                        const int64_t fz_int = std::llround(fz * FIXED_POINT_SCALE);
                        fxi_fix += fx_int;
                        fyi_fix += fy_int;
                        fzi_fix += fz_int;
                        fx_buf[3 * c + 0] -= fx_int;
                        fx_buf[3 * c + 1] -= fy_int;
                        fx_buf[3 * c + 2] -= fz_int;
                    }
                    if (need_virial) {
                        const double wx = static_cast<double>(dx_arr[lane]) * inv_L0;
                        const double wy = static_cast<double>(dy_arr[lane]) * inv_L1;
                        const double wz = static_cast<double>(dz_arr[lane]) * inv_L2;
                        accum_virial9(vir, fx, fy, fz, wx, wy, wz);
                    }
                }
            }
        }

        // Scalar tail (< 4 remaining neighbors)
        for (; k < k_end; ++k) {
            const size_t c = static_cast<size_t>(col_idx[k]);
            const double dx0 = qxi - morton_q[3 * c + 0];
            const double dy0 = qyi - morton_q[3 * c + 1];
            const double dz0 = qzi - morton_q[3 * c + 2];

            double dx, dy, dz, wx, wy, wz;
            apply_mic(
                has_box, false,
                L0, L1, L2, inv_L0, inv_L1, inv_L2,
                nullptr, nullptr,
                dx0, dy0, dz0, need_virial,
                dx, dy, dz, wx, wy, wz
            );

            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= plan.rc2 || r2 < 1e-24) continue;

            const double dist = std::sqrt(r2);
            const double inv_r = 1.0 / dist;
            const double inv_r2 = inv_r * inv_r;

            double u_pair = 0.0, c_total = 0.0;
            if (do_lj_global && row_has_lj) {
                const double seps_j = sqrt_eps[c];
                if (seps_j > 0.0) {
                    const double sij = half_sig[r] + half_sig[c];
                    if (sij > 0.0) {
                        const double eij = sqrt_eps[r] * seps_j;
                        const double sij2 = sij * sij;
                        const double s2 = sij2 * inv_r2;
                        const double s6 = s2 * s2 * s2;
                        const double s12 = s6 * s6;
                        const double s2c = sij2 * plan.inv_rc2;
                        const double s6c = s2c * s2c * s2c;
                        const double s12c = s6c * s6c;
                        double u_lj = 4.0 * eij * ((s12 - s6) - (s12c - s6c));
                        double c_lj = 24.0 * eij * (2.0 * s12 - s6) * inv_r2;
                        if (use_lfs) {
                            const double duc = 24.0 * eij * (s6c - 2.0 * s12c) * plan.inv_rc;
                            u_lj -= duc * (dist - plan.rcut);
                            c_lj += duc * inv_r;
                        }
                        u_pair += u_lj;
                        c_total += c_lj;
                    }
                }
            }
            if (do_coul_global && row_has_coul) {
                const double chg_j = chg[c];
                if (chg_j != 0.0) {
                    const double kq = coul_pref * chg[r] * chg_j;
                    double phi0 = 0.0, dphi0 = 0.0;
                    tbl.lookup(r2, phi0, dphi0);
                    u_pair += kq * ((inv_r - phi0) - plan.Brc + plan.Arc * dist);
                    c_total += kq * (inv_r2 * inv_r + 2.0 * dphi0 - plan.Arc * inv_r);
                }
            }

            u_loc += u_pair;
            const double fx = c_total * dx;
            const double fy = c_total * dy;
            const double fz = c_total * dz;

            if (do_forces) {
                fxi += fx;
                fyi += fy;
                fzi += fz;
                f_buf[3 * c + 0] -= fx;
                f_buf[3 * c + 1] -= fy;
                f_buf[3 * c + 2] -= fz;
            }
            if (do_fixed) {
                const int64_t fx_int = std::llround(fx * FIXED_POINT_SCALE);
                const int64_t fy_int = std::llround(fy * FIXED_POINT_SCALE);
                const int64_t fz_int = std::llround(fz * FIXED_POINT_SCALE);
                fxi_fix += fx_int;
                fyi_fix += fy_int;
                fzi_fix += fz_int;
                fx_buf[3 * c + 0] -= fx_int;
                fx_buf[3 * c + 1] -= fy_int;
                fx_buf[3 * c + 2] -= fz_int;
            }
            if (need_virial) {
                accum_virial9(vir, fx, fy, fz, wx, wy, wz);
            }
        }

        if (do_forces) {
            f_buf[3 * r + 0] += fxi;
            f_buf[3 * r + 1] += fyi;
            f_buf[3 * r + 2] += fzi;
        }
        if (do_fixed) {
            fx_buf[3 * r + 0] += fxi_fix;
            fx_buf[3 * r + 1] += fyi_fix;
            fx_buf[3 * r + 2] += fzi_fix;
        }
    }
}
#endif // __aarch64__ || _M_ARM64

inline void eval_verlet_scaled_range(
    const XbtVerletPlan& plan,
    const double* morton_q,
    bool has_box,
    bool is_triclinic,
    double L0, double L1, double L2,
    double inv_L0, double inv_L1, double inv_L2,
    const double* box_H,
    const double* box_Hinv,
    size_t s_start,
    size_t s_end,
    bool do_forces,
    bool do_fixed,
    bool need_virial,
    double* f_buf,
    int64_t* fx_buf,
    double* vir,
    double& u_loc
) noexcept {
    const VerletScaledPair* sp = plan.scaled_pairs.data();
    const double rc2 = plan.rc2;
    const double rcut = plan.rcut;
    const double inv_rc = plan.inv_rc;
    const double inv_rc2 = plan.inv_rc2;
    const bool use_lfs = plan.use_lfs;
    const bool do_lj_global = plan.do_lj;
    const bool do_coul_global = plan.do_coul;
    const double coul_pref = plan.coulomb_prefactor;
    const double Arc = plan.Arc;
    const double Brc = plan.Brc;
    const double* half_sig = plan.half_sig.data();
    const double* sqrt_eps = plan.sqrt_eps.data();
    const double* chg = plan.chg.data();
    const DsfAnalyticTable& tbl = plan.dsf_table;

    for (size_t k = s_start; k < s_end; ++k) {
        const size_t r = static_cast<size_t>(sp[k].i);
        const size_t c = static_cast<size_t>(sp[k].j);
        const double w_lj = sp[k].w_lj;
        const double w_coul = sp[k].w_coul;

        const double dx0 = morton_q[3 * r + 0] - morton_q[3 * c + 0];
        const double dy0 = morton_q[3 * r + 1] - morton_q[3 * c + 1];
        const double dz0 = morton_q[3 * r + 2] - morton_q[3 * c + 2];

        double dx, dy, dz, wx, wy, wz;
        apply_mic(
            has_box, is_triclinic,
            L0, L1, L2, inv_L0, inv_L1, inv_L2,
            box_H, box_Hinv,
            dx0, dy0, dz0, need_virial,
            dx, dy, dz, wx, wy, wz
        );

        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= rc2 || r2 < 1e-24) continue;

        const double dist = std::sqrt(r2);
        const double inv_r = 1.0 / dist;
        const double inv_r2 = inv_r * inv_r;

        double u_pair = 0.0;
        double c_total = 0.0;

        if (do_lj_global && w_lj != 0.0) {
            const double eij = sqrt_eps[r] * sqrt_eps[c];
            if (eij > 0.0) {
                const double sij = half_sig[r] + half_sig[c];
                if (sij > 0.0) {
                    const double sij2 = sij * sij;
                    const double s2 = sij2 * inv_r2;
                    const double s6 = s2 * s2 * s2;
                    const double s12 = s6 * s6;

                    const double s2c = sij2 * inv_rc2;
                    const double s6c = s2c * s2c * s2c;
                    const double s12c = s6c * s6c;

                    double u_lj = 4.0 * eij * ((s12 - s6) - (s12c - s6c));
                    double c_lj = 24.0 * eij * (2.0 * s12 - s6) * inv_r2;

                    if (use_lfs) {
                        const double duc = 24.0 * eij * (s6c - 2.0 * s12c) * inv_rc;
                        u_lj -= duc * (dist - rcut);
                        c_lj += duc * inv_r;
                    }

                    u_pair += w_lj * u_lj;
                    c_total += w_lj * c_lj;
                }
            }
        }

        if (do_coul_global && w_coul != 0.0) {
            const double qiqj = chg[r] * chg[c];
            if (qiqj != 0.0) {
                const double kq = w_coul * coul_pref * qiqj;
                double phi0 = 0.0, dphi0 = 0.0;
                tbl.lookup(r2, phi0, dphi0);
                const double erfc_over_r = inv_r - phi0;
                const double inv_r3 = inv_r2 * inv_r;
                u_pair += kq * (erfc_over_r - Brc + Arc * dist);
                c_total += kq * (inv_r3 + 2.0 * dphi0 - Arc * inv_r);
            }
        }

        u_loc += u_pair;
        const double fx = c_total * dx;
        const double fy = c_total * dy;
        const double fz = c_total * dz;

        if (do_forces) {
            f_buf[3 * r + 0] += fx;
            f_buf[3 * r + 1] += fy;
            f_buf[3 * r + 2] += fz;
            f_buf[3 * c + 0] -= fx;
            f_buf[3 * c + 1] -= fy;
            f_buf[3 * c + 2] -= fz;
        }
        if (do_fixed) {
            const int64_t fx_int = std::llround(fx * FIXED_POINT_SCALE);
            const int64_t fy_int = std::llround(fy * FIXED_POINT_SCALE);
            const int64_t fz_int = std::llround(fz * FIXED_POINT_SCALE);
            fx_buf[3 * r + 0] += fx_int;
            fx_buf[3 * r + 1] += fy_int;
            fx_buf[3 * r + 2] += fz_int;
            fx_buf[3 * c + 0] -= fx_int;
            fx_buf[3 * c + 1] -= fy_int;
            fx_buf[3 * c + 2] -= fz_int;
        }
        if (need_virial) {
            accum_virial9(vir, fx, fy, fz, wx, wy, wz);
        }
    }
}

} // anonymous namespace

XBT_EXPORT void* xbt_verlet_plan_create(
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

    auto plan = std::make_unique<XbtVerletPlan>();
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
        plan->two_alpha_sqrt_pi = 2.0 * alpha_dsf / SQRT_PI;
        const double z_rc = alpha_dsf * rcut;
        const double erfc_rc = std::erfc(z_rc);
        const double exp_rc = std::exp(-alpha_dsf * alpha_dsf * plan->rc2);
        plan->Arc = erfc_rc * plan->inv_rc2 + plan->two_alpha_sqrt_pi * exp_rc * plan->inv_rc;
        plan->Brc = erfc_rc * plan->inv_rc + plan->Arc * rcut;
        plan->dsf_table.init(rcut, alpha_dsf);
    }

    plan->morton_to_atom.resize(n_atoms, -1);
    plan->half_sig.resize(n_atoms, 0.0);
    plan->sqrt_eps.resize(n_atoms, 0.0);
    plan->chg.resize(n_atoms, 0.0);
    plan->half_sig_f.resize(n_atoms, 0.0f);
    plan->sqrt_eps_f.resize(n_atoms, 0.0f);
    plan->chg_f.resize(n_atoms, 0.0f);
    plan->row_flags.resize(n_atoms, 0);
    plan->morton_q.resize(3 * n_atoms, 0.0);
    plan->csr_row_ptr.assign(n_atoms + 1, 0);

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
        const size_t ua = static_cast<size_t>(atom0);

        plan->morton_q[3 * r + 0] = q[3 * ua + 0];
        plan->morton_q[3 * r + 1] = q[3 * ua + 1];
        plan->morton_q[3 * r + 2] = q[3 * ua + 2];

        uint8_t flags = 0;
        if (plan->do_lj) {
            const double hs = 0.5 * sigma[ua];
            const double eps_val = epsilon[ua];
            const double se = (eps_val > 0.0) ? std::sqrt(eps_val) : 0.0;
            plan->half_sig[r] = hs;
            plan->sqrt_eps[r] = se;
            plan->half_sig_f[r] = static_cast<float>(hs);
            plan->sqrt_eps_f[r] = static_cast<float>(se);
            if (se > 0.0) {
                flags |= XbtVerletPlan::FLAG_HAS_LJ;
            }
        }
        if (plan->do_coul) {
            const double c_val = charge[ua];
            plan->chg[r] = c_val;
            plan->chg_f[r] = static_cast<float>(c_val);
            if (c_val != 0.0) {
                flags |= XbtVerletPlan::FLAG_HAS_COUL;
            }
        }
        plan->row_flags[r] = flags;
    }
    plan->order_is_perm = is_perm;

    if (n_tile_pairs == 0 || !tile_i || !tile_j || !order || !tile_starts || !tile_ends) {
        return plan.release();
    }

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

    XbtPersistentPool& pool = XbtPersistentPool::instance();
    int T = (n_tile_pairs >= 128) ? std::min(pool.max_threads(), static_cast<int>(n_tile_pairs)) : 1;

    struct EdgePair {
        int32_t r;
        int32_t c;
    };
    std::vector<std::vector<EdgePair>> local_edges(static_cast<size_t>(T));
    std::vector<std::vector<VerletScaledPair>> local_scaled(static_cast<size_t>(T));

    struct BuildCtx {
        const XbtVerletPlan* plan;
        size_t n_atoms;
        size_t n_tile_pairs;
        const int32_t* tile_i;
        const int32_t* tile_j;
        const int32_t* order;
        const int32_t* tile_starts;
        const int32_t* tile_ends;
        bool has_box;
        bool tri;
        double L0, L1, L2;
        double inv_L0, inv_L1, inv_L2;
        const double* box_H;
        const double* box_Hinv;
        double r_list2;
        bool check_topo;
        bool has_excl;
        size_t n_excl;
        const uint64_t* excl_keys;
        bool has_scaled;
        size_t n_scaled;
        const uint64_t* scaled_keys;
        const double* scaled_weights;
        double lj14_scale;
        double coul14_scale;
        std::vector<std::vector<EdgePair>>* local_edges;
        std::vector<std::vector<VerletScaledPair>>* local_scaled;
    } ctx{
        plan.get(), n_atoms, n_tile_pairs, tile_i, tile_j, order, tile_starts, tile_ends,
        has_box, tri, L0, L1, L2, inv_L0, inv_L1, inv_L2, box_H, box_Hinv, r_list2,
        check_topo, has_excl, n_excl, excl_keys, has_scaled, n_scaled, scaled_keys,
        scaled_weights, lj14_scale, coul14_scale, &local_edges, &local_scaled
    };

    pool.run(T, [&ctx](int t, int n_t) {
        const size_t start_k = (ctx.n_tile_pairs * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
        const size_t end_k = (ctx.n_tile_pairs * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
        auto& edges = (*ctx.local_edges)[static_cast<size_t>(t)];
        auto& scaled = (*ctx.local_scaled)[static_cast<size_t>(t)];
        const double* mq = ctx.plan->morton_q.data();

        auto test_pair = [&](int32_t a_pos0, int32_t b_pos0, int32_t i1, double qxi, double qyi, double qzi) {
            const int32_t j1 = ctx.order[b_pos0];
            if (j1 < 1 || static_cast<size_t>(j1) > ctx.n_atoms || b_pos0 == a_pos0) return;

            const size_t ub = static_cast<size_t>(b_pos0);
            const double dx0 = qxi - mq[3 * ub + 0];
            const double dy0 = qyi - mq[3 * ub + 1];
            const double dz0 = qzi - mq[3 * ub + 2];

            double dx, dy, dz, wx, wy, wz;
            apply_mic(
                ctx.has_box, ctx.tri,
                ctx.L0, ctx.L1, ctx.L2, ctx.inv_L0, ctx.inv_L1, ctx.inv_L2,
                ctx.box_H, ctx.box_Hinv,
                dx0, dy0, dz0, false,
                dx, dy, dz, wx, wy, wz
            );
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= ctx.r_list2) return;

            const int32_t lo_pos = (a_pos0 < b_pos0) ? a_pos0 : b_pos0;
            const int32_t hi_pos = (a_pos0 < b_pos0) ? b_pos0 : a_pos0;

            if (ctx.check_topo) {
                const uint64_t lo1 = static_cast<uint64_t>(i1 < j1 ? i1 : j1);
                const uint64_t hi1 = static_cast<uint64_t>(i1 < j1 ? j1 : i1);
                const uint64_t key = (lo1 << 32) | hi1;
                if (ctx.has_excl && std::binary_search(ctx.excl_keys, ctx.excl_keys + ctx.n_excl, key)) {
                    return;
                }
                if (ctx.has_scaled) {
                    const uint64_t* it = std::lower_bound(ctx.scaled_keys, ctx.scaled_keys + ctx.n_scaled, key);
                    if (it != ctx.scaled_keys + ctx.n_scaled && *it == key) {
                        const size_t pos = static_cast<size_t>(it - ctx.scaled_keys);
                        const double w = ctx.scaled_weights ? ctx.scaled_weights[pos] : ctx.lj14_scale;
                        const double w_lj = (w != 1.0) ? w : ctx.lj14_scale;
                        const double w_coul = ctx.coul14_scale;
                        if (w_lj != 0.0 || w_coul != 0.0) {
                            scaled.push_back(VerletScaledPair{lo_pos, hi_pos, w_lj, w_coul});
                        }
                        return;
                    }
                }
            }

            edges.push_back(EdgePair{lo_pos, hi_pos});
        };

        for (size_t k = start_k; k < end_k; ++k) {
            const int32_t ti1 = ctx.tile_i[k];
            const int32_t tj1 = ctx.tile_j[k];
            const int32_t si1 = ctx.tile_starts[ti1 - 1];
            const int32_t ei1 = ctx.tile_ends[ti1 - 1];

            if (ti1 == tj1) {
                for (int32_t a = si1; a < ei1; ++a) {
                    const int32_t a_pos0 = a - 1;
                    const int32_t i1 = ctx.order[a_pos0];
                    if (i1 < 1 || static_cast<size_t>(i1) > ctx.n_atoms) continue;
                    const size_t ua = static_cast<size_t>(a_pos0);
                    const double qxi = mq[3 * ua + 0];
                    const double qyi = mq[3 * ua + 1];
                    const double qzi = mq[3 * ua + 2];
                    for (int32_t b = a + 1; b <= ei1; ++b) {
                        test_pair(a_pos0, b - 1, i1, qxi, qyi, qzi);
                    }
                }
            } else {
                const int32_t sj1 = ctx.tile_starts[tj1 - 1];
                const int32_t ej1 = ctx.tile_ends[tj1 - 1];
                for (int32_t a = si1; a <= ei1; ++a) {
                    const int32_t a_pos0 = a - 1;
                    const int32_t i1 = ctx.order[a_pos0];
                    if (i1 < 1 || static_cast<size_t>(i1) > ctx.n_atoms) continue;
                    const size_t ua = static_cast<size_t>(a_pos0);
                    const double qxi = mq[3 * ua + 0];
                    const double qyi = mq[3 * ua + 1];
                    const double qzi = mq[3 * ua + 2];
                    for (int32_t b = sj1; b <= ej1; ++b) {
                        test_pair(a_pos0, b - 1, i1, qxi, qyi, qzi);
                    }
                }
            }
        }
    });

    // Assemble CSR rows in Morton order
    size_t total_edges = 0;
    size_t total_scaled = 0;
    for (int t = 0; t < T; ++t) {
        total_edges += local_edges[static_cast<size_t>(t)].size();
        total_scaled += local_scaled[static_cast<size_t>(t)].size();
        for (const EdgePair& ep : local_edges[static_cast<size_t>(t)]) {
            plan->csr_row_ptr[static_cast<size_t>(ep.r) + 1]++;
        }
    }

    for (size_t r = 0; r < n_atoms; ++r) {
        plan->csr_row_ptr[r + 1] += plan->csr_row_ptr[r];
    }

    plan->csr_col_idx.resize(total_edges);
    std::vector<size_t> write_pos(plan->csr_row_ptr.begin(), plan->csr_row_ptr.begin() + n_atoms);
    for (int t = 0; t < T; ++t) {
        for (const EdgePair& ep : local_edges[static_cast<size_t>(t)]) {
            const size_t pos = write_pos[static_cast<size_t>(ep.r)]++;
            plan->csr_col_idx[pos] = ep.c;
        }
    }

    // Sort neighbor indices in ascending Morton order within each CSR row for monotonic streaming
    if (total_edges > 0) {
        struct SortCtx {
            XbtVerletPlan* plan;
        } sctx{plan.get()};
        const int T_sort = (n_atoms >= 512) ? std::min(pool.max_threads(), static_cast<int>(n_atoms)) : 1;
        pool.run(T_sort, [&sctx](int t, int n_t) {
            const size_t r_start = (sctx.plan->n_atoms * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
            const size_t r_end = (sctx.plan->n_atoms * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
            const size_t* rp = sctx.plan->csr_row_ptr.data();
            int32_t* ci = sctx.plan->csr_col_idx.data();
            for (size_t r = r_start; r < r_end; ++r) {
                const size_t b = rp[r];
                const size_t e = rp[r + 1];
                if (e - b > 1) {
                    std::sort(ci + b, ci + e);
                }
            }
        });
    }

    plan->scaled_pairs.reserve(total_scaled);
    for (int t = 0; t < T; ++t) {
        const auto& ls = local_scaled[static_cast<size_t>(t)];
        plan->scaled_pairs.insert(plan->scaled_pairs.end(), ls.begin(), ls.end());
    }

    return plan.release();
}

XBT_EXPORT double xbt_verlet_plan_eval_mt(
    void* plan_ptr,
    const double* q,
    const double* box_H,
    const double* box_Hinv,
    int is_triclinic,
    int use_fast_math,
    int n_threads,
    double* out_forces,
    int64_t* out_forces_fixed,
    double* out_virial9
) {
    if (!plan_ptr) return 0.0;
    XbtVerletPlan& plan = *static_cast<XbtVerletPlan*>(plan_ptr);
    const size_t n_atoms = plan.n_atoms;
    zero_output_buffers(n_atoms, out_forces, out_forces_fixed, out_virial9);

    if (!q || n_atoms == 0) return 0.0;
    const size_t n_csr = plan.csr_col_idx.size();
    const size_t n_scaled = plan.scaled_pairs.size();
    if (n_csr == 0 && n_scaled == 0) return 0.0;

    std::lock_guard<std::mutex> plan_lock(plan.eval_mtx);

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

    // Permute coordinates into Morton order once (O(N), zero heap allocation)
    double* mq = plan.morton_q.data();
    const int32_t* m2a = plan.morton_to_atom.data();
    for (size_t r = 0; r < n_atoms; ++r) {
        const int32_t atom = m2a[r];
        if (atom >= 0 && static_cast<size_t>(atom) < n_atoms) {
            const size_t ua = static_cast<size_t>(atom);
            mq[3 * r + 0] = q[3 * ua + 0];
            mq[3 * r + 1] = q[3 * ua + 1];
            mq[3 * r + 2] = q[3 * ua + 2];
        }
    }

    XbtPersistentPool& pool = XbtPersistentPool::instance();
    const size_t total_pairs = n_csr + n_scaled;
    int T = std::min(resolve_thread_count(n_threads), pool.max_threads());
    if (total_pairs < 4096) {
        T = 1;
    }

    const bool do_forces = (out_forces != nullptr);
    const bool do_fixed = (out_forces_fixed != nullptr);
    const bool need_vir = (out_virial9 != nullptr && has_box);
    const size_t f_len = do_forces ? 3 * n_atoms : 0;
    const size_t fx_len = do_fixed ? 3 * n_atoms : 0;
    pool.prepare_scratch(T, f_len, fx_len, need_vir);

    // Balance CSR row ranges across threads by cumulative edge count
    std::array<size_t, XbtPersistentPool::MAX_TOTAL_THREADS + 1> row_bounds{};
    row_bounds[0] = 0;
    row_bounds[static_cast<size_t>(T)] = n_atoms;
    if (T > 1) {
        const auto& rp = plan.csr_row_ptr;
        for (int t = 1; t < T; ++t) {
            const size_t target = (n_csr * static_cast<size_t>(t)) / static_cast<size_t>(T);
            auto it = std::lower_bound(rp.begin(), rp.end(), target);
            size_t r_bound = static_cast<size_t>(it - rp.begin());
            if (r_bound > n_atoms) r_bound = n_atoms;
            row_bounds[static_cast<size_t>(t)] = r_bound;
        }
    }

    const bool fast_neon = (use_fast_math != 0) && !tri;

    struct VerletEvalCtx {
        XbtPersistentPool* pool;
        const XbtVerletPlan* plan;
        const double* morton_q;
        bool has_box;
        bool tri;
        double L0, L1, L2;
        double inv_L0, inv_L1, inv_L2;
        const double* box_H;
        const double* box_Hinv;
        bool fast_neon;
        bool do_forces;
        bool do_fixed;
        bool need_vir;
        const size_t* row_bounds;
        size_t n_scaled;
    } ctx{
        &pool, &plan, mq, has_box, tri,
        L0, L1, L2, inv_L0, inv_L1, inv_L2,
        box_H, box_Hinv, fast_neon, do_forces, do_fixed, need_vir,
        row_bounds.data(), n_scaled
    };

    pool.run(T, [&ctx](int t, int n_t) {
        WorkerScratch& acc = ctx.pool->scratch(t);
        double* f_ptr = ctx.do_forces ? acc.f.data() : nullptr;
        int64_t* fx_ptr = ctx.do_fixed ? acc.fx.data() : nullptr;
        double* vir_ptr = ctx.need_vir ? acc.vir : nullptr;
        double u_loc = 0.0;

        const size_t r_start = ctx.row_bounds[static_cast<size_t>(t)];
        const size_t r_end = ctx.row_bounds[static_cast<size_t>(t + 1)];
        if (r_start < r_end) {
#if defined(__aarch64__) || defined(_M_ARM64)
            if (ctx.fast_neon) {
                eval_verlet_csr_neon_row_range(
                    *ctx.plan, ctx.morton_q, ctx.has_box,
                    ctx.L0, ctx.L1, ctx.L2, ctx.inv_L0, ctx.inv_L1, ctx.inv_L2,
                    r_start, r_end, ctx.do_forces, ctx.do_fixed, ctx.need_vir,
                    f_ptr, fx_ptr, vir_ptr, u_loc
                );
            } else
#endif
            {
                eval_verlet_csr_exact_row_range(
                    *ctx.plan, ctx.morton_q, ctx.has_box, ctx.tri,
                    ctx.L0, ctx.L1, ctx.L2, ctx.inv_L0, ctx.inv_L1, ctx.inv_L2,
                    ctx.box_H, ctx.box_Hinv,
                    r_start, r_end, ctx.do_forces, ctx.do_fixed, ctx.need_vir,
                    f_ptr, fx_ptr, vir_ptr, u_loc
                );
            }
        }

        if (ctx.n_scaled > 0) {
            const size_t s_start = (ctx.n_scaled * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
            const size_t s_end = (ctx.n_scaled * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
            if (s_start < s_end) {
                eval_verlet_scaled_range(
                    *ctx.plan, ctx.morton_q, ctx.has_box, ctx.tri,
                    ctx.L0, ctx.L1, ctx.L2, ctx.inv_L0, ctx.inv_L1, ctx.inv_L2,
                    ctx.box_H, ctx.box_Hinv,
                    s_start, s_end, ctx.do_forces, ctx.do_fixed, ctx.need_vir,
                    f_ptr, fx_ptr, vir_ptr, u_loc
                );
            }
        }

        acc.u = u_loc;
    });

    double total_U = 0.0;
    for (int t = 0; t < T; ++t) {
        const WorkerScratch& acc = pool.scratch(t);
        total_U += acc.u;
        if (need_vir) {
            for (int v = 0; v < 9; ++v) {
                out_virial9[v] += acc.vir[v];
            }
        }
    }

    if (do_forces || do_fixed) {
        auto scatter_range = [&](size_t r_start, size_t r_end) {
            for (size_t r = r_start; r < r_end; ++r) {
                const int32_t atom = m2a[r];
                if (atom < 0 || static_cast<size_t>(atom) >= n_atoms) continue;
                const size_t ua = static_cast<size_t>(atom);
                if (do_forces) {
                    double sx = 0.0, sy = 0.0, sz = 0.0;
                    for (int t = 0; t < T; ++t) {
                        const double* tf = pool.scratch(t).f.data();
                        sx += tf[3 * r + 0];
                        sy += tf[3 * r + 1];
                        sz += tf[3 * r + 2];
                    }
                    out_forces[3 * ua + 0] += sx;
                    out_forces[3 * ua + 1] += sy;
                    out_forces[3 * ua + 2] += sz;
                }
                if (do_fixed) {
                    int64_t sx = 0, sy = 0, sz = 0;
                    for (int t = 0; t < T; ++t) {
                        const int64_t* tfx = pool.scratch(t).fx.data();
                        sx += tfx[3 * r + 0];
                        sy += tfx[3 * r + 1];
                        sz += tfx[3 * r + 2];
                    }
                    out_forces_fixed[3 * ua + 0] += sx;
                    out_forces_fixed[3 * ua + 1] += sy;
                    out_forces_fixed[3 * ua + 2] += sz;
                }
            }
        };

        if (plan.order_is_perm && n_atoms >= 2048 && T > 1) {
            struct ScatterCtx {
                decltype(scatter_range)* fn;
                size_t n_atoms;
            } sctx{&scatter_range, n_atoms};

            pool.run(T, [&sctx](int t, int n_t) {
                const size_t r_start = (sctx.n_atoms * static_cast<size_t>(t)) / static_cast<size_t>(n_t);
                const size_t r_end = (sctx.n_atoms * static_cast<size_t>(t + 1)) / static_cast<size_t>(n_t);
                (*sctx.fn)(r_start, r_end);
            });
        } else {
            scatter_range(0, n_atoms);
        }
    }

    return total_U;
}

XBT_EXPORT void xbt_verlet_plan_destroy(void* plan_ptr) {
    if (!plan_ptr) return;
    delete static_cast<XbtVerletPlan*>(plan_ptr);
}

XBT_EXPORT size_t xbt_verlet_plan_num_pairs(void* plan_ptr) {
    if (!plan_ptr) return 0;
    const XbtVerletPlan* plan = static_cast<const XbtVerletPlan*>(plan_ptr);
    return plan->csr_col_idx.size() + plan->scaled_pairs.size();
}
