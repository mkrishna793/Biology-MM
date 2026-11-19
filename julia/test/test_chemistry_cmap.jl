# CMAP correction factor: interpolation accuracy, finite-difference forces,
# 2π periodicity, patch continuity, map index convention, degenerate geometry.
#
# These are mathematical checks of the implemented periodic bicubic map, not a
# force-field validation: the analytic map below is a stand-in for a tabulated
# AMBER/CHARMM CMAP.

using FoundationCore
using Random

const _CT = FoundationCore.Types
const _CF = FoundationCore.Factors

# The module is either registered in FoundationCore, or already `include`d into
# Main by the run command, or not included at all — in that last case include it
# here; `using FoundationCore` already exported `Types`/`Factors` into Main, so
# its `using ..Types` / `using ..Factors` resolve to the real parent modules.
if !isdefined(FoundationCore, :ChemistryCMAP) && !isdefined(Main, :ChemistryCMAP)
    Base.include(Main, joinpath(@__DIR__, "..", "src", "ChemistryCMAP.jl"))
end

const _CM = isdefined(FoundationCore, :ChemistryCMAP) ? FoundationCore.ChemistryCMAP :
            Main.ChemistryCMAP

"Analytic stand-in for a tabulated map: smooth and 2π-periodic in both angles."
map_analytic(phi, psi) = 2.0 * cos(phi) + 1.5 * sin(2 * psi) + 0.7 * cos(phi - psi)

function map_analytic_grad(phi, psi)
    dphi = -2.0 * sin(phi) - 0.7 * sin(phi - psi)
    dpsi = 3.0 * cos(2 * psi) + 0.7 * sin(phi - psi)
    return dphi, dpsi
end

"Sample `f(φ,ψ)` on the n×n CMAP grid; row index is φ, column index is ψ."
function sample_grid(f, n)
    delta = 2 * pi / n
    grid = Matrix{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        grid[i, j] = f((i - 1) * delta, (j - 1) * delta)
    end
    return grid
end

function check(cond, msg)
    cond || error("FAIL: " * msg)
    return nothing
end

"Rotate `atom` about the axis `axis_i → axis_j` by `theta` (rigid, angle-only)."
function rotate_about!(q, axis_i, axis_j, atom, theta)
    ax = [q[1, axis_j] - q[1, axis_i], q[2, axis_j] - q[2, axis_i],
          q[3, axis_j] - q[3, axis_i]]
    ax ./= sqrt(sum(abs2, ax))
    K = [0.0 -ax[3] ax[2]; ax[3] 0.0 -ax[1]; -ax[2] ax[1] 0.0]
    R = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] + sin(theta) * K +
        (1 - cos(theta)) * (K * K)
    origin = [q[1, axis_j], q[2, axis_j], q[3, axis_j]]
    v = [q[1, atom], q[2, atom], q[3, atom]] - origin
    w = origin + R * v
    q[1, atom], q[2, atom], q[3, atom] = w[1], w[2], w[3]
    return q
end

function main()
    println("="^72)
    println("CMAP correction tests")
    println("="^72)

    rng = MersenneTwister(0x5eed)
    n = 36
    delta = 2 * pi / n
    atoms = (1, 2, 3, 4, 5, 6, 7, 8)

    grid = sample_grid(map_analytic, n)
    cmap = _CM.CMAPCorrection(atoms, grid)
    check(size(cmap.coeffs) == (4, 4, n, n), "coefficient array must be 4×4×n×n")
    check(_CF.factor_support_size(cmap) == 8, "support size must be 8")
    check(_CF.read_support(cmap) == atoms, "read_support must be the 8 atoms")

    # --- the patch interpolates the tabulated nodes exactly ------------------
    nodal_err = 0.0
    for j in 1:n, i in 1:n
        e, _, _ = _CM.cmap_energy_grad(cmap, (i - 1) * delta, (j - 1) * delta)
        nodal_err = max(nodal_err, abs(e - grid[i, j]))
    end
    check(nodal_err <= 1e-12, "patch must reproduce grid nodes, err=$nodal_err")

    # --- interpolation and angle derivatives against the analytic map -------
    npts = 400
    int_err = 0.0
    dphi_err = 0.0
    dpsi_err = 0.0
    for _ in 1:npts
        phi = 2 * pi * rand(rng)
        psi = 2 * pi * rand(rng)
        e, dphi, dpsi = _CM.cmap_energy_grad(cmap, phi, psi)
        adphi, adpsi = map_analytic_grad(phi, psi)
        int_err = max(int_err, abs(e - map_analytic(phi, psi)))
        dphi_err = max(dphi_err, abs(dphi - adphi))
        dpsi_err = max(dpsi_err, abs(dpsi - adpsi))
    end
    check(int_err <= 2e-3, "interpolation error $int_err > 2e-3 kJ/mol")
    check(dphi_err <= 2e-3, "∂E/∂φ error $dphi_err > 2e-3")
    check(dpsi_err <= 2e-3, "∂E/∂ψ error $dpsi_err > 2e-3")

    # --- finite-difference forces on a non-collinear geometry --------------
    q0 = [0.0 1.5 2.3 3.0 0.0 1.4 2.2 3.1;
          0.0 0.0 1.1 1.0 2.0 2.2 3.0 3.2;
          0.0 0.0 0.25 1.4 0.5 0.3 0.9 0.2]
    ctx = _CF.Context(_CT.NoBox(), _CT.REDUCED, 8)
    ta = _CF.dihedral_angle_grad(q0, _CT.NoBox(), 1, 2, 3, 4)
    tb = _CF.dihedral_angle_grad(q0, _CT.NoBox(), 5, 6, 7, 8)
    check(ta !== nothing && tb !== nothing, "fixture geometry must be non-collinear")
    _, dph0, dps0 = _CM.cmap_energy_grad(cmap, ta[1], tb[1])
    check(abs(dph0) > 1e-2 && abs(dps0) > 1e-2,
          "fixture must load both map axes (dE/dφ=$dph0, dE/dψ=$dps0)")

    acc = zeros(3, 8)
    u0 = _CF.evaluate!(acc, cmap, q0, ctx)
    check(isfinite(u0), "energy must be finite on the fixture geometry")
    check(maximum(abs, acc) > 1e-3, "fixture must produce non-trivial forces")

    h = 1e-6
    fd_err = 0.0
    fd_scale = 1.0
    for j in 1:8, d in 1:3
        qp = copy(q0); qm = copy(q0)
        qp[d, j] += h; qm[d, j] -= h
        up = _CF.evaluate!(zeros(3, 8), cmap, qp, ctx)
        um = _CF.evaluate!(zeros(3, 8), cmap, qm, ctx)
        fdc = -(up - um) / (2h)
        fd_err = max(fd_err, abs(fdc - acc[d, j]))
        fd_scale = max(fd_scale, abs(fdc), abs(acc[d, j]))
    end
    check(fd_err <= 1e-6 * fd_scale, "FD force error $fd_err > 1e-6×$fd_scale")

    # --- allocation discipline ---------------------------------------------
    # The patch evaluation owns no allocation at all. `evaluate!` additionally
    # pays for the boxed union that the shared `Factors.dihedral_angle_grad`
    # returns (112 B per torsion), which the existing torsion factors pay too.
    _CF.evaluate!(acc, cmap, q0, ctx)          # warm up
    su = @allocated _CF.evaluate!(acc, cmap, q0, ctx)
    sm = @allocated _CM.cmap_energy_grad(cmap, 1.0, 2.0)
    pdref = _CF.PeriodicDihedral(1, 2, 3, 4, 1.0, 2, 0.3)
    _CF.evaluate!(acc, pdref, q0, ctx)
    spd = @allocated _CF.evaluate!(acc, pdref, q0, ctx)
    check(sm == 0, "patch evaluation must allocate nothing (got $sm bytes)")
    check(su <= 2 * spd,
          "CMAP evaluation allocated $su B, more than two dihedral factors ($(2 * spd) B)")

    # --- 2π invariance: angle space and geometry space ----------------------
    per_err = 0.0
    for _ in 1:50
        phi = 2 * pi * rand(rng) - pi
        psi = 2 * pi * rand(rng) - pi
        e, dphi, dpsi = _CM.cmap_energy_grad(cmap, phi, psi)
        ep, dp1, _ = _CM.cmap_energy_grad(cmap, phi + 2 * pi, psi)
        eq, _, dq2 = _CM.cmap_energy_grad(cmap, phi, psi + 2 * pi)
        em, _, _ = _CM.cmap_energy_grad(cmap, phi - 2 * pi, psi - 2 * pi)
        per_err = max(per_err, abs(ep - e), abs(eq - e), abs(em - e),
                      abs(dp1 - dphi), abs(dq2 - dpsi))
    end
    check(per_err <= 1e-12, "2π shift changed energy/derivative by $per_err")

    # Rotating the terminal atom of either torsion by 2π must not move anything.
    acc0 = zeros(3, 8)
    u0g = _CF.evaluate!(acc0, cmap, q0, ctx)
    qr = rotate_about!(copy(q0), 2, 3, 4, 2 * pi)
    qr = rotate_about!(qr, 6, 7, 8, 2 * pi)
    accr = zeros(3, 8)
    ur = _CF.evaluate!(accr, cmap, qr, ctx)
    geo_err = max(abs(ur - u0g), maximum(abs, accr - acc0))
    check(geo_err <= 1e-12, "2π rotation changed energy/forces by $geo_err")

    # --- patch boundaries and the 2π seam ----------------------------------
    seam_err = 0.0
    for node in (1, 4, n - 1, n)
        phi = (node - 1) * delta
        e1v, _, _ = _CM.cmap_energy_grad(cmap, phi - 1e-9, 0.7)
        e2v, _, _ = _CM.cmap_energy_grad(cmap, phi + 1e-9, 0.7)
        check(isfinite(e1v) && isfinite(e2v), "patch boundary must stay finite")
        jump = abs(e1v - e2v)
        seam_err = max(seam_err, jump)
        check(jump <= 1e-6, "patch jump at φ=$phi is $jump")
    end
    # Straddle 2π: φ = 2π-ε must join φ = 0+ε, and the atan2 cut -π / +π must
    # join as well (wrapping maps -π+ε to π+ε).
    for psi in (0.0, 1.3, 4.9)
        ea, _, _ = _CM.cmap_energy_grad(cmap, 2 * pi - 1e-9, psi)
        eb, _, _ = _CM.cmap_energy_grad(cmap, 1e-9, psi)
        ec, _, _ = _CM.cmap_energy_grad(cmap, -pi + 1e-9, psi)
        ed, _, _ = _CM.cmap_energy_grad(cmap, pi - 1e-9, psi)
        seam_err = max(seam_err, abs(ea - eb), abs(ec - ed))
        check(abs(ea - eb) <= 1e-6, "2π seam jump $(abs(ea - eb))")
        check(abs(ec - ed) <= 1e-6, "atan2 cut jump $(abs(ec - ed))")
    end

    # --- map index convention ---------------------------------------------
    # Constant in ψ, cos-shaped in φ: an axis permutation returns cos(ψ) instead.
    gridc = Matrix{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        gridc[i, j] = 3.0 * cos((i - 1) * delta)
    end
    cosmap = _CM.CMAPCorrection(atoms, gridc)
    cphi = 0.3
    cpsi = 2.0
    ec, _, _ = _CM.cmap_energy_grad(cosmap, cphi, cpsi)
    cos_err = abs(ec - 3.0 * cos(cphi))
    check(cos_err <= 2e-3, "cos-map interpolation error $cos_err > 2e-3")

    swapmap = _CM.CMAPCorrection(atoms, transpose(gridc))
    es, _, _ = _CM.cmap_energy_grad(swapmap, cphi, cpsi)
    swap_err = abs(es - 3.0 * cos(cphi))
    check(swap_err > 1e-3,
          "transposed grid was not distinguishable (err=$swap_err): φ/ψ axes look swapped")

    # --- degenerate geometry ----------------------------------------------
    qline = zeros(3, 8)
    for i in 1:8
        qline[1, i] = 0.3 * (i - 1)
    end
    accd = zeros(3, 8)
    ud = _CF.evaluate!(accd, cmap, qline, ctx)
    check(ud == Inf, "fully collinear geometry must return Inf")
    check(all(iszero, accd), "degenerate geometry must scatter no force")

    qhalf = copy(q0)
    for i in 1:4                        # first torsion collinear, second intact
        qhalf[1, i] = 0.4 * (i - 1)
        qhalf[2, i] = 0.0
        qhalf[3, i] = 0.0
    end
    fill!(accd, 0.0)
    uh = _CF.evaluate!(accd, cmap, qhalf, ctx)
    check(uh == Inf, "collinear first torsion must return Inf")
    check(all(iszero, accd), "collinear torsion must scatter no force")

    # --- constructor hygiene ----------------------------------------------
    bad = try
        _CM.CMAPCorrection(atoms, zeros(3, 4))
        false
    catch err
        err isa ArgumentError
    end
    check(bad, "non-square grid must be rejected")
    bad2 = try
        _CM.CMAPCorrection((1, 2, 3), grid)
        false
    catch err
        err isa ArgumentError
    end
    check(bad2, "wrong atom count must be rejected")

    println("  nodal interpolation:      $nodal_err")
    println("  map interpolation:        $int_err  (<= 2e-3)")
    println("  ∂E/∂φ, ∂E/∂ψ:             $dphi_err, $dpsi_err  (<= 2e-3)")
    println("  FD force (h=1e-6):        $fd_err  rel $(fd_err / fd_scale)  (<= 1e-6)")
    println("  2π angle shift:           $per_err  (<= 1e-12)")
    println("  2π geometry rotation:     $geo_err  (<= 1e-12)")
    println("  patch/seam jump:          $seam_err  (<= 1e-6)")
    println("  cos-map error:            $cos_err  (<= 2e-3)")
    println("  transposed-grid error:    $swap_err  (> 1e-3)")
    println("  allocations:              $su / $sm bytes")
    println("CMAP tests PASSED")
    return nothing
end

main()
