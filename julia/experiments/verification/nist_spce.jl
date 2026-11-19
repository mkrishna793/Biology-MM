using FoundationCore
using LinearAlgebra
using Printf

const F = FoundationCore.Factors
const T = FoundationCore.Types
const E = FoundationCore.Electrostatics
const P = FoundationCore.PME
const PC = P.PairCorrections
const KCOUL_K_A = 1.0e10 * (1.602176565e-19)^2 /
                  (4pi * 8.854187817e-12 * 1.3806488e-23)
erf64(x::Float64) = ccall(:erf, Cdouble, (Cdouble,), x)
erfc64(x::Float64) = ccall(:erfc, Cdouble, (Cdouble,), x)

function load_config(path)
    lines = readlines(path)
    L = parse.(Float64, split(strip(lines[1])))
    nw = parse(Int, strip(lines[2]))
    q = zeros(3, 3nw)
    types = String[]
    for (i, line) in enumerate(lines[3:3+3nw-1])
        fields = split(strip(line))
        q[:, i] .= parse.(Float64, fields[2:4])
        push!(types, fields[5])
    end
    charges = [x == "O" ? -0.8476 : 0.4238 for x in types]
    return L, nw, q, types, charges
end

function run()
    path = get(ENV, "SPCE_CONFIG", "")
    isempty(path) && error("set SPCE_CONFIG to the extracted NIST configuration")
    L, nw, q, types, charges = load_config(path)
    alpha = 5.6 / minimum(L)
    rc = prevfloat(minimum(L) / 2)
    sigma = 3.16555789
    epsilon = 78.19743111
    n_o = findall(==("O"), types)
    Ulj = 0.0
    for a in 1:length(n_o)-1, b in a+1:length(n_o)
        i, j = n_o[a], n_o[b]
        d = q[:, i] - q[:, j]
        d .-= L .* round.(d ./ L)
        r = norm(d)
        r < 10.0 || continue
        sr6 = (sigma / r)^6
        Ulj += 4epsilon * (sr6^2 - sr6)
    end
    rho = nw / prod(L)
    lrc = 8pi * nw^2 * epsilon * (sigma^12 / (9 * 10.0^9) - sigma^6 / (3 * 10.0^3)) / prod(L)
    pref = KCOUL_K_A
    Ereal = 0.0
    Eintra_real = 0.0
    for i in 1:length(charges)-1, j in i+1:length(charges)
        d = q[:, i] - q[:, j]
        d .-= L .* round.(d ./ L)
        r = norm(d)
        if r < 10.0
            term = pref * charges[i] * charges[j] * erfc64(alpha * r) / r
            Ereal += term
            if (i - 1) ÷ 3 == (j - 1) ÷ 3
                Eintra_real += term
            end
        end
    end
    Efourier_sphere = 0.0
    for nx in -5:5, ny in -5:5, nz in -5:5
        nx == ny == nz == 0 && continue
        nx^2 + ny^2 + nz^2 < 27 || continue
        kv = 2pi .* [nx/L[1], ny/L[2], nz/L[3]]
        k2 = dot(kv, kv)
        C = sum(charges[i] * cos(dot(kv, q[:, i])) for i in eachindex(charges))
        S = sum(charges[i] * sin(dot(kv, q[:, i])) for i in eachindex(charges))
        Efourier_sphere += pref * (2pi / prod(L)) * exp(-k2/(4alpha^2))/k2 * (C^2 + S^2)
    end
    Eself = -pref * alpha / sqrt(pi) * sum(abs2, charges)
    # NIST removes intramolecular erf(alpha*r)/r from full Ewald, not the
    # engine's point-Coulomb (scale-1)/r pair-correction convention.
    Eintra = 0.0
    for m in 0:nw-1, i in (3m+1):(3m+3), j in (i+1):(3m+3)
        d = q[:, i] - q[:, j]
        d .-= L .* round.(d ./ L)
        r = norm(d)
        Eintra -= pref * charges[i] * charges[j] * erf64(alpha * r) / r
    end
    # NIST's practical real-space expression excludes same-molecule pairs;
    # its separate intramolecular erf term is the only intramolecular term.
    Ureal_nist = Ereal - Eintra_real
    # Engine APIs require rcut < L/2; prevfloat implements the same strict
    # distance comparison except at the exact half-box boundary.
    Uew, _ = E.evaluate_ewald(q, charges, L; alpha, rcut=rc, kmax=5,
                               prefactor=pref)
    box = T.OrthorhombicBox(Tuple(L))
    plan = P.PMEPlan(q, box.L; alpha, rcut=rc, grid=(32,32,32), order=4,
                     prefactor=pref)
    Upme, _ = P.pme_energy_forces(plan, q, charges)
    e_ref = -4.88604e5
    e_ref_mol = e_ref / nw
    e_ew = Ulj + lrc + (Uew - Eintra_real) + Eintra
    e_pme = Ulj + lrc + (Upme - Eintra_real) + Eintra
    @printf("Coulomb terms diagnostic K: real_all=%.6f real_intra_excluded=%.6f real_NIST=%.6f fourier_sphere=%.6f self=%.6f intra_erf=%.6f\n", Ereal, Eintra_real, Ureal_nist, Efourier_sphere, Eself, Eintra)
    @printf("NIST SPC/E config1: nw=%d L=%.1fA rc=%.17gA alpha=%.8g/A kmax_cube=5 grid=32^3 order=4\n", nw, L[1], rc, alpha)
    @printf("KCOUL_K_A=%.12g\n", KCOUL_K_A)
    @printf("LJ_disp_K=%.10f LRC_K=%.10f Ewald_K=%.10f PME_K=%.10f NIST_intramolecular_correction_K=%.10f\n", Ulj, lrc, Uew, Upme, Eintra)
    @printf("NIST_total_K=%.10f NIST_K_per_mol=%.10f\n", e_ref, e_ref_mol)
    @printf("engine_direct_total_K=%.10f per_mol=%.10f delta_per_mol=%.10f rel=%.8g\n", e_ew, e_ew/nw, (e_ew-e_ref)/nw, (e_ew-e_ref)/e_ref)
    @printf("engine_pme_total_K=%.10f per_mol=%.10f delta_per_mol=%.10f rel=%.8g\n", e_pme, e_pme/nw, (e_pme-e_ref)/nw, (e_pme-e_ref)/e_ref)
end

run()
