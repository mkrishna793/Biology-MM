# Independent Verification Lane Report

Run context: 2026-11-12, macOS 27.2 arm64. Repository thread count was 1 for the Julia lane; no heavy benchmarks or trajectories were run. This lane is additive and does not modify `FoundationCore` source or its project files.

## Tools and environments

| Tool | Version / location | Status |
|---|---|---|
| Julia | 1.13.3, `~/.cache/biology-mm/julia-1.13.3/bin/julia` | Used |
| TaylorSeries.jl | 0.23.4, isolated `verification-env`, `verify-depot` | Used |
| Enzyme.jl | 0.13.219, same isolated Julia environment | Used |
| IntervalArithmetic.jl | 1.0.14, same isolated Julia environment | Used, limited enclosures |
| pymbar | 4.0.3, isolated `~/.cache/biology-mm/pyenv` | Used on synthetic samples |
| InterMol | 0.1.2+6.g7125764, commit `7125764d42f6e6c589dc2d1df71e3e812b3a7b27`, micromamba prefix `~/.cache/biology-mm/intermol-verify` | Conversion suite executed; matched engine energy/force comparison NOT RUN |
| NIST SPC/E | NIST page and sample archive fetched 2026-11-12 | Config 1 evaluated; large unresolved Coulomb discrepancy |
| NIST TraPPE ethane/octane/butane | Pages fetched 2026-11-12 | Thermophysical pages only; bonded fixture comparison NOT RUN |

Reproduction commands:

```sh
bash julia/experiments/verification/run_all.sh
micromamba create -y -p "$HOME/.cache/biology-mm/intermol-verify" -c conda-forge python=3.10 git pip
micromamba run -p "$HOME/.cache/biology-mm/intermol-verify" python -m pip install 'git+https://github.com/shirtsgroup/InterMol.git'
micromamba run -p "$HOME/.cache/biology-mm/intermol-verify" python -c 'import intermol; print(getattr(intermol,"__version__","unknown"), intermol.__file__)'
```

InterMol import succeeded after installing NumPy 1.26.4, ParmEd 4.3.1, six 1.17.0, Pint 0.24.4, NetworkX 3.4.2, SciPy 1.15.3, pytest 9.2.0, and setuptools 80.10.2 to the isolated prefix. Command: `micromamba run -p "$HOME/.cache/biology-mm/intermol-verify" python -m intermol.tests.test_all --type unit --program gromacs`. The suite produced GROMACS-to-GROMACS conversions and some conversions to other installed target formats; summary is `all_results.txt`, and generated files are in `unit_test_outputs/`. The initial invocation omitted `--energy`, so it compares no energies or forces. A subsequent `--energy` attempt failed before conversion on missing `pkg_resources`; pinning setuptools <81 resolved the import, but the energy-enabled suite has not been rerun. These two repository-root artifacts were created by the conversion tool outside the authorized report paths; they are left untouched.

NIST SPC/E canonical page fetched 2026-11-12: `https://www.nist.gov/mml/csd/chemical-informatics-group/spce-water-reference-calculations-cuboid-cell-10a-cutoff`; sample archive: `https://www.nist.gov/document/spcesampleconfigurations-targz`, SHA-256 `f9e969b45299b8c20c657fefef63f08f445c683781ad1a0f4b63470337e42b42`. NIST config 1 has 100 waters, 20 Å cubic box; reference total is `-4.88604e5 K` (`-4886.04 K/molecule`, six significant figures). Its page supplies no numeric acceptance tolerance and no virial/pressure result. Corrected SRSW index fetched: `https://www.nist.gov/programs-projects/nist-standard-reference-simulation-website`.

TraPPE pages fetched 2026-11-12: `http://mmlapps.nist.gov/srs/ETHANE/index.htm`, `http://mmlapps.nist.gov/srs/ETHANE/ethane_sat.htm`, `http://mmlapps.nist.gov/srs/OCTANE/index.htm`, `http://mmlapps.nist.gov/srs/OCTANE/octane_sat.htm`, and `https://www.nist.gov/chemical-informatics-research-group/sat-tmmc-liquid-vapor-coexistence-properties-trappe-ua-n-butane`. The ethane and octane pages state thermophysical properties (including phase potential energies), but do not provide a static coordinate fixture with termwise bond/angle/torsion energies or forces. No bonded-force result is inferred from those benchmarks.

## Per-term comparison status

| Quantity | Engine convention / model | Independent result | Tolerance / status |
|---|---|---|---|
| Bond | `U = (k/2)(r-r0)^2`; six-atom local chain lane | Aggregate bonded forces cross-checked; no isolated bond-vs-external-engine table | Taylor/Enzyme relative force residuals below; InterMol and NIST NOT RUN |
| Angle | `U = (kθ/2)(θ-θ0)^2`, `θ=acos(cosθ)` | Included in aggregate derivative lane; no external parameterized comparison | No independent convention verdict; angle guaranteed-domain interval NOT RUN |
| Proper torsion | `U = kφ[1+cos(nφ-φ0)]`, `φ=atan2(y,x)` | Included in aggregate Taylor/Enzyme force checks, but no independent force-field converter/reference | Phase/sign/improper conventions NOT externally validated |
| Improper | No separate improper fixture or convention in this lane | NOT RUN | Gap |
| 1-4 scaling/exclusions | No matched bonded/nonbonded force-field topology fixture | NOT RUN | Gap; no numerical value to report |
| Virial / pressure | No validated force-virial output is exposed to this lane | No energy box-scaling Jet; NIST cuboid reference page has no virial/pressure table | NOT RUN; NPT not validated |
| Constraints | Coupled constraint projection of rigid-water geometry and tangent velocities | Internal derivative/interval checks only; not comparison to SETTLE/RATTLE reference implementation | No independent solver comparison |

Exact bonded parameters used by the derivative fixture are defined in `common.jl` (`CHAIN`); the fixture is a synthetic mixed chain, not an Amber/CHARMM parameterization. No external per-term energy/force numbers or tolerances exist yet. The aggregate energy parity (`0.000e+00`) pins the independent law transcription to the engine energy, but is not independent physics evidence.

## NIST SPC/E Energy Comparison

Command: `SPCE_CONFIG=julia/experiments/verification/spce_sample_config_periodic1.txt JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/verify-depot:$HOME/.cache/biology-mm/julia-depot" "$HOME/.cache/biology-mm/julia-1.13.3/bin/julia" --project="$HOME/.cache/biology-mm/verification-env" julia/experiments/verification/nist_spce.jl` (Julia 1.13.3, one thread). Archive geometry was held fixed, not dynamically constrained. Model inputs: SPC/E charges O `-0.8476 e`, H `+0.4238 e`; O LJ sigma `3.16555789 Å`, epsilon/kB `78.19743111 K`; alpha `0.28 Å^-1`; engine cutoff `prevfloat(10 Å)` because API requires strict `rcut < L/2`; direct Ewald integer cube `kmax=5`; PME grid `32^3`, order 4; CODATA 2010 Coulomb prefactor `167100.956632 K Å/e²`. This differs from the NIST reciprocal spherical mask `k²<27` and the strict half-box cutoff was adjusted by one representable Float64 step.

| Term | Measured value | Reference / tolerance | Assessment |
|---|---:|---|---|
| LJ dispersion | `99538.7362119 K` | NIST table reports approximately `9.95387e4 K` | AGREES at displayed precision |
| LJ long-range correction | `-823.7149949 K` | NIST table reports approximately `-8.23715e2 K` | AGREES at displayed precision |
| Real Coulomb, all site pairs (diagnostic recomputation) | `-7915968.917865 K` | NIST omits same-molecule terms from real-space sum | Diagnostic only |
| Real Coulomb, intermolecular subset | `-558888.919728 K` | NIST `-558889 K` | Matches at displayed precision |
| Fourier spherical diagnostic `k²<27` | `6270.093784 K` | NIST table `6270.09 K` | Matches |
| Self diagnostic | `-2844691.573530 K` | NIST table approximately `-2.84469e6 K` | Matches at displayed precision |
| Intramolecular `erf` correction | `+2809992.199837 K` | NIST `+2809990 K` | Matches at reported six-digit precision |
| Total after NIST real-space exclusion and computed intra correction | Direct `-488595.741494 K`; PME `-488595.330576 K` | NIST `-488604 K`; no stated tolerance | Close measured agreement; delta `+0.08259 K/molecule` direct, `+0.08669 K/molecule` PME; relative deviations `1.6902e-5` and `1.7743e-5` |

NIST's practical formula explicitly excludes same-molecule pairs from `Ereal`; the evaluator subtracts the diagnostic intra-real contribution and applies the separate `erf` correction. The intramolecular correction uses periodic minimum-image displacements: some water molecules cross the coordinate-box boundary, so raw coordinate differences are not bonded displacements. With this correction, intermolecular real, spherical Fourier, self, and intramolecular terms reproduce the rounded NIST components. Direct and PME totals agree with the NIST total to roughly `0.08 K/molecule`. NIST states six significant figures but no numeric tolerance. This is strong measured agreement, not a formally tolerance-qualified pass. The direct reciprocal sum uses the full cube `kmax=5` rather than the reference sphere, and the PME uses a finite `32^3` mesh; the residual is reported without attributing it to a source defect.

## Derivative and interval measurements

Command: `bash julia/experiments/verification/run_all.sh` (Julia 1.13.3, one thread).

| Check | Measured | Interpretation |
|---|---:|---|
| Independent generic-law vs engine energy parity | 0.000e+00 | Same tested formula/fixture, not convention validation |
| TaylorSeries force residual | 4.564e-16 | First-order coefficients vs engine scatter, aggregate bonded chain |
| Enzyme force residual | 7.417e-16 | Reverse-mode derivative of engine energy path vs scatter |
| Central-FD directional residual | 1.588e-09 | Sanity check only, not above-machine-precision evidence |
| Projected-water tangent derivative | 7.974e-17 | O-H tangency for tested geometry |
| Interval chain-energy width | 1.194e-03 | Coordinate hull radius 1e-6; all 200 sampled engine energies contained |
| Bond-force interval width | max 1.285e-04 | Component enclosure around one chain bond |
| Constraint geometry | pass at tested fixture | Projected water with 1e-13 coordinate intervals; contains target distances and zero residual |
| Barostat interval | pass at tested tuple | Encloses a `log_acceptance` point; not a virial enclosure |
| Barostat identity/bookkeeping | 2.220e-16 symmetry; 7.002e-16 step consistency | Algebra/bookkeeping only, not physical pressure correctness |

These intervals do **not** cover the requested domains (`theta∈[1e-3,π-1e-3]`, `r∈[0.5,5]`, full torsion period/atan2 branch cut, coincident-atom projection, or all PME modes). No claim of global boundedness, non-NaN behavior, or IEEE 1788 proof over those domains is made. The existing interval lane samples engine evaluations within a small coordinate box; it is not a validated interval implementation of the Float64 engine.

Parameter derivatives `dU/dtheta` for parameter overrides were not evaluated with TaylorSeries or Enzyme in this lane. Constraint Jacobian rows `Dg(q)` were not directly extracted as Jets. Barostat box-derivative/virial relation was not tested.

## MBAR

Analyzed only a synthetic four-state harmonic replica dataset: `U_k(x)=k x²/2`, beta 1, `k=[1,2,4,8]`, 2,000 independent exact Gaussian samples per state, seed 20261112. It does not use engine parallel-tempering or barostat output.

Julia fixed point converged in 18 iterations for 8,000 samples. Estimated `Δf` relative to state 1: `[0.00000, 0.33791, 0.67981, 1.02357]`; analytic values `[0.00000, 0.34657, 0.69315, 1.03972]`. Worst absolute error `1.615e-02`; maximum of 100-replicate within-state bootstrap SEs `1.470e-02`. pymbar 4.0.3 gave maximum deviation from analytic `0.016149852780049434` (within four combined standard errors; tolerance `0.05635928098772002`) and differed from Julia by `4.884981308350689e-14`. No overlap matrix/overlap eigenvalue diagnostics are reported by this lane. This validates estimator agreement on a synthetic known ensemble, not engine sampling or replica exchange.

## Proven / measured / assumed / not run

- **Measured:** TaylorSeries and Enzyme force/scatter agreement for the tested aggregate bonded chain; projected-water tangent derivative; small-neighborhood interval containment; barostat acceptance bookkeeping; synthetic harmonic MBAR vs analytic values and pymbar.
- **Assumed:** The independently transcribed chain energy laws correctly mirror their intended formulas; synthetic Gaussian samples are independently drawn from their stated distributions. These assumptions do not establish Amber/CHARMM convention compatibility.
- **Measured:** InterMol unit conversion suite ran without energy comparison; SPC/E config 1 direct Ewald and PME totals agree with NIST within `0.09 K/molecule`, with component diagnostics matching NIST rounded terms.
- **Not run:** InterMol matched energy/force comparison and energy-enabled test suite; NIST TraPPE bonded-force comparison; term-resolved external bond/angle/torsion/1-4 comparison; improper convention; virial/box-scaling Jet; broad-domain interval proof (including torsion branch cut, near-collinear angles, coincident constraints and all PME modes); parameter-gradient Jet/AD; direct constraint-Jacobian Jet; engine PT/barostat samples; MBAR overlap diagnostics; independent SETTLE/RATTLE comparison.
- **Proven:** No new formal theorem. No physics claim in this report is machine-checked proof.

## Ranked discrepancies and owner actions

1. **Critical validation gap, virial/pressure:** no code virial was compared against a box-scaling derivative; this NIST page supplies no virial. NPT remains unvalidated.
2. **Critical validation gap, torsion phase/improper/1-4 conventions:** InterMol converted test fixtures but no matched energy/force fixture was completed; no independent bonded force-field fixture was compared.
3. **Critical validation gap, broad-domain intervals and coupled constraints:** full angular/torsional/constraint/PME domains and external SETTLE/RATTLE comparison remain not run.
4. **InterMol artifacts outside permitted scope:** `all_results.txt` and `unit_test_outputs/` were generated at repository root by the suite. They were left untouched; user review/removal is needed if cleanup is desired.

No unified engine source diff is proposed: no source defect was demonstrated by the corrected SPC/E comparison. Existing source/design audit findings are outside this lane's authority and are not re-audited here.
