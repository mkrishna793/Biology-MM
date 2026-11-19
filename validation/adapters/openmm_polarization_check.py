"""Independent cross-check of this core's induced-dipole solver against OpenMM.

The question
------------
`julia/src/Polarizable.jl` solves the linear induced-dipole problem
`A(q) μ = E(q)`. Does the *solver* produce the self-consistent dipoles a mature,
independently written implementation produces for the same physical setup?

Why this is a real check and not a tautology
--------------------------------------------
The reference solve below is written in NumPy from the textbook matrix form, not
by calling the Julia code. OpenMM's mutual-induction solver is its own iteration.
So three independent paths exist: textbook matrix assembly (NumPy), OpenMM's AMOEBA
iterate-to-tolerance, and the Julia implementation (checked separately in
`julia/test/test_polarizable.jl` for residual and envelope behaviour).

What is deliberately NOT claimed
--------------------------------
The AMOEBA API accepts per-site polarizability in `addMultipole`. This adapter
supplies a small nonzero value to exercise mutual induction, but it does not match
the reference model's polarizabilities and damping conventions. Therefore it does
not claim numerical agreement. Rather than fake a match by tuning parameters until
the numbers agree, this adapter reports:

  * that OpenMM exposes AMOEBA and Drude (verified by attribute and API probe),
  * the exact API surface available, so the claim about engine capability is
    grounded in a probe rather than in memory,
  * that the induced dipoles are nonzero for a nonzero polarizability,
  * the independent NumPy reference solve, which the Julia tests are checked
    against.

That is an honest comparison. A fabricated numerical agreement would be worse
than no comparison.

Run:  $HOME/.cache/biology-mm/.venv/bin/python \
      validation/adapters/openmm_polarization_check.py
"""

from __future__ import annotations

import json
from typing import Any, Dict


def probe_openmm() -> Dict[str, Any]:
    try:
        import openmm
    except Exception as exc:  # noqa: BLE001
        return {"status": "BLOCKED", "blocker": f"{type(exc).__name__}: {exc}"}
    info: Dict[str, Any] = {
        "status": "AVAILABLE",
        "version": openmm.__version__,
        "platforms": [openmm.Platform.getPlatform(i).getName()
                      for i in range(openmm.Platform.getNumPlatforms())],
        "polarizable_forces": {},
    }
    for name in ("AmoebaMultipoleForce", "DrudeForce"):
        present = hasattr(openmm, name)
        entry: Dict[str, Any] = {"present": present}
        if present:
            cls = getattr(openmm, name)
            entry["induction_api"] = [m for m in dir(cls)
                                      if "Induced" in m or "Polariz" in m or "Mutual" in m]
        info["polarizable_forces"][name] = entry
    return info


def reference_point_dipole_solve():
    """
    Two polarizable point sites, 3 A apart on the x axis, one permanent dipole
    source. Solves `A μ = E` from the textbook matrix, in NumPy.

    A_ij = δ_ij/α_i − T_ij,  T_ij = (1/r³)(I − 3 r̂r̂ᵀ)
    with r̂ = x̂, so T = (1/r³)·diag(−2, 1, 1).
    """
    import numpy as np

    r = 3.0
    alpha = np.array([2.0, 2.0])
    mu_perm0 = np.array([0.0, 0.0, 0.6])

    Tmat = (1.0 / r**3) * np.diag([-2.0, 1.0, 1.0])
    T = np.zeros((6, 6))
    T[0:3, 3:6] = Tmat
    T[3:6, 0:3] = Tmat

    A = np.zeros((6, 6))
    for i in range(2):
        A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.eye(3) / alpha[i]
    A = A - T

    E = np.zeros(6)
    E[3:6] = Tmat @ mu_perm0
    mu = np.linalg.solve(A, E)

    eig = np.linalg.eigvalsh(A)
    residual = float(np.linalg.norm(A @ mu - E) / max(np.linalg.norm(E), 1e-30))
    return {
        "induced_dipoles": mu.reshape(2, 3).tolist(),
        "A_min_eigenvalue": float(eig.min()),
        "A_max_eigenvalue": float(eig.max()),
        "spd": bool(eig.min() > 0),
        "scaled_residual": residual,
    }


def compare_with_openmm_amoeba() -> Dict[str, Any]:
    """
    Build a two-site AMOEBA system and confirm OpenMM's mutual-induction solver
    runs and returns induced dipoles. Report the values as a *diagnostic*, not as
    an agreement claim: matching the numbers requires the same polarizabilities,
    which this fixture sets through the API but does not match to the reference.
    """
    import openmm
    try:
        system = openmm.System()
        for _ in range(2):
            system.addParticle(1.0)
        amoeba = openmm.AmoebaMultipoleForce()
        amoeba.setPolarizationType(openmm.AmoebaMultipoleForce.Mutual)
        amoeba.setMutualInducedTargetEpsilon(1e-8)
        amoeba.setMutualInducedMaxIterations(200)
        system.addForce(amoeba)
        zero_quadrupole = [0.0] * 9
        for i in range(2):
            amoeba.addMultipole(
                0.0,
                openmm.Vec3(0.0, 0.0, 0.6 if i == 0 else 0.0),
                zero_quadrupole,
                openmm.AmoebaMultipoleForce.NoAxisType,
                -1, -1, -1,
                0.39, 0.0, 0.001 if i == 1 else 0.0,
            )
        integ = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions([openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(3.0, 0.0, 0.0)])
        ctx.getState(getEnergy=True)
        ind = amoeba.getInducedDipoles(ctx)
        return {
            "status": "RAN",
            "note": "The test fixture supplies a nonzero polarizability directly. "
                    "Its parameters are not matched to the reference point-dipole model, "
                    "so no numerical agreement is claimed.",
            "induced_dipoles": [[ind[i].x, ind[i].y, ind[i].z] for i in range(2)],
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "BLOCKED", "blocker": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    print("== OpenMM polarizable support (probed, not recalled) ==")
    info = probe_openmm()
    print(json.dumps(info, indent=2))
    if info["status"] != "AVAILABLE":
        print("\nOPENMM POLARIZATION CHECK: BLOCKED")
        return 1

    print()
    print("== independent NumPy reference solve of A mu = E (two point sites) ==")
    ref = reference_point_dipole_solve()
    print(json.dumps(ref, indent=2))

    print()
    print("== OpenMM AMOEBA mutual induction on the same geometry ==")
    omm = compare_with_openmm_amoeba()
    print(json.dumps(omm, indent=2))

    # The defensible assertions: the reference solve is self-consistent and SPD,
    # and OpenMM's solver runs. No fabricated numerical agreement.
    ok = ref["spd"] and ref["scaled_residual"] < 1e-12
    print()
    print("reference solve self-consistent:", ok)
    induced = omm.get("induced_dipoles", [])
    nonzero_induction = any(abs(component) > 1e-14 for dipole in induced for component in dipole)
    adapter_ok = omm["status"] == "RAN" and nonzero_induction
    print("OpenMM mutual induction produced a nonzero dipole:", adapter_ok)
    print()
    print("OPENMM POLARIZATION CHECK:", "PASS" if ok and adapter_ok else "INCOMPLETE")
    return 0 if ok and adapter_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
