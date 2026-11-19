"""Optional third-party validation adapters.

Each adapter is *optional*: it probes for the package, and if the package is
absent it reports a structured `BLOCKED`/`UNAVAILABLE` status rather than failing
the whole suite or, worse, pretending to have run.

Run:  python3 validation/adapters/probe_environment.py
      python3 validation/adapters/trajectory_diagnostics.py --help

What each adapter is and is not allowed to claim is documented per-adapter,
because the failure mode here is a small run being reported as thermodynamics.
"""
