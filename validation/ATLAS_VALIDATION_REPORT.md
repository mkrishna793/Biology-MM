# Biology-MM v2.6.0 AlphaFold pLDDT/PAE & ATLAS-Style Flexibility Validation

**Status:** `PASS` (1.96s)  
**Reproduce:** `PYTHONPATH="python:studio/backend" python3 validation/run_atlas_validation.py`

| Metric | Measured Value | Target / Criterion |
| :--- | ---: | :--- |
| PAE Rigid Domains Detected | `2` | `2` (exact two-domain split) |
| PAE Flexible Hinge Residues | `12` | `12` (linker residues 25–36) |
| Flexible Loop Mean RMSF (`pLDDT < 70`) | `0.8088 Å` | Higher mobility than rigid core |
| Rigid Core Mean RMSF (`pLDDT >= 90`) | `0.4514 Å` | Anchored by smooth pLDDT restraints |
| Loop / Core RMSF Ratio | **`1.792×`** | `> 1.35×` |
| Flexibility Pearson $r$ | **`0.6229`** | `> 0.50` |
| Simulation Health / Shadow-$H$ Drift | **`bounded`** (`4.01e-04`) | `bounded` |
