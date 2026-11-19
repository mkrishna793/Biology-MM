"""
Phase-2 chemistry import: parameter-file -> strict normalized record ->
`ResolvedSystem`.

The module is deliberately split into three layers:

* `ChemistrySchema.jl` — fixed-unit (`nm`, `kJ/mol`, `e`, `amu`) JSON record,
  strict validation, deterministic serialization. It knows nothing about
  factors.
* `ChemistryBuild.jl` — the single builder from a normalized record to
  `ResolvedSystem` factors, topology, exclusions/1-4 scales and protocol.
* `ChemistryDiagnostics.jl` — preparation/consistency audit with atom/residue
  references; it never repairs or invents parameters.

Referenced force-field files remain external inputs. This module is not a force
field and makes no claim that an imported system is equilibrated.
"""
module Chemistry

using JSON
using LinearAlgebra

using ..Types
using ..Factors
using ..ChemistryFactors
using ..ChemistryCMAP
using ..ResolvedSystem
using ..StreamedLJ
using ..PME
using ..Constraints

include("ChemistrySchema.jl")
include("ChemistryBuild.jl")
include("ChemistryDiagnostics.jl")

export CHEM_SCHEMA_VERSION, CHEM_UNITS, CHEM_COULOMB_PREFACTOR, ChemistryError,
       NormalizedSystem, NormalizedAtom, NormalizedBox, BondTerm, AngleTerm,
       UreyBradleyTerm, TorsionTerm, DihedralTerm, ImproperTerm, CmapTerm,
       Pair14Term, VirtualSiteTerm, ConstraintTerm, ImportProvenance,
       DiagnosticRecord, SimulationProtocol, AuditReport, RawAuditReport,
       read_normalized, write_normalized, validate_normalized!,
       normalization_summary, default_protocol, build_resolved, build_report,
       audit_parameterized, audit_raw_structure, n_atoms

end # module Chemistry
