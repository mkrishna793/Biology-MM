#!/usr/bin/env python3
"""Register or inspect one Phase-2 chemistry fixture.

No new schema and no second parser: the manifest key set is the existing one and
parameter reading is delegated to the existing extractors.  The only new
artifact is ``chemistry/reference/capabilities.json``.

    python register_fixture.py --spec s.json --out /tmp/x.json [--make-reference]
    python register_fixture.py --only-capabilities

Exit codes: 0 accepted, 2 refused (typed reason on stderr), 1 usage/io.
"""
from __future__ import annotations
import argparse, datetime as _dt, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
MANIFEST = REPO / "chemistry" / "fixtures" / "manifest.json"
REFERENCE_DIR = REPO / "chemistry" / "reference"
CAPABILITIES = REFERENCE_DIR / "capabilities.json"
EXTRACTORS = {"amber": TOOLS / "extract_amber.py", "charmm": TOOLS / "extract_charmm.py"}

sys.path.insert(0, str(TOOLS))
import common  # noqa: E402  (existing record validation, not a second parser)

CLASSES = ("water+ions", "solvated protein", "DNA", "lipid membrane", "protein-ligand complex")
FAMILIES, STATUSES = ("amber", "charmm"), ("ready", "generated", "blocked")
ALWAYS = ("id", "class", "family", "force_field", "status")
PATHS = ("parameter_file", "coordinate_file", "structure_file")
OPTIONAL = ("source_license", "generator", "source_structure", "blocker")
SPEC_KEYS = ALWAYS + PATHS + OPTIONAL
TERMS = ("bonds", "angles", "urey_bradleys", "dihedrals", "impropers", "cmaps", "exclusions",
         "pair14", "lj146", "virtual_sites", "constraints")
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CHARMM_GAP = ("blocked: the manifest carries one parameter_file and the consumer test "
              "passes it as --parm with no --prm, so CHARMM parameters cannot be described "
              "yet (Phase 3+ manifest key)")
A = "chemistry/tools/extract_amber.py"
C = "chemistry/tools/extract_charmm.py"
B = "julia/src/ChemistryBuild.jl"
S = "julia/src/ChemistrySchema.jl"
SC = "julia/src/SystemContract.jl"
PM = "julia/src/PME.jl"
MB = "julia/src/MonteCarloBarostat.jl"
# term, families, status, owner symbol, (evidence file, required substring), note.
# A status is emitted only when the evidence needle is found; nothing is claimed blind.
MATRIX = (
    ("bonds", "amber,charmm", "supported", "Factors.HarmonicBond", (B, "Factors.HarmonicBond"), "k/2 (r-r0)^2; prmtop K doubled"),
    ("angles", "amber,charmm", "supported", "Factors.HarmonicAngle", (B, "Factors.HarmonicAngle"), "k/2 (theta-theta0)^2"),
    ("urey_bradley", "amber,charmm", "supported", "ChemistryFactors.UreyBradley", (B, "ChemistryFactors.UreyBradley"), "empty for plain AMBER prmtops"),
    ("proper_torsion_multiterm", "amber,charmm", "supported", "ChemistryFactors.MultiPeriodicDihedral", (B, "ChemistryFactors.MultiPeriodicDihedral"), "one entry per quadruple"),
    ("harmonic_improper", "amber", "partial", "ChemistryFactors.HarmonicImproper", (B, "ChemistryFactors.HarmonicImproper"), "only via CHARMM_IMPROPERS (ChamberParm)"),
    ("harmonic_improper", "charmm", "supported", "ChemistryFactors.HarmonicImproper", (B, "ChemistryFactors.HarmonicImproper"), "PSF psi_k/psi_eq"),
    ("cmap", "amber,charmm", "supported", "ChemistryCMAP.CMAPCorrection", (B, "ChemistryCMAP.CMAPCorrection"), "flat periodic grid M[s + n*t]"),
    ("exclusions_1_2_1_3", "amber,charmm", "supported", "ChemistrySchema exclusions", (S, "exclusions"), "explicit zero-interaction pairs"),
    ("pair14_scaling", "amber,charmm", "supported", "ChemistryFactors.PairTableNonbonded", (B, "ChemistryFactors.PairTableNonbonded"), "sCoul/sLJ on the full pair"),
    ("lj_1_6_4_minus_C_over_r4", "amber", "supported", "ChemistryFactors.PairC64Table", (B, "ChemistryFactors.PairC64Table"), "LENNARD_JONES_CCOEF"),
    ("lj_1_6_4_minus_C_over_r4", "charmm", "refused", "extract_charmm.py", (C, '"lj146": []'), "no CHARMM 1-6-4 term exists"),
    ("nbfix", "amber,charmm", "refused", "extract_amber.check_nbfix", (A, "def check_nbfix"), "hard refusal by default"),
    ("ten_twelve_hbond", "amber", "refused", "extract_amber.check_hbond", (A, "10-12 (HBOND)"), "HBOND_ACOEF/BCOEF refused"),
    ("virtual_sites", "amber", "refused", "SystemContract._refuse_virtual_sites", (SC, "UnsupportedCapability(:virtual_sites"), "carried, execution refused"),
    ("virtual_sites", "charmm", "refused", "extract_charmm.py massless-atom gate", (C, "lone pairs/virtual sites are not supported"), "refused at extraction"),
    ("triclinic_nonbonded", "amber,charmm", "refused", "ChemistryBuild.build_resolved", (B, "orthorhombic-only backends"), "imported, never flattened"),
    ("pme", "amber,charmm", "supported", "PME.jl", (PM, "orthorhombic cells"), "scalar-charge PME, orthorhombic"),
    ("npt_isotropic", "amber,charmm", "partial", "MonteCarloBarostat.jl", (MB, "Metropolis"), "isotropic volume move only"),
)

class RegistrationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message

def _repo_path(value: str, field: str) -> Path:
    if os.path.isabs(value) or "\\" in value or ".." in Path(value).parts:
        raise RegistrationError("path", "%s must be repository-relative without '..': %r" % (field, value))
    return REPO / value

def validate_spec(spec: dict, manifest: dict) -> dict:
    if not isinstance(spec, dict):
        raise RegistrationError("schema", "the registration spec must be one JSON object")
    unknown = sorted(set(spec) - set(SPEC_KEYS))
    if unknown:
        raise RegistrationError("schema", "unknown key(s) %s; allowed keys are %s" % (unknown, list(SPEC_KEYS)))
    missing = [k for k in ALWAYS if not spec.get(k)]
    if missing:
        raise RegistrationError("schema", "missing required key(s): %s" % missing)
    if not ID_RE.match(spec["id"]):
        raise RegistrationError("id", "id must match %s: %r" % (ID_RE.pattern, spec["id"]))
    if any(fx["id"] == spec["id"] for fx in manifest["fixtures"]):
        raise RegistrationError("duplicate", "id %r already exists in %s" % (spec["id"], MANIFEST))
    for key, allowed in (("class", CLASSES), ("family", FAMILIES), ("status", STATUSES)):
        if spec[key] not in allowed:
            raise RegistrationError(key, "%s must be one of %s; a sixth class is Phase 3+" % (key, list(allowed)))
    out = {k: spec[k] for k in ALWAYS}
    out["force_field"] = str(out["force_field"])
    out.update({k: spec[k] for k in PATHS + OPTIONAL if spec.get(k) not in (None, "")})
    for key in PATHS:
        if key in out:
            _repo_path(out[key], key)
    if out["status"] == "ready":
        for key in ("parameter_file", "coordinate_file"):
            if key not in out or not _repo_path(out[key], key).is_file():
                raise RegistrationError("missing_file", "status 'ready' requires existing %s: %r"
                      % (key, out.get(key)))
    elif out["status"] == "generated" and not out.get("generator"):
        raise RegistrationError("status", "status 'generated' requires 'generator'")
    elif out["status"] == "blocked" and not out.get("blocker"):
        raise RegistrationError("status", "status 'blocked' requires 'blocker' with the exact missing input")
    return {k: out[k] for k in SPEC_KEYS if k in out}

def run_extractor(spec: dict, prms: list[str]) -> tuple[dict | None, tuple[str, str] | None]:
    """Return (record, None) or (None, (code, message)); never parses parameters itself."""
    handle, out_path = tempfile.mkstemp(suffix=".json", prefix="register_")
    os.close(handle)
    cmd = [sys.executable, str(EXTRACTORS[spec["family"]]), "--parm",
           str(_repo_path(spec["parameter_file"], "parameter_file")),
           "--crd", str(_repo_path(spec["coordinate_file"], "coordinate_file")),
           "--out", out_path, "--force-field", spec["force_field"], "--repo-root", str(REPO)]
    for prm in prms if spec["family"] == "charmm" else []:
        cmd += ["--prm", str(_repo_path(prm, "prm"))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        os.unlink(out_path)
        match = re.search(r"^error\[([a-z_]+)\]: (.*)$", proc.stderr, re.M)
        return None, ((match.group(1), match.group(2)) if match
                      else ("extractor", proc.stderr.strip()[-400:]))
    try:
        return common.read_record(out_path), None
    finally:
        os.unlink(out_path)

def verify_hashes(record: dict, spec: dict) -> dict:
    checks = {}
    for key, prov_key in (("parameter_file", "source_sha256"),
                          ("coordinate_file", "coordinate_sha256")):
        if key not in spec:
            continue
        declared = record["provenance"].get(prov_key, "")
        if not re.fullmatch(r"[0-9a-f]{64}", declared or ""):
            raise RegistrationError("bad_hash", "%s is not 64 lowercase hex characters: %r" % (prov_key, declared))
        digest = hashlib.sha256(_repo_path(spec[key], key).read_bytes()).hexdigest()
        if declared != digest:
            raise RegistrationError("bad_hash", "%s %s hashes to %s..., provenance says %s..."
                                    % (key, spec[key], digest[:12], declared[:12]))
        checks[key] = "verified"
    return checks

def term_summary(record: dict) -> dict:
    counts = {key: len(record[key]) for key in TERMS}
    counts["atoms"], counts["dihedral_terms"] = len(record["atoms"]), sum(
        len(d["terms"]) for d in record["dihedrals"])
    touched = {i for group in ("bonds", "angles", "urey_bradleys", "dihedrals",
                               "impropers", "constraints")
               for term in record[group] for i in
               (term[k] for k in ("i", "j", "k", "l") if k in term)}
    touched |= {i for cmap in record["cmaps"] for i in cmap["i"]}
    touched |= {site["site"] for site in record["virtual_sites"]}
    touched |= {p for site in record["virtual_sites"] for p in site["parents"]}
    counts["atoms_in_bonded_terms"] = len(touched)
    counts["atoms_nonbonded_only"] = len(record["atoms"]) - len(touched)
    return counts

def capability_report(spec: dict, record, error, hashes: dict) -> dict:
    matrix = {row[0]: {"status": row[2], "owner": row[3], "note": row[5]}
              for row in MATRIX if spec["family"] in row[1].split(",")}
    ref = REFERENCE_DIR / ("%s.json" % spec["id"])
    meta = json.loads(ref.read_text()) if ref.is_file() else None
    reference = {"exists": meta is not None, "n_atoms": (meta or {}).get("n_atoms"),
                 "configs": (meta or {}).get("n_configs")}
    report = {"schema": "1.0.0", "fixture": spec, "hashes": hashes, "reference": reference,
              "extraction": "refused" if error else ("skipped" if record is None else "ok"),
              "term_capabilities": matrix,
              "consumer_extraction": CHARMM_GAP if spec["family"] == "charmm" else "ok"}
    if error:
        lowered = error[1].lower()
        report["refusal"] = {"code": error[0], "message": error[1]}
        report["refusal_terms"] = [name for name, needle in
                                   (("nbfix", "nbfix"), ("ten_twelve", "10-12"),
                                    ("virtual_sites", "lone pair"), ("cmap", "cmap"))
                                   if needle in lowered]
        return report
    if record is None:
        return report
    kind = record["box"]["kind"]
    report["counts"] = term_summary(record)
    report["recorded_capabilities"] = {
        "box_kind": kind,
        "triclinic_nonbonded": "refused" if kind == "triclinic" else "not-applicable",
        "virtual_sites": "refused-at-execution" if record["virtual_sites"] else "not-present",
        "nbfix": "not-present (extractor refuses NBFIX by default)",
        "lj_1_6_4_minus_C_over_r4": ("present (%d pairs)" % len(record["lj146"])
                                     if record["lj146"] else "not-present"),
        "pme": "available (orthorhombic)" if kind == "orthorhombic" else "needs an orthorhombic box"}
    return report

def print_report(report: dict) -> None:
    fx, h = report["fixture"], report["hashes"]
    print("fixture   : %s (%s, %s, %s)" % (fx["id"], fx["class"], fx["family"], fx["status"]))
    print("extraction: %s | reference: %s | consumer: %s"
          % (report["extraction"], report["reference"]["exists"], report["consumer_extraction"]))
    print("sha256    : " + ", ".join("%s=%s" % item for item in sorted(h.items())))
    if "counts" in report:
        print("terms     : " + ", ".join("%s=%s" % (k, v) for k, v in sorted(report["counts"].items())))
        print("capability: " + ", ".join("%s=%s" % (k, v)
                                         for k, v in sorted(report["recorded_capabilities"].items())))
    if "refusal" in report:
        print("refused   : [%s] %s" % (report["refusal"]["code"], report["refusal"]["message"]))
    print("matrix    : " + ", ".join("%s=%s" % (k, v["status"])
                                     for k, v in sorted(report["term_capabilities"].items())))

def build_capabilities(now: str | None = None) -> dict:
    terms = []
    for term, families, status, owner, (relpath, needle), note in MATRIX:
        path = REPO / relpath
        if not path.is_file():
            raise RegistrationError("evidence", "capability evidence file is missing: %s" % relpath)
        if needle not in path.read_text(encoding="utf-8", errors="replace"):
            raise RegistrationError("evidence", "%s lacks %r; refusing to claim %s" % (relpath, needle, term))
        terms.extend({"family": family, "term": term, "status": status, "owner": owner,
                      "evidence": [relpath], "notes": note}
                     for family in families.split(","))
    return {"schema": "1.0.0",
            "generated_at": now or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "chemistry/tools/register_fixture.py",
            "scope": "Phase-2 capability boundary (mdparams 1.0.0); Phase 3+ is in EXTENDING.md",
            "status_legend": {"supported": "implemented and exercised by the fixture suite",
                              "partial": "implemented only on the stated narrow path",
                              "refused": "typed error instead of an approximation"},
            "terms": sorted(terms, key=lambda row: (row["family"], row["term"]))}

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", help="registration spec JSON ('-' reads stdin)")
    p.add_argument("--prm", action="append", default=[],
                   help="extra CHARMM parameter file (repeatable; required for charmm)")
    p.add_argument("--out", help="write the normalized JSON to this path")
    p.add_argument("--make-reference", action="store_true", help="run make_reference.py")
    p.add_argument("--no-extract", action="store_true", help="offline: skip the extractor")
    p.add_argument("--normalized", help="offline: verify this normalized JSON")
    p.add_argument("--only-capabilities", action="store_true", help="write the matrix and exit")
    p.add_argument("--capabilities-out", default=str(CAPABILITIES))
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    args = p.parse_args(argv)
    try:
        if args.only_capabilities:
            Path(CAPABILITIES).write_text(
                json.dumps(build_capabilities(), indent=2, sort_keys=True) + "\n")
            print(args.capabilities_out)
            return 0
        if not args.spec:
            p.error("--spec is required unless --only-capabilities is used")
        spec = validate_spec(json.loads(sys.stdin.read() if args.spec == "-"
                                        else Path(args.spec).read_text()),
                             json.loads(MANIFEST.read_text()))
        record, error = None, None
        hashes = {"parameter_file": "not-run", "coordinate_file": "not-run"}
        if args.no_extract:
            if args.normalized:
                record = common.read_record(args.normalized)
        else:
            if spec["status"] != "ready":
                raise RegistrationError("status", "extraction requires status 'ready'")
            if spec["family"] == "charmm" and not args.prm:
                raise RegistrationError("charmm_parameters", CHARMM_GAP)
            record, error = run_extractor(spec, args.prm)
            if record is not None:
                hashes = verify_hashes(record, spec)
        if record is not None and args.out:
            Path(args.out).write_text(common.json_text(record))
        report = capability_report(spec, record, error, hashes)
        if args.make_reference:
            rc = subprocess.run([sys.executable, str(TOOLS / "make_reference.py"),
                                 "--fixture", spec["id"]]).returncode
            report["make_reference"] = "ok" if rc == 0 else "failed"
            if rc != 0:
                print("make_reference.py refused %r (unknown ids fail argparse; see "
                      "CONFIG_COUNT)" % spec["id"], file=sys.stderr)
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else "", end="")
        print_report(report) if not args.json else None
        if error:
            print("error[%s]: %s" % error, file=sys.stderr)
            return 2
        return 0 if report.get("make_reference", "ok") == "ok" else 2
    except RegistrationError as exc:
        print("error[%s]: %s" % (exc.code, exc.message), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print("error[io]: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
