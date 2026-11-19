# Trium-XBT-compiler — Evaluation Against the TIP3P 1728-Water MD Workload

**Date:** 2026-11-13 (third pass; supersedes the 2026-11-12 draft of this file)
**Evaluated commit:** `313f669e9e605c09dc84fe2db8d200a5aee6f302` (`main`), audit clone at
`~/.cache/biology-mm/trium-xbt-eval/Trium-XBT-compiler` — `git status --porcelain` empty, 0 modified,
0 untracked, before and after this audit.
**Target workload:** TIP3P 1728-water NVT trajectory, **912.7 ms/step**, float64 CPU
(`experiments/tip3p_trajectory/runs/w200_corrected/RUN.md:181-188`; 1728-water tier,
`validation/STATUS.md:442`)

---

## 0. Verdict

**REJECT for MD acceleration.** This is a real, working compiler — not a mock-up — but it is a
scalar-only language whose back end is "emit LLVM IR text, spawn the host `clang`, `dlopen` the
result." Nothing in it can express or beat the existing float64 + FFTW + in-tree threaded
`StreamedLJ` pipeline on this workload.

Four independent blockers, each executed this session:

1. **The language cannot express the workload.** No `for`, no arrays, no subscripting, and **no
   `/` operator at all**. Integers are 39-trit, not 64-bit. The LJ pair loop, the cell list, and
   the PME B-spline spread are all inexpressible.
2. **The compiler contributes no code generation of its own.** `nm -u` on both `bt-compiler` and
   `bt-opt` finds **0** LLVM and **0** upstream-MLIR undefined symbols. It writes IR text and
   delegates to `clang`. Measured: the whole `--native` round trip is **≈340 ms** for a program
   whose IR compiles in **43 ms** — and **≈295 ms of that is macOS `dlopen` of the freshly written
   dylib**, not compilation (§5.4).
3. **Ternary FP is a downgrade.** `BTFP<20>` carries **29.53–30.11** effective mantissa bits
   (binade-dependent, not a single number); `BTFP<40>` carries **61.23–61.81**. Both run **~200–330×**
   slower than hardware `double` for a mul-add. Worse, `BTFP<20> v(3.5)` **silently yields 3** —
   precision loss with no diagnostic at default warning levels.
4. **The radix-3 FFT is slower than what we already run.** 1.2–1.5× slower than a naive scalar
   radix-2 at matched sizes (per-run band 1.13–1.78×), forces N = 3^k (64³ → 81³, **2.0–2.2×** more
   work per grid point), and `PME.jl:205-206` already calls FFTW.

A fifth finding, new in this pass and independent of MD suitability: **the runtime is not
diagnostically safe**. `bt-compiler --test` is a no-op in Release while printing "PASSED WITH 0
FAILURES" (§2 #9); `--native` reports int39 overflow by printing "Execution completed" with an empty
stderr and exit 1 (§2 #11); `HardwareMMUZeroMatrix::reset()` leaves stale trits readable (§2 #10);
and an unhandled `BinaryOp` lowers to an unchecked `add i64` (`ternary_llvm_codegen.hpp:1018`).
These do not change the verdict — the language already cannot express the workload — but they would
have to be fixed before anything built on this compiler could be trusted in a simulation loop.

Tags used throughout: **VERIFIED** (executed on this host this session), **DOCUMENTED-CLAIM**
(stated in repo docs, not independently reproduced), **INFERRED** (reasoned from code or absence),
**REFUTED** (a prior-pass claim that did not survive re-checking), **CODE-READ** (established by
reading a file in full, without execution).

---

## 1. Method, isolation, and what the audits bought

### 1.1 Process facts — what is provable, and what is not

The original brief asked for **exactly 12 agents in two sequential waves of six**. That protocol was
**not** executed as specified. Stating this precisely rather than labelling the work "three waves":

**Provable from artifacts and this session's own tool results:**

| # | Lane | Evidence of existence | Artifact time |
|---|---|---|---|
| 1 | W1/A1 front end | `audit/w1_a1.txt` (24 paths) | 22:39:13 |
| 2 | W1/A2 IR/MLIR | `w1_a2.txt` (39) + `w1_a2_report.md` | 22:40:19 / 22:50:38 |
| 3 | W1/A3 back ends | `w1_a3.txt` (44) | 22:39:13 |
| 4 | W1/A4 SDKs/FFI | `w1_a4.txt` (49) | 22:40:19 |
| 5 | W1/A5 numerics/hw | `w1_a5.txt` (66) | 22:40:05 |
| 6 | W1/A6 build/tests | `w1_a6.txt` (43) + `w1_a6_report.md` | 22:40:19 / 22:47:37 |
| 7 | W2/A1 docs & claims | `w2_a1.txt` (98) + `w2_a1_report.md` | 22:57:12 / 23:06:31 |
| 8 | W2/A2 snapshot diff | `w2_a2.txt` (36) + `w2_a2_report.md` | 22:57:12 / 23:07:21 |
| 9 | W2/A3 claim re-verify | `w2_a3.txt` (30) + `w2_a3_report.md` | 22:57:12 / 23:09:39 |
| 10 | W2/A6 coverage & SDK | `w2_a6_report.md` only (no slice) | 23:07:28 |
| 11 | W2/A4 build & security | `sessionID ses_e2efc2c52ffeCYrFUC3fIqZNoO` (this session) | ~23:45 |
| 12 | W2/A5 end-to-end | `sessionID ses_e2efc217affe3bwX8b8q9WwydX` (this session) | ~23:48 |

**Twelve distinct agents did run.** But the **chronology is 6 → 4 → 2**, not 6 → 6:

- Wave 1 (lanes 1–6) is consistent with one parallel batch: all six manifests written within
  **1m 06s** (22:39:13–22:40:19).
- Wave 2 produced **only four** agents' artifacts (lanes 7–10), at 22:57–23:09. Lanes 11 and 12
  have no artifacts from that period.
- Lanes 11–12 were launched **by this session**, roughly 35 minutes later, when it found the W2/A4
  and W2/A5 reports missing from disk.

So the honest statement is: **12 agents total; two batches of six did occur (W1 = 6, and W2's six
lanes span 7–12), but wave 2 was not launched as a single sequential block — it completed in two
pieces, and the last two lanes were a re-launch caused by a missing-report gap, not part of the
original second wave.** The earlier draft's phrase "three waves" was an accurate description of
execution order but obscured this. It is corrected here.

**What cannot be verified:** the exact wall-clock ordering *within* each batch, which agent produced
which on-disk report where only one artifact exists, and whether any W2 lane failed and was retried
before 22:57. Those details are not recoverable from the artifacts, and this report does not guess
at them. Wave-1 lane contents are attested only by their manifests plus two surviving reports.

**Isolation.** Audit repo at `~/.cache/biology-mm/trium-xbt-eval/`. Builds went to `/tmp/*` and
`/tmp/w2a4/*`; scratch to `/tmp/xbt_neg`, `/tmp/sec`, `/tmp/hij*`. Biology MM was touched only at
`COMPILER_EVAL.md`. No commits, pushes, resets, checkouts, stashes, deletes, or global
installs in either repo; `~/.gemini/config` untouched. The clone ends clean at `313f669` with 0
modified and 0 untracked paths.

### 1.2 Inventoried versus deeply read — the distinction that matters

Broad coverage and deep inspection are **different claims**, and only the second can support a
finding about a file's logic.

| Measure | Value |
|---|---|
| Tracked files | 428 |
| Files appearing in some manifest (inventoried/opened) | **428 (100%)** |
| Code files (`.hpp/.cpp/.h/.c/.py/.rs`) | 230 (77,940 lines) |
| **Deeply read line-by-line** (this session's six deep lanes) | **26 files, ≈34,600 lines (≈44%)** |
| Code files ≥400 lines | 60 (50,261 lines) |
| Deeply read among those 60 | 19 |
| **Definitions/oracle JSON, not prose** | `specs/layout_conformance.json` (63,946 lines, 2,100 records) |

Deep reads this session, all line-by-line with recorded ranges: `ternary_frontend_ast.hpp` (2,689),
`libbt.hpp` (1,802), `ternary_llvm_codegen.hpp` (1,695), `ternary_dense_algebra.hpp` (1,625),
`ternary_memory_arena.hpp` (1,495), `ternary_fp.hpp` (1,475), `bt_compiler.cpp` (1,368),
`ternary_jit_engine.hpp` (1,136), `include/Ternary/Blas/TernaryGemmGpu.hpp` (1,022),
`ternary_bigint.hpp` (913), `include/Ternary/Target/XbtInstructionSelector.hpp` (916),
`ternary_core.hpp` (867), `include/Ternary/Blas/TernaryMicroKernel.hpp` (691),
`bt_backend_emitters.hpp` (616), `lib/Ternary/TernaryDialect.cpp` (642),
`test/Unit/test_unit_libbt.cpp` (584), plus the smaller security/codegen files named in §6.3.

**Still NOT deeply read** (≥400 lines, opened but not line-audited): `test/Unit/test_unit_frontend_fp_sparse.cpp`
(1,038), `bench_cycle_accurate.cpp` (733), `verification/xbt_isa.py` (698),
`test/Unit/test_unit_microkernel.cpp` (692), `test/Unit/test_unit_fp_sparse.cpp` (687),
`hw/xbt_alu_model.hpp` (658), `test/Unit/test_unit_bt_opt.cpp` (605),
`test/Unit/test_unit_bigint_symplectic.cpp` (896), `ternary_kokkos_bridge.hpp` (736),
`include/Ternary/IR/GLSL.hpp` (537), `verification/verify_redundant.py` (525),
`bench_comparative_binary_vs_ternary.cpp` (525), `ternary_sparse.hpp` (524),
`test_frontend_ast.cpp` (517), `tools/gen_targets.py` (494), `test/Unit/test_xbt_selector.cpp` (482),
`tools/gen_backend.py` (458), `verification/engine.py` (457), `ffi/rust/src/lib.rs` (456),
`test_ternary_memory.cpp` (456), `verification/verify_tensor_shapes.py` (454),
`test/Unit/test_unit_core.cpp` (441), `tools/heavy_cpu_benchmark.cpp` (432),
`lib/Ternary/ternary_c_abi.cpp` (432), `verification/verify_scalar_mul.py` (421),
`test/Unit/test_unit_mlir.cpp` (416). These remain **inventoried only**; no claim in this report
depends on their internals beyond what a shallow read showed.

**Duplicate paths inflate the tracked count.** `upload trium core files/core_sources/*` is 25 files
that are byte-identical or near-identical copies of root sources (19 of 25 byte-identical, §2 #7),
so several "large file" rows above are the same content counted twice.

**Coverage arithmetic** (reconciled with `git ls-files -z`, not `test -e`):

| Wave | Lanes | Manifest entries | New files added |
|---|---|---|---|
| 1 | A1 front end · A2 IR/MLIR · A3 backends · A4 SDKs/FFI · A5 numerics/hw/proofs · A6 build/tests/CI | 24 · 39 · 44 · 49 · 66 · 43 = 265 | 265 |
| 2 | A1 docs/claims · A2 snapshot diff · A3 aux/scripts · A4 build & security · A5 end-to-end · A6 coverage/SDK | 98 · 36 · 30 (A4/A5/A6 concern-based, no slice) | 163 |
| **Union** | nine file manifests | — | **428 of 428 tracked, 0 uncovered** |

**The phantom entry.** Manifest union is 429 lines, of which exactly one is not a git path:
`lib/SDK/SDK.cpp`, listed in `audit/w1_a4.txt:4`.
`git ls-files --error-unmatch lib/SDK/SDK.cpp` → `error: pathspec ... did not match any file(s)
known to git`. The real file is `lib/SDK/sdk.cpp` (26,124 B), tracked and covered. `test -e
lib/SDK/SDK.cpp` returns **true** anyway because APFS is case-insensitive — a trap for anyone
verifying coverage by existence check. With it excluded: **428 distinct real files covered, 0
tracked-uncovered, 0 duplicates** (each lane's "new" count equals its manifest size).

**Naming hazard:** `audit/w2_a3.txt` is the *auxiliary file-slice* manifest; the agent labelled
"W2 A3" was the **high-risk claim re-verification** lane. Different things, same number.

**What coverage does not mean.** A manifest entry means a file was opened, at whatever depth.
It is not behavioural coverage, and §16 says where the floor is.

---

## 2. Corrections — including to my own previous draft

Each item was re-derived from source or re-measured. The superseded claim is quoted so it cannot be
quietly restated. Rows 1–3 correct the **previous draft of this file**, rows 4–8 the first-pass
report it replaced.

| # | Claim | Status | What is actually true |
|---|---|---|---|
| 1 | "Bare `bt-compiler <file>` prints a fixed notice and **exits 0** without compiling" | **REFUTED (mine)** | It exits **1** with a real error and a usage hint. My earlier reading piped stdout into `head`, which reports `head`'s status. `bt_compiler.cpp:918`. |
| 2 | "`ffi/python/bt_ffi.py` binds **0** callables" | **REFUTED (mine)** | It binds **27 distinct `_LIB.bt_*` symbols** (52 `argtypes`/`restype` statements) and exports **41 public names**. My `dir(module)` filter looked for a `bt_` prefix, but the module exposes unprefixed wrappers (`fp32_to_double`, `dot_product`, `radix3_fft`). With the runtime present, `python3 ffi/python/bt_ffi.py` prints **ALL PYTHON FFI VERIFICATION TESTS PASSED**; ABI reports 1; `fp32_to_double(fp32_from_double(0.75))` = 0.7499999997849021. |
| 3 | "`#32 Legacy_FFI_Hardening` fails on Linux-only assumptions (`.so`/`LD_LIBRARY_PATH`)" | **REFUTED (mine)** | It fails **because it caught a real macOS CWD library hijack**. `bt_ffi.py:48` falls through to `ctypes.CDLL("libbt_runtime.so")` with a bare leaf name, and macOS `dlopen` **does** search the CWD for slash-free names. Demonstrated from a pristine directory: a `libbt_runtime.so` with a constructor that writes a marker file **is loaded and its constructor runs** (§6.3). The test is right; the code is wrong. My earlier "wrong reason" diagnosis was wrong. |
| 4 | "`BTFP<20>` carries **29.11** bits; `BTFP<40>` **60.81**" | **REFUTED (mine, method)** | Single-ULP spacing is binade-dependent: **29.53 → 30.11** bits for `<20>`, **61.23 → 61.81** for `<40>`. 29.11 is the *lower* half-binade count only; 31 is the *upper*. See §5.1 for why no single number is right. |
| 5 | "Native path is slower than `clang -O3` itself (442 ms vs 0.39 s)" | **REFUTED (mine, attribution)** | The number is real but the cause was misattributed. The loop's *execution* is not slow: bt's own emitted IR, compiled and run, takes **2.9 ms** for 20 M iterations vs **2.2 ms** for hand-written C. The ~340 ms in `--native` is **macOS `dlopen` of a just-written dylib** (≈295 ms) plus clang (≈43 ms). See §5.4. |
| 6 | "`BTFP` has only `int64_t` constructors" | **PARTLY REFUTED** | Constructors accept floating types; `BTFP<20> v(3.5)` compiles and **silently yields 3**. The bug is real and worse than described: silent truncation, diagnosable only with `-Wall` (`-Wliteral-conversion`). |
| 7 | "Only 6 of 25 core sources are byte-identical" | **REFUTED (inverted)** | **19 of 25 identical**; 6 differ, and every diff is *additive in the live tree*. No code exists only in the snapshot. |
| 8 | "No Rust CI job exists" | **REFUTED** | `.github/workflows/core-foundation.yml:241` defines `rust-client` (`cargo test`, `:254`), and it passes. |
| 9 | "`--test` runs an internal invariant suite" (as implied by §6.2 and `bt_compiler.cpp:537`) | **REFUTED (deep read)** | In Release the four checks are **`assert`-only** (`bt_compiler.cpp:474,479,487,495`) and `bt-compiler` is built with `-DNDEBUG` and **no `-UNDEBUG`** — `CMakeLists.txt:193` applies the flag only to targets declared after line 193, while `bt-compiler` is at `:157`. Verified: the assertion strings are **absent from the Release binary** (`strings … \| grep -c "Dual-rail invariant failed"` → 0) while it still prints "ALL COMPILER SELF-TESTS PASSED WITH 0 FAILURES" and exits 0. The self-test is vacuous. |
| 10 | "`HardwareMMUZeroMatrix::reset()` is advisory `madvise`; stale data survives" | **CONFIRMED, and worse than stated** | `MADV_DONTNEED` **is** defined on darwin, so the `MADV_FREE` branch (`:1472`) is unreachable. Measured directly: a page filled with `0xAB` **still reads `0xAB`** after `madvise(MADV_DONTNEED)` returns 0. On a live arena, `set_byte` then `reset()` then `get_byte` returns the **old bytes** (`ab cd` → `ab cd`). `reset()` does not clear. |
| 11 | (implicit) "the type checker makes the native path safe" | **REFUTED (deep read)** | `--native` lowers `+ * -` through `check_integer` (§ below), but the **default fallthrough emits a bare unchecked `add i64`** (`ternary_llvm_codegen.hpp:1018`), and `@bt_error_state` is **never read after `fn()`** (`:1683-1689`). An int39 overflow prints "Execution completed" with an **empty stderr** and rc=1 — no message at all, while `run` and `--core-run` name the overflow exactly. |
| 12 | "39-trit integers, scalar-only" | **NUANCED** | Correct for `Int`. But `PackedWord` carries up to **64 trits** (`ternary_frontend_ast.hpp:2618`), and `TritMatrix`/`SparseMatrix`/`tritword` are real types (`:826-828`) with matrix ops (`:2580-2586`). "No arrays/subscripting" and "no `for`/`/`" all hold; "everything is a 39-trit scalar" does not. |

Nothing else was contradicted: the no-`for`/no-array/no-`/` result, the clang-delegation back end,
the FFT existence-and-slowness result, the absence of silicon, and the §13 ranking all survived.

---

## 3. What the project actually is

| Component | What it is | Evidence |
|---|---|---|
| `.bt` front end | Hand-written lexer/parser → AST → typed core. `while` works; **no `for`, no arrays/subscripts, no `/`**. Integers are 39-trit | VERIFIED (§4.1) |
| IR / dialect | Custom **text-level** dialect named "bt"; not upstream MLIR (`MLIR_DIR-NOTFOUND`, `BT_HAS_MLIR` false) | VERIFIED (§6.1) |
| Native back end | IR text → host `clang` (`-O3 -shared -fPIC`, `bt_process.hpp:158`) → dylib → `dlopen`/`dlsym("bt_main")`. Not a JIT | VERIFIED |
| `bt-opt` | Statically carries the project's **own** `bt::mlir` shim (160 symbols); `otool -L` = libc++/libSystem only; **0** upstream verifier symbols | VERIFIED |
| `--core-run` | AST-tree interpreter with a **hard 10 M-tick budget**; 9 ticks per loop iteration | VERIFIED (§4.3) |
| "xbt" target | 64-trit dual-rail; `--list-targets` reports "scalar multiply: no (lane-wise only)". **1** registered backend + 4 YAML target descriptions | VERIFIED |
| `BTFP<W,E>` | Balanced-ternary FP emulation, int64 storage; `from_double`/`to_double` at `:910`/`:870` | VERIFIED (§5.1) |
| Radix-3 FFT | Real Cooley–Tukey in `libbt.hpp:1182` (`transform_inplace` `:1228`, reference DFT `:1432`), unit-tested `test/Unit/test_unit_libbt.cpp:314-399` | VERIFIED |
| C / C++ SDK | `include/Ternary/SDK/sdk.h` (26 prototypes) + `SDK.hpp`; links `libtrium_sdk` | VERIFIED (§5.3) |
| Python FFI | `bt_ffi.py` is **functional** (27 bound symbols, 9 suites pass) once a runtime is built | VERIFIED (§2 #2) |
| Python SDK | `trium/` requires an **absolute** library path; bare `trium.Context()` raises | VERIFIED (§5.3) |
| Rust SDK | Real binding; 15 tests pass with `BT_LIB_DIR` | VERIFIED |
| GPU back ends | **No GPU binary is ever produced.** `bt_compiler.cpp:918` throws for Metal/Vulkan/x86/arm64 emit | VERIFIED (§7) |
| IREE plugin | Text-emitting stub — **0 `iree::` symbols**, no `find_package(IREE)` | VERIFIED |
| Hardware | 615-line SystemVerilog RISC-V XBT sketch; no testbench, no constraints, no synthesis | DOCUMENTED-CLAIM (§8) |
| Formal | Lean 5 files / **64** theorems, compile, **0** `sorry`/`Admitted`/`Axiom`; **no CMake target, no CI job**. Rocq: **61** theorems, wired (`Core_Rocq`, CI `native-rocq`) but skips locally (exit 77) | VERIFIED (§8) |
| Own thesis | README: advantage is memory bandwidth, not FLOPs; GPU "dispatch pending"; RTL not measured | DOCUMENTED-CLAIM |
| Exotic marketing | Zero hits repo-wide for `clockless`, `durail`, `nonary` | VERIFIED (grep) |

---

## 4. The language, exercised

### 4.1 What it rejects (all VERIFIED, exact diagnostics)

```text
var a = 1 / 2;      → 4:19 Expected ';' after variable declaration (found: ERROR '/')
for (...)           → 3:8  Expected expression, got KEYWORD(var) ('var')
var a[3];           → 1:6  DELIMITER('[')
int main() {}       → 1:5  Expected ';' after expression statement
```

Integer domain is **39 trits** (`Type::Int39`), max literal `2026277576509488133` = (3³⁹−1)/2; the
next value is rejected with "integer literal outside the 39-trit scalar domain". This is not a
64-bit integer in costume — arithmetic silently has a different range. `5/2` has no operator form;
`-` on trits is unary negation.

The restriction is structural, not merely syntactic: `ternary_frontend_ast.hpp:2526,2531` and
`ternary_ir_lowering.hpp:18` both state scalar-only.

**All 12 example and regression programs exit 0** under `run` (7 examples + 5 regression). The
shipped corpus is internally consistent and does work; it is the wrong corpus for this job.

### 4.2 Dialect emit — two real defects

**Defect A — script-style input produces a silent empty module.** `--emit-bt-dialect` accepts only
`func`-style programs. Given `var a = 1; var b = 2; print(a+b);` it exits **0**, writes **123 bytes**,
and reports `(0 ops, 0 function(s))`, with **0 `func.func` and 0 `bt.` ops**. No error, no warning,
success exit code — a downstream consumer compiles an empty module and never knows.

**Defect B — multiply is refused in dialect mode but silently accepted elsewhere.**
`--emit-bt-dialect examples/06_core_cuda_map.bt -o ...` → **exit 1**:

```text
Error: bt dialect lowering incomplete for 1 function(s):
  transform: bt dialect: scalar multiplication is not implemented
  (bt.lane_mul drops inter-lane carries); refusing to emit incorrect code
```

The *same file* under `--native` → **exit 0** (≈415 ms), under `--core-run` → exit 0, under `run` →
exit 0. One lane is honest about the carry-dropping limitation; three produce output anyway. Credit
where due: the refusal message is a model of honesty. The inconsistency is the defect.

**CLI contract:** `--emit-bt-dialect` takes `-o <out>`. The positional form `--emit-bt-dialect IN OUT`
exits **1** with `Error: multiple input files are not supported` and writes nothing.

**`--convert-all` is not idempotent.** `annotateFirstOpSite`
(`lib/Ternary/TernaryPasses.cpp:38`) has no already-annotated check. This is *first-textual-occurrence*
annotation with operand-independent templates — not canned output. FileCheck asserts both positive
patterns and `CHECK-NOT`, so tests are not vacuous; they are semantically weak.

**`--trace` does not exist** → exit 1, `Error: unknown option '--trace'`.

### 4.3 `--core-run` has a hard, undocumented-in-practice budget

`bt_compiler.cpp:823`: `executionBudget>0 ? executionBudget : 10000000`. The tick is at
`include/Ternary/IR/Execute.hpp:21`, called per statement and per back-edge. Bisected on this host:

| Loop body | Last N that passes | First N that fails | Ticks/iteration |
|---|---|---|---|
| `i = i + 1;` only | 1,111,109 | **1,111,110** | 9.00 |
| `s = s + i; i = i + 1;` | 769,229 | **769,230** | 13.00 |

Consequence: `--core-run` cannot execute any loop above ~1.1 M iterations — including every MD-like
workload — and fails with `Error: typed core execution budget exceeded`. `--help` says
`0 = unmetered default`, which is **false**: `--execution-budget=0` still trips at the 10 M default.
`--execution-budget=100000000` does permit a 2 M-iteration loop. VERIFIED.

---

## 5. Numerics, measured

### 5.1 `BTFP` precision and speed

Balanced-ternary FP is **not uniformly spaced**: representable steps change by 3× across each
binade, so "mantissa bits" is a range, not a scalar. Measured by exact ULP probing and by counting
representable values in the `[1,2)` binade:

| Type | ULP at 1.0 | ULP at 2.0 | Effective bits (best → worst) | Values in `[1,2)` |
|---|---|---|---|---|
| `BTFP<20>` | 2^-30.114 | 2^-29.529 | **30.11 → 29.53** | 774,840,978 (log2 = 29.53) |
| `BTFP<40>` | 2^-61.814 | 2^-61.229 | **61.81 → 61.23** | 2,701,703,435,345,984,178 (log2 = 61.23) |
| `double` | 2^-52 | 2^-51 | 52 → 51 | — |

The earlier "29.11" counted only the lower half-binade (581,130,733 values); "31" counted the upper
(193,710,244). Both are half-answers. **Round-trip error**, 200,000 samples in `[0.5, 2)`:
`BTFP<20>` max relative **8.57e-10** (2^-30.12), mean 2.81e-10; `BTFP<40>` **exactly 0** on all
samples (it has more precision than the 53-bit input at these magnitudes).

**Speed** (20,000 × `s = s + a*b`, best of 5, same binary, two runs to show variance):

| Type | ns/op | vs `double` |
|---|---|---|
| `double` | 0.56–1.08 | 1× |
| `BTFP<20>` | 118–264 | **~200–250×** |
| `BTFP<40>` | 183–270 | **~210–330×** |

The arithmetic is *correct*; it is slower and, at usable widths, narrower than the hardware it
replaces. Ratios are the transferable claim; absolute ns are host-specific.

**Silent truncation (VERIFIED):** `BTFP<20> v(3.5)` compiles and yields **3**. `-Wall` gives
`-Wliteral-conversion`; `-Werror` fails the build. Default flags: silent. Use
`BTFP::from_double` (`ternary_fp.hpp:910`), which gives 3.5000000012905872.

### 5.2 FFT

A correct, unit-tested textbook radix-3 Cooley–Tukey. But it requires **N = 3^k**
(`is_power_of_3(64) == 0`), so a 64³ PME grid becomes 81³, and it measures slower than a naive
scalar radix-2 at matched sizes. Seven runs, best-of and full band:

| N (3^k) | Radix-3 best | Radix-2 best | Ratio (best) | Ratio band |
|---|---|---|---|---|
| 3^6 = 729 | 0.012 ms | 0.010 ms | 1.20× | 1.13–1.78× |
| 3^8 = 6561 | 0.118 ms | 0.087 ms | 1.36× | 1.30–1.42× |
| 3^10 = 59049 | 1.225 ms | 0.914 ms | 1.34× | 1.34–1.44× |
| 3^12 = 531441 | 13.079 ms | 8.948 ms | 1.46× | 1.46–1.67× |

3-D reality, three runs: radix-2 64³ = **16.7–25.7 ns/point**, radix-3 81³ = **35.7–49.2 ns/point**
→ radix3 is **1.92–2.13× more work per grid point**. We already use FFTW (`PME.jl:205-206`), which is
multithreaded and SIMD — the gap widens further (INFERRED; no Julia/FFTW toolchain on this host, so
the comparison is against a same-style scalar proxy, and *only* the proxy numbers are VERIFIED).

### 5.3 SDK and FFI runtime checks

26 `_bt_sdk_*` exported symbols ↔ 26 prototypes in `sdk.h`, set-identical. Surface counts C 32 /
Python 49 / Rust 76. `tools/check_sdk_parity.py --check` → **OK (27 capabilities, 15 justified
asymmetries)**, exit 0, working tree unchanged. The tool **rewrites `docs/SDK_PARITY.md` unless run
with `--check`** — never run it bare in a CI that diffs docs.

Personally compiled and ran:

- **C:** `clang -std=c11 -Iinclude examples/sdk/main.c -Lbuild -ltrium_sdk` → exit 0;
  `transform: 0 1 2`, `sqrt(40/9)=3`, `norm2(40/3,40/4)=5`.
- **C++:** `clang++ -std=c++20` → exit 0, `CPU SDK and exact-ratio BTFP input: 0.5`.
- **Rust:** `cargo test` → 15 passed (needs `BT_LIB_DIR`). Bare `cargo test --offline` fails to link.
- **Python FFI:** builds and runs; **all 9 suites PASSED** including radix-3 FFT round-trip, T-CSR
  SpMV, and the non-linear engine. Bound via `BT_RUNTIME_LIBRARY` (absolute path).
- **Python SDK (`trium/`):** requires an absolute path
  (`ffi/python/trium/__init__.py:35-38`); a bare `trium.Context()` raises.

Header is `<Ternary/SDK/sdk.h>`; there is no `ternary_sdk.h`.

### 5.4 Where `--native` time actually goes (corrects §2 #5)

Instrumented breakdown of one `--native` run, tiny program, `bt::host::clang_command()` verbatim:

| Stage | Time |
|---|---|
| parse | 0.53 ms |
| codegen to IR text | 0.47 ms |
| write `.ll` | 0.67 ms |
| **`clang -O3 -shared -fPIC`** | **43–46 ms** |
| **`dlopen` the fresh dylib** | **295–348 ms** |
| `dlsym` | 0.02 ms |
| run `bt_main` | 0.001 ms (trivial program) |

Total ≈ 340 ms, matching the reported `[Native JIT] Execution completed in ~345 ms`. The clang
subprocess is only 13% of it. The dominant term is macOS `dyld` loading a dylib written microseconds
earlier: the *same file* reopened takes **0.2 ms** (dyld cache), and a `cp` of an identical dylib
still costs **40 ms**. Independent confirmation: a bare `dlopen` of a fresh 33 KB dylib costs
270–295 ms; of a trivial one `clang -shared` just wrote, 270–295 ms too — it is a filesystem/dyld
effect, not a property of bt's output.

**Code quality of the emitted IR is fine.** It is naive (every local is `alloca`/`store`/`load`;
**0 vector types**; no `phi` in `bt_main`), but `clang -O3` optimizes it well: bt's own emitted IR for
a 20 M-iteration loop, compiled and run, takes **2.9 ms** vs **2.2 ms** for equivalent hand-written
C. So: the emitted code is competitive; the *delivery mechanism* costs ~340 ms per invocation, which
is fatal for short runs and irrelevant for long ones.

### 5.5 Codegen and runtime correctness defects (deep read, new this pass)

These come from reading the emitter and the tree-walk JIT in full, and each was reproduced.

- **Silent native overflow.** `--native` on `mul(2000000000000000000, 2000000000000000000)`:
  `run` and `--core-run` both print
  `integer overflow in multiplication: … exceeds the representable balanced-ternary range
  [-2026277576509488133, 2026277576509488133]`, but `--native` prints
  `[Native JIT] Execution completed in 382.232 ms` with **empty stderr** and rc=1. Mechanism: the
  overflow flag is stored to `@bt_error_state` in IR (`ternary_llvm_codegen.hpp:116`) but never read
  after `fn()` (`:1683-1689`); the only signal is the process exit code. A CI that runs `--native`
  would see a red exit with no explanation, and a user watching stdout would see success.
- **Unchecked fallthrough op.** `ternary_llvm_codegen.hpp:1018` — any `BinaryOp` not in the explicit
  cases emits a bare `add i64`. `check_integer` (`:106-113`) guards only `+ - *`, widening via
  `sext i64→i128` and range-checking against `contract::integer_min/max`.
- **Arena `reset()` does not reset.** §2 #10. Also `allocate_slow_path`
  (`ternary_memory_arena.hpp:570-579`) sets `c->used = 0` with **no `memset`**, unlike `reset()` at
  `:734`, so the lookahead path hands out dirty memory.
- **`TritWord64` invariant is `assert`-only** (`ternary_core.hpp:103`), with `P`/`N` as **public
  mutable members** (`:98-99`) — any `w.P |= mask` silently breaks disjointness, and
  `ternarySelect` (`:354-366`) then ORs both rails.
- **Bigint silently drops capacity.** `add` discards the final carry (`ternary_bigint.hpp:190-197`);
  `mul` drops terms with `k >= 64*NumWords` (`:225`); `shift_trits_left` returns zero past capacity
  (`:146`). Only `to_int64` (`:820-824`) guards.
- **Shared error slot.** `task_error_slot()` (`ternary_kokkos_bridge.hpp:204-226`) is a
  process-global, not thread-local, so overlapping parallel dispatches can rethrow each other's
  exception.
- **Unreachable/dead code.** `parse_comparison` alias (`ternary_frontend_ast.hpp:2139-2141`),
  `LoopNode` (`:1055`), `FunctionCallNode` (`:1147`), ~50 duplicated lines (`:1795-1859` vs
  `:2281-2344`), and a `KwSparseMatrix` fallthrough (`:1787-1791`). `Lexer::diagnostics()` (`:370`)
  is populated but **never consumed**, so lexer diagnostics cannot reach the user.
- **Minor, non-exploitable:** emitted identifiers are not escaped (`%arg_<name>`, `@bt.user.<name>`,
  `ternary_llvm_codegen.hpp:614-617,661`), but the lexer restricts identifiers to
  `[A-Za-z0-9_]` (`:519`), so injection from source is unreachable.


---

## 6. Build, tests, CI, and what the suite actually covers

### 6.1 Build (VERIFIED)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # configures clean, no MLIR needed
cmake --build build -j8                     # exit 0, 51 targets
ctest --test-dir build -N                   # 82 registered
```

Two independent clean Release builds into `/tmp/w2a4/b1` and `/tmp/w2a4/b2` both exited 0 with **55
compile lines / 51 links / 52 targets / 82 tests** each. Of 53 Mach-O + ar artifacts: **43
byte-identical**; `bt-compiler`, `bt-opt`, `libtrium_sdk` byte-identical. The 10 that differ do so
via `ar` member mtimes, `LC_UUID`/adhoc `CDHash`, and **1 byte of `LC_RPATH`** (`b1`→`b2`).

**Network — PASS:** a `sandbox-exec` build with networking denied succeeded (control `curl` in the
same profile exited 6). No `FetchContent`, `ExternalProject`, or `file(DOWNLOAD)` anywhere.

**Artifact hygiene — FAIL:** 8 artifacts embed the absolute build path (`benchmark_arithmetic`,
`benchmark_wide_math`, `libbt_runtime.dylib`, `unit_c_abi_ffi`, `unit_numerical`, `unit_phase1`,
`unit_sdk_gemm_layout`, `unit_sdk_host`). Cause: `CMakeLists.txt:150` and `:306` set
`BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"` without `BUILD_WITH_INSTALL_RPATH`. `bt-compiler` and
`libtrium_sdk` are clean; the installed `lib/libtrium_sdk` is clean (`@rpath`, no `LC_RPATH`). All
binaries are adhoc linker-signed with **no hardened runtime** (`0x10000` absent).

### 6.2 Tests — the three readings, honestly

| Build dir | Passed | Skipped | Failed | Total time |
|---|---|---|---|---|
| `/tmp/xbt_build_a6` | **70** | **11** | **1** | 60.5–63.3 s |
| existing `build/` (earlier run) | 70 | 11 | 1 | 62.05 s |
| an earlier environment | 75 | 6 | 1 | — |

The single failure is constant: **`#32 Legacy_FFI_Hardening`** — and it is a *true positive* (§2 #3,
§6.3). Skips are environment-gated (`#56 Core_Rocq`, `#67 SDK_Rust_FFI_ABI`, `#44/#45/#46`
differentials, `#60/#61`, `#80/#81` Vulkan). `#44–#46` skip with `SKIP: independent host adapter
requires Linux and C++`. Any future report must state its build dir.

**Five vacuous or weak tests** (the honest ceiling on what the suite certifies):

- `test_unit_kernel_isa_paths.cpp` — silently skips 2 ISA lanes, exits 0;
- `test/SDK/package.py` — **0 assertions** in 18 lines (runs as CTest `#82`);
- `test_unit_carry_free.cpp:306` — `(void)resolve_parallel(...)`, return value discarded;
- `test_translation_validation.cpp` — bespoke `g_fails` counting, not a standard harness;
- `test_xbt_conformance.cpp` — zero assertions, reports only totals.

**A sixth, and the most misleading: `--test` itself.** See §2 #9. The four "invariant tests" are
`assert`-only, `bt-compiler` is compiled with `-DNDEBUG`, and the flag that would restore them
(`-UNDEBUG`, `CMakeLists.txt:193`) is scoped to targets declared after line 193 — `bt-compiler`
is at `:157`. The assertions are provably absent from the Release binary, yet it prints
"ALL COMPILER SELF-TESTS PASSED WITH 0 FAILURES" and exits 0. Anyone citing `bt-compiler --test`
as evidence is citing a no-op. This is the single strongest argument in this report for treating
"the project's own tests pass" as weak evidence.


**`bt-opt` validation is braces-only.** An unknown op (`"bt.not_a_real_op"`), an undefined SSA use,
and wrong arity on a real `bt.*` op all exit **0**, contradicting its own help text that dialect
errors abort. Unbalanced braces do exit 1 (`unbalanced braces: missing 2 closing brace(s)`).

**Gating defect:** `CMakeLists.txt:458` gates on `BT_ENABLE_MLIR` rather than `BT_HAS_MLIR`, so a
user who requests MLIR without it available gets confusing behaviour rather than a clear "not
found". Exactly **1** of the 82 `add_test` calls is MLIR-gated (`:459`). Also
`CMakeLists.txt:193` sets `-UNDEBUG`, which is what saves `bt_compiler.cpp:493`, where `unpacked`
is referenced only inside `assert()`.

### 6.3 CI and security posture

- **0 network operations** during the build (VERIFIED, sandboxed).
- No secrets in workflows; `permissions: contents: read`. Good.
- **All 12 GitHub Actions are tag/branch-pinned, 0 SHA-pinned** — `checkout@v4` ×6,
  `setup-python@v5` ×4, `ocaml/setup-ocaml@v3`, and `dtolnay/rust-toolchain@stable` (a mutable
  branch) at `:245`. A real supply-chain exposure; PR code does execute in CI.
- `LSAN_OPTIONS: exitcode=0` (`:102`) — leak findings will not fail the build. Deliberate, but
  "green" does not mean "leak-free".
- CI is `ubuntu-24.04` only (6 jobs) — **no macOS job**, despite every defect above being a macOS
  defect.
- **`lib/SDK/VulkanRuntime.cpp:29`** `dlopen`s `getenv("BT_SDK_VULKAN_LIBRARY")` with **no
  absolute-path or ownership validation**, unlike `NativeRuntime.cpp:31-34` which does. An
  env-var-controlled library load.
- **`ffi/python/bt_ffi.py:48` — genuine CWD library hijack on macOS.** After the four absolute
  search paths miss, the fallback is `ctypes.CDLL("libbt_runtime.so")` — a bare leaf name, and macOS
  `dlopen` resolves slash-free names against the **current working directory**. Demonstrated from a
  pristine `/tmp/sec`: a `libbt_runtime.so` whose constructor writes `PWNED` **is loaded and the
  constructor runs**. A C `dlopen` of a bare leaf name reproduces it in 3 lines. Setting
  `DYLD_LIBRARY_PATH` (or the test's `LD_LIBRARY_PATH`, which the audit harness exports) suppresses
  it — which is exactly why the code comment "never the current directory" is wrong and the test is
  right to fail. The fix is to reject bare names outright or resolve via an absolute path only.
- `system()`/`popen()` appear only in 2 tests: `test_unit_xbt_hw.cpp:202` runs a PATH-derived
  `verilator` unquoted; `test_unit_frontend_fp_sparse.cpp:760` uses a literal prefix (safe).

---

## 7. Back ends and end-to-end runs

**GPU claims are false as stated.** `bt_compiler.cpp:918` throws on `--emit-metal` / `--emit-vulkan`
/ `--emit-x86` / `--emit-arm64` ("emit a FIXED kernel template ... do NOT compile your .bt source"),
and `:925` on GPU binaries. No GPU binary is produced. `benchmark_portable_gpu.py` reports `runs=3`,
**`devices: 0`**. The IREE plugin is a text stub: 0 `iree::` symbols, no `find_package(IREE)`.

**Deep read of the GPU sources confirms and sharpens this.** `include/Ternary/Blas/TernaryGemmGpu.hpp`
(1,022 lines, read in full) contains real GLSL 450 kernel *strings* (`:60-145`, `:155-239`,
`:705-859`) and host-side C++ emulators (`:377-479`, `:878-981`), but
`ternary_gemm_gpu()` (`:1001-1019`) **never allocates device memory and never launches a kernel**:
`require_gpu` throws `GpuUnavailable` (`:1003-1006`), otherwise it returns `r.ok=false` and computes
the CPU reference (`:1012`). There are **zero** CUDA/HIP/Metal/Vulkan calls anywhere in the file.
To its credit it does *not* falsely claim GPU — disclaimers at `:7-16` and `:990-999` — but the
`gmacs` figure (`:1016-1017`) is a **CPU** throughput number.

`bt_backend_emitters.hpp` (616 lines, read in full) splits cleanly: **real** emitters are
`ternary_logic64` (`:255-288`), `ternary_logic512` (`:290-295`), `dotProduct128_neon` (`:301-328`),
CSA (`:51-238`); **fixed templates that ignore their input** are Metal (`:340-392`, note
`(void)cfg` at `:341`), Vulkan (`:398-453`), x86 AVX-512 (`:459-484`), and CUDA PTX (`:490-608`).
No emitter throws — the refusal happens earlier in the driver.

**The ISA selector has one correctness-relevant weakness.** `native-scalar-mul`
(`XbtInstructionSelector.hpp:473-487`) emits the *identical* `GROUP2/MUL` encoding as lane-wise
`lane-mul` (`:492-508`); the advertised "scalar multiply: no (lane-wise only)" guarantee lives
solely in `matches`→`is_single_trit` (`:357-359`), which requires a *literal* in {−1, 0, 1}. Nothing
validates rd/rs1/rs2 ranges (they are silently masked, `:117-127`), and `materialise` silently
truncates constants beyond 40 trits (`:828`). The micro-kernel does use real SIMD — AVX2 (`:156-196`,
guard `:69-72`), AVX-512 (`:198-217`, needs `__AVX512VPOPCNTDQ__` **and** `__AVX512F__`), NEON
(`:219-243`) — with correct `__builtin_cpu_init()` before `supports` (`:118`) and unaligned-safe
loads (`loadu` at `:163-164`, `:205`; `vld1q_u64` at `:226`). One real gap: forcing
`GemmOptions::isa` (`:505,510/516`) bypasses CPUID, so requesting `Isa::Avx2` on a non-AVX2 host
raises an illegal instruction.


**Lanes that do emit real code:** `--emit-llvm`, `-S`, `-c`, `--emit-asm`, `--emit-obj`,
`--emit-llvm-bc` (real Mach-O / bitcode, exit 0), plus `run`, `--native`, `--core-run`. Five target
descriptions are listed with `xbt` first ("64-trit dual-rail", "scalar multiply: no (lane-wise
only)"); exactly **1** backend is registered.

**End-to-end (VERIFIED):**

| Program | `run` | `--native` | `--core-run` | `--emit-llvm` |
|---|---|---|---|---|
| `03_while_loops.bt` | 0, `6`, `720` | 0, same | 0, same | 0, 4357 B |
| `04_floating_point.bt` | 0, `4.75 / 2.25 / 4.375` | 0, same | 0, same | 0, 4060 B |
| `06_core_cuda_map.bt` | 0, **no output** | 0, **no output** | 0, **no output** | 0, 4456 B |

`06` prints nothing because it only *defines* `transform` and never calls it; its `bt_main` is
`ret i32 0`. Exit 0 on an empty run is honest here, but it is also why "it runs" proves little.

**Script-style end-to-end** silently emits a 0-op / 0-function module with exit 0 (§4.2 Defect A).
**Bare `bt-compiler <file>` exits 1** with a template-notice error (§2 #1). `--trace` does not exist.

---

## 8. Formal methods and hardware

**Lean:** 5 files, **64** theorems (`BalancedTernary` 16, `HamiltonianTernary` 9,
`TernaryAlgebraProofs` 24, `TernaryCompilerMath` 7, `TernaryPhase3Math` 8), all compile, and
**0** occurrences of `sorry` / `Admitted` / `Axiom` across the Lean and Rocq sources. But there is
**no CMake target and no CI job** — the proofs are never invoked by build or test. They are
documentation in a rigorous notation, not a gated invariant.

**Rocq:** **61** theorems across 4 `.v` files, properly wired — CTest `#56 Core_Rocq`, CI job
`native-rocq` (`core-foundation.yml:148`) with `opam install rocq-core.9.1.1`. Locally it skips
(exit 77) because no Rocq runtime is installed. `verification/rocq_bridge` → `BRIDGE: PASSED`,
exit 0. So: wired, credible, **not executed in this evaluation**.

**Hardware:** `hw/riscv_xbt_extension.sv`, 615 lines, no testbench, no `.xdc`/`.sdc`/`.qsf`, no
synthesis or FPGA result. The repo says so itself
(`docs/CHIP_HARDWARE_RESEARCH_AND_PLAN.md:88`). `physical-gate.json` contains
`"physical_execution": "not_performed"`. INFERRED: nothing here reaches silicon in our horizon.

---

## 9. Claim audit — docs vs. code

The docs are the weakest part of the project.

- **Metal/Vulkan/CUDA "produce running GPU binaries"** — FALSE (§7).
- **ch06 "MLIR/GPU pipeline"** — FALSE; the MLIR is a custom text dialect and the GPU path throws.
- **"Z3/Lean prove kernels are bug-free"** — MISLEADING. Lean is unwired (§8); `z3core.py` is inert
  (no `__main__`, no z3 import).
- **"104.53 GTOPS on Apple Silicon ARM64"** — UNVERIFIABLE. Every archived benchmark JSON is
  `Linux-6.1.158-x86_64`; the `cntvct_el0` cycle numbers have no checked-in evidence, only
  `bench_cycle_accurate.cpp` (733 lines). "743 GTOPS" appears only in the upload dossier.
- **"1-cycle hardware `umulh`"** — FALSE: zero `umulh` occurrences repo-wide.
- **`SHA256SUMS`** — 19 of 23 entries verify; 3 files missing, and the checksum file's own entry
  does not match.
- **"66 proofs"** — overstated: 61 in Rocq; the doc's own 28-in-`BalancedTernaryFoundations.v`
  figure matches.
- **`build_docs.py`** (278 lines) is wired to neither CMake nor CI.
- **`test/audit_regressions.py`** → PASS, exit 0 (real and green).
- **"All compiler self-tests pass"** — VACUOUS in Release; the assertions are compiled out (§2 #9).
- **"`reset()` returns the matrix to zero"** — FALSE on darwin; stale trits are readable after
  `reset()` (§2 #10, §5.5).
- **Three byte-identical copies of the 4,325-line guide.** `docs/Next_Gen_Foundation_Compiler_Guide.md`,
  `NEXT_GEN_FOUNDATION_COMPILER_COMPLETE_GUIDE.md`, and `TRIUM_XBT_FOUNDATIONAL_COMPILER_COMPLETE_GUIDE.md`
  share SHA-256 `2fa40b305bbd…`; the three `.html` counterparts share `4d16453d70…`. That is
  **≈25,950 duplicated lines** of generated output tracked in git, which also inflates any
  "documentation size" claim.
- **Snapshot fidelity:** 19/25 byte-identical; all 6 diffs additive in the live tree (§2 #7).
- **Stale orphans:** root `./TernaryDialect.td` and `./TernaryOps.td` are never compiled —
  `CMakeLists.txt:68` points at `include/Ternary/TernaryOps.td`.
  `include/Ternary/Detail/ternary_fp.hpp` is a 198-byte forward stub to the real root file.

---

## 10. Auxiliary scripts and orphan fixtures

- `lib/SDK/sdk.cpp` is a **real** 270-line SDK v1.2.0 implementation — not a stub.
- Five root `test_*.cpp` fixtures compile and pass manually; **no target, script, or CI job
  references them**. Orphans.
- `verification/gen_vectors` → `total: 34412 vectors`, exit 0.
- `verification/rocq_bridge` → `BRIDGE: PASSED`, exit 0.
- `z3core.py` — inert.
- `benchmark_portable_gpu.py` → `runs=3`, `devices: 0` (§7).

---

## 11. Expressiveness: the decisive point

MD needs (a) the LJ neighbour loop `for i<j` with cutoff, minimum image, and cell lists; (b) the PME
B-spline charge spread over a 64³ grid; (c) constraint/STEP integrators. The language has **no
`for`, no indexing, no arrays, and no division operator** (VERIFIED, §4.1), and the restriction is
structural in the IR (`ternary_ir_lowering.hpp:18`). **None of (a)–(c) is expressible in `.bt`.**
Every other finding is downstream: even a perfect back end cannot compile a loop the parser rejects.

---

## 12. Why it cannot accelerate our MD

- **Expressiveness** — no loops, no arrays, no division (§11, VERIFIED).
- **Compiler** — emits IR text, spawns host `clang`, `dlopen`s. **0** LLVM and **0** upstream-MLIR
  symbols linked; ≈340 ms per `--native` invocation of which ≈295 ms is macOS `dlopen` (§5.4).
  Emitted code itself is competitive once compiled (2.9 ms vs 2.2 ms for the same loop).
- **Ternary FP** — 29.5–30.1 bits at ~200–250× `double` cost; 61.2–61.8 bits at ~210–330×, and
  `BTFP<20> v(3.5)` silently equals 3.
- **FFT** — 1.2–1.5× slower than naive scalar radix-2, 1.9–2.1× more grid work, and we already use
  FFTW.
- **Hardware** — unsynthesised sketch, `"physical_execution": "not_performed"`.
- **Their own thesis** — "advantage is memory bandwidth, not FLOPs" — is untested on our workload.
  INFERRED: 912.7 ms/step is dominated by neighbour-loop FLOPs and FFT, both already
  float64-optimised.
- **Runtime diagnostics** — the native lane reports overflow with a success-shaped message and an
  empty stderr; `reset()` on the zero-matrix arena leaves stale data; `--test` is a no-op in Release
  (§5.5, §2 #9–#11). Even if the language could express the workload, these would make it unsafe to
  trust in a production loop.


**Phase 5 tuning section: SKIPPED — non-starter**, per the eval protocol.

---

## 13. Counter-proposal ranking (what would actually cut the 912.7 ms/step)

Grounded against the current `julia` tree.

1. **Profile the step** (gate; nothing below is sized without it). `RUN.md` reports throughput only —
   912.7 ms/step at 1728 waters — and **no per-component breakdown exists in the repo** (INFERRED
   from absence). One profiled step separating LJ real-space, PME (spread/FFT/sum), BAOAB O-step,
   RATTLE projections, and drift tells us where the 912.7 ms actually is.
2. **Multithread the LJ kernel — already in-tree, one flag.** `StreamedLJ` has a deterministic
   per-cell `Threads.@threads :static` kernel (`StreamedLJ.jl:833-834`) behind an opt-in
   `threaded=true`, active only when `Threads.nthreads() > 1` (`:565`). `water_field` defaults
   `threaded = false` (`WaterBox.jl:501`) and the 912.7 ms/step run never passes it
   (`run_tip3p.jl:322`). Cheapest real win: `water_field(water; threaded = true)` +
   `JULIA_NUM_THREADS=n julia …`. PME stays serial (FFTW plan single-threaded, `PME.jl:205`), so
   expect gain on the LJ share only.
3. **Cache the neighbour/cell list.** `water_field` builds the LJ factor with `skin = 0.0`
   (`WaterBox.jl:517`), so `StreamedLJ` rebuilds the cell linked list **every step** (`:542-562`). A
   skin-cached neighbour list exists and is tested — `Neighbors.build_neighbors` /
   `needs_rebuild!` (`Neighbors.jl:50,181`) — but is used only in `bench/` and `test/`. Wiring it in
   removes the per-step repack. Modest (O(N) vs O(pairs)), free once 1 is done.
4. **SIMD the hot per-atom loops.** The O-step, COM-noise, and kick/drift loops
   (`WaterBox.jl:1003-1036`) are straight `@inbounds` float64 loops. `@loop_unroll`/LoopVectorization
   on the pair-reduction and projection loops is the explicit move. Medium effort; gain bounded
   because the neighbour pair loops are memory/branch-limited (INFERRED).
5. **SETTLE the rigid molecules.** Highest ceiling, highest risk: the integrator does 3 RATTLE
   projections + a reversible principal-axis drift + exact OU noise projection per step
   (`_baoab_step!`, `WaterBox.jl:989-1039`). Replacing RATTLE with SETTLE re-validates the whole
   BAOAB ensemble. Rank last, after 1–3.

---

## 14. Recommendation

**Reject `Trium-XBT-compiler` for MD acceleration.** No component — language, back end, BTFP,
radix-3 FFT, hardware — can express or beat the existing float64 + FFTW + in-tree threaded
`StreamedLJ` pipeline on this workload. The actionable output is §13: item 2 is a one-flag change to
code already in the tree, and item 1 is the gate.

Worth saying plainly, because the audit corrected itself: this is a **competently built** project.
The hand-written parser is clean, the 39-trit integer domain is consistently enforced, the emitted
IR compiles to code that matches hand-written C, the FFT is a correct textbook implementation, and
several refusals (the multiply one, the template-emit error) are more honest than most compilers.
It is simply aimed at a different problem.

Optional curiosities, not adopted: `BTFP` is a decent reference for balanced-ternary FP arithmetic
(use `from_double`), and the radix-3 FFT is correct and tested — just not a performance target.

**Bugs worth filing upstream** (all verified here, all theirs, none blocking us):

1. **CWD library hijack, macOS:** `ffi/python/bt_ffi.py:48` bare-name `ctypes.CDLL` resolves against
   the working directory and executes attacker-controlled constructors. Highest severity here.
2. `#32 Legacy_FFI_Hardening` fails because of #1 — it is a true positive, not a platform quirk.
3. `--emit-bt-dialect` writes a silent empty module for script-style input, exit 0.
4. `--emit-bt-dialect` refuses multiply (exit 1) while `--native`/`--core-run`/`run` accept it.
5. `--core-run` hard-caps at 10 M ticks (9/iteration) and silently ignores `--execution-budget=0`,
   contradicting `--help`.
6. `bt-opt` accepts unknown ops, undefined SSA uses, and wrong arity with exit 0.
7. `--convert-all` is not idempotent (`TernaryPasses.cpp:38`).
8. `BTFP` constructors silently truncate (`BTFP<20> v(3.5)` → 3).
9. `HardwareMMUZeroMatrix::reset()` does **not** clear on darwin: `MADV_DONTNEED` returns 0 but
   leaves pages intact, and `get_byte` then returns stale data (measured: `ab cd` → `ab cd`).
   `reclaim_block` shares the same advisory-only assumption, and `allocate_slow_path`
   (`ternary_memory_arena.hpp:570-579`) hands out dirty memory with no `memset`.
10. `VulkanRuntime.cpp:29` `dlopen`s an unvalidated env-var path.
11. `CMakeLists.txt:458` gates on `BT_ENABLE_MLIR` instead of `BT_HAS_MLIR`.
12. `LC_RPATH` leaks the build directory into 8 shipped artifacts (`CMakeLists.txt:150,306`).
13. GitHub Actions tag-pinned, not SHA-pinned; `dtolnay/rust-toolchain@stable` floats.
14. `test/SDK/package.py` has zero assertions yet is registered CTest `#82`.
15. CTest `#67 SDK_Rust_FFI_ABI` uses Linux-only `nm -D` and never runs on macOS.
16. `SHA256SUMS` is 19/23 valid and self-inconsistent.
17. **`--test` is vacuous in Release builds** — assertions compiled out, yet prints
    "PASSED WITH 0 FAILURES" and exits 0 (`bt_compiler.cpp:474-495`; `CMakeLists.txt:157` vs `:193`).
18. **`--native` swallows the overflow diagnostic** — stores `@bt_error_state` in IR, never reads it
    after `fn()`, prints "Execution completed" and no reason while returning 1
    (`ternary_llvm_codegen.hpp:116,1683-1689`).
19. **Unchecked `add i64` fallthrough** for any unhandled `BinaryOp` (`ternary_llvm_codegen.hpp:1018`).
20. **`Lexer::diagnostics()` is never consumed** (`ternary_frontend_ast.hpp:370`), so lexer
    diagnostics cannot reach a user.
21. **`task_error_slot()` is a process-global, not thread-local** (`ternary_kokkos_bridge.hpp:204-226`);
    concurrent dispatches can rethrow each other's exceptions.
22. **`TritWord64::P`/`N` are public mutable** with an `assert`-only invariant
    (`ternary_core.hpp:98-99,103`), so disjointness can be broken silently in Release.
23. **Bigint silently drops capacity** in `add`/`mul`/`shift_trits_left`
    (`ternary_bigint.hpp:190-197,225,146`).
24. Three byte-identical copies of a 4,325-line generated guide are tracked in git
    (≈25,950 duplicate lines).

---

## 15. Evidence ledger

| # | Claim | Tag | Source |
|---|---|---|---|
| 1 | Standalone Release build succeeds on macOS arm64 without MLIR | VERIFIED | `cmake -B`, `--build -j8`, exit 0 |
| 2 | 82 tests; **70 / 11 / 1** in `/tmp/xbt_build_a6`, 60.5–63.3 s | VERIFIED | `ctest`, three runs |
| 3 | No `for`, no arrays/indexing, no `/` operator; exact parse errors | VERIFIED | `bt-compiler run` on probes (§4.1) |
| 4 | Integer type is 39-trit; max literal 2026277576509488133 | VERIFIED | literal probes; `Type::Int39` |
| 5 | Native path = IR text → host `clang` → `dlopen("bt_main")`; 0 LLVM / 0 upstream-MLIR symbols | VERIFIED | `nm -u` on both binaries; `otool -L`; `bt_process.hpp:158` |
| 6 | Emitted code is competitive: 2.9 ms vs 2.2 ms for a 20 M-iteration loop | VERIFIED | clang `-O3` on bt's own `.ll` |
| 7 | `--native` ≈340 ms, of which `dlopen` ≈295 ms and clang ≈43 ms | VERIFIED | instrumented `/tmp/h2` breakdown; bare-`dlopen` control |
| 8 | `--emit-bt-dialect` silent empty module on script input (123 B, 0 ops, exit 0) | VERIFIED | §4.2 Defect A |
| 9 | Multiply refused in dialect (exit 1) but accepted by `--native`/`--core-run`/`run` (exit 0) | VERIFIED | `examples/06`, all lanes |
| 10 | `--emit-bt-dialect` requires `-o`; positional output exits 1 | VERIFIED | CLI probes |
| 11 | `--core-run` budget = 10 M ticks, **9/iteration**; 1,111,109 passes, 1,111,110 fails | VERIFIED | bisection; `bt_compiler.cpp:823`; `Execute.hpp:21` |
| 12 | `--execution-budget=0` is **not** unmetered, contradicting `--help` | VERIFIED | flag probes |
| 13 | Bare `bt-compiler <file>` exits **1** with a template notice | VERIFIED | `bt_compiler.cpp:918` |
| 14 | `--convert-all` non-idempotent | CODE-READ | `TernaryPasses.cpp:38` |
| 15 | `--trace` does not exist (exit 1) | VERIFIED | CLI probe |
| 16 | MLIR is a custom text dialect; `BT_HAS_MLIR` false, `MLIR_DIR-NOTFOUND` | VERIFIED | configure output; `nm` |
| 17 | Scalar-only enforced at AST and IR level | DOCUMENTED-CLAIM (code read) | `ternary_frontend_ast.hpp:2526,2531`; `ternary_ir_lowering.hpp:18` |
| 18 | `BTFP<20>` = **29.53–30.11** effective bits; `BTFP<40>` = **61.23–61.81** | VERIFIED | ULP probe + exact binade counts (§5.1) |
| 19 | `BTFP<20>` ~200–250× `double`; `BTFP<40>` ~210–330× | VERIFIED | `/tmp/fp2.cpp`, two runs |
| 20 | `BTFP<20> v(3.5)` silently yields 3; warns only under `-Wall` | VERIFIED | compile probe |
| 21 | `BTFP<20>` round-trip max rel 8.57e-10; `BTFP<40>` exactly 0 on 200 k samples | VERIFIED | `/tmp/rt.cpp` |
| 22 | Radix-3 FFT exists and is tested | VERIFIED | `libbt.hpp:1182`; `test_unit_libbt.cpp:314-399` |
| 23 | Radix-3 1.20–1.46× slower (band 1.13–1.78×); 1.92–2.13× per-point work | VERIFIED | `probe/fft_cmp.cpp`, `fft3d.cpp`, 7 runs + 3 runs |
| 24 | FFTW already used for PME | CODE-READ | `PME.jl:205-206` |
| 25 | GPU emit lanes never produce a binary; they throw | VERIFIED | `bt_compiler.cpp:918,925`; portable bench `devices: 0` |
| 26 | IREE plugin is a text stub; 0 `iree::` symbols | VERIFIED | `iree/TernaryIREEPlugin.cpp`; `nm` |
| 27 | 1 backend registered; 5 target descriptions; `scalar multiply: no` | VERIFIED | `--list-targets` |
| 28 | 26 SDK symbols ↔ 26 prototypes, set-identical; surfaces C 32 / Py 49 / Rust 76 | VERIFIED | `sdk.h`; symbol diff |
| 29 | C, C++, Rust, and Python FFI all build and run correctly | VERIFIED | §5.3, executed |
| 30 | `bt_ffi.py` binds **27** `bt_*` symbols, exports 41 names, 9 suites pass | VERIFIED | `argtypes` count; `python3 ffi/python/bt_ffi.py` |
| 31 | **CWD library hijack on macOS** via `bt_ffi.py:48` bare leaf name | VERIFIED | `/tmp/sec` pristine reproduction; C `dlopen` control |
| 32 | `#32` fails as a true positive of #31, not a platform quirk | VERIFIED | `legacy_ffi_loader.py:74`; marker file created |
| 33 | `trium/` requires an absolute library path | VERIFIED | `ffi/python/trium/__init__.py:35-38` |
| 34 | `package.py`: 0 assertions, 18 lines, runs as CTest `#82` | VERIFIED | file read + `ctest -N` |
| 35 | Five vacuous/weak tests enumerated | CODE-READ | §6.2 |
| 36 | Two clean `/tmp` Release builds exit 0; 43/53 artifacts byte-identical; 55 compile lines | VERIFIED | W2/A4 lane |
| 37 | 0 network operations (sandboxed build; control `curl` exit 6) | VERIFIED | W2/A4 lane |
| 38 | `LC_RPATH` leaks build dir into 8 artifacts; no hardened runtime | VERIFIED | `otool -l`; `codesign` |
| 39 | CI actions tag-pinned, 0 SHA-pinned; `dtolnay/rust-toolchain@stable` floats | VERIFIED | `.github/workflows/core-foundation.yml:245` |
| 40 | Rust CI job exists and passes | VERIFIED | `core-foundation.yml:241,254` |
| 41 | `LSAN_OPTIONS: exitcode=0` suppresses leak failures | VERIFIED | workflow `:102` |
| 42 | CI is ubuntu-24.04 only, 6 jobs, no macOS | VERIFIED | workflow scan |
| 43 | `CMakeLists.txt:458` gates on wrong variable; only 1 test MLIR-gated; `:193` `-UNDEBUG` | VERIFIED | file read |
| 44 | `bt_compiler.cpp:493` `unpacked` used only in `assert()` | CODE-READ | source |
| 45 | `VulkanRuntime.cpp:29` dlopens unvalidated `BT_SDK_VULKAN_LIBRARY` | CODE-READ | source |
| 46 | `bt-opt` accepts unknown op / undefined SSA / wrong arity with exit 0; braces-only validation | VERIFIED | probe + `TernaryDialect.cpp:291-330` |
| 47 | `HardwareMMUZeroMatrix::reset()` does **not** clear on darwin; stale bytes readable after reset | VERIFIED | `/tmp/madv.c` + `/tmp/arena3.cpp`: `ab cd` → `ab cd` |
| 48 | Lean: 5 files, **64** theorems, compile, 0 `sorry`; no CMake target, no CI job | VERIFIED | scan + `CMakeLists.txt` |
| 49 | Rocq: **61** theorems; CTest `#56`, CI `native-rocq` `:148`; skips locally (77) | VERIFIED | ctest + workflow + count |
| 50 | `rocq_bridge` → `BRIDGE: PASSED`; `gen_vectors` → 34412 vectors | VERIFIED | aux lane |
| 51 | `z3core.py` inert; `benchmark_portable_gpu.py` `devices: 0` | VERIFIED | aux lane |
| 52 | 615-line SV sketch; no testbench/constraints/synthesis; `"physical_execution": "not_performed"` | DOCUMENTED-CLAIM | `hw/riscv_xbt_extension.sv`; `CHIP_HARDWARE_RESEARCH_AND_PLAN.md:88`; `physical-gate.json` |
| 53 | "104.53 GTOPS ARM64" unverifiable; archives are x86_64 Linux; zero `umulh` | VERIFIED (grep) | benchmark JSONs; repo-wide grep |
| 54 | `SHA256SUMS` 19/23 valid, 3 missing, self-mismatched | VERIFIED | `shasum -c` |
| 55 | 19/25 snapshot sources byte-identical; all 6 diffs additive in live tree | VERIFIED | `cmp` over 25 pairs |
| 56 | "66 proofs" is actually 61 Rocq theorems (28 in `BalancedTernaryFoundations.v`) | VERIFIED | proof scan |
| 57 | Root `TernaryDialect.td`/`TernaryOps.td` stale orphans; `CMakeLists.txt:68` points elsewhere | VERIFIED | configure graph |
| 58 | `include/Ternary/Detail/ternary_fp.hpp` is a 198-B forward stub | VERIFIED | size/read |
| 59 | **428** tracked files, 2702 on disk, ~205,681 tracked lines | VERIFIED | `git ls-files -z \| wc -l`; `find`; `wc -l` |
| 60 | 9 manifests, union 429 lines, **428 real covered / 0 uncovered**, 1 phantom | VERIFIED | `git ls-files --error-unmatch` |
| 61 | `lib/SDK/sdk.cpp` is a real 270-line SDK v1.2.0 | VERIFIED | file read |
| 62 | 5 root `test_*.cpp` fixtures compile and pass but are unreferenced | VERIFIED | manual build |
| 63 | `test/audit_regressions.py` → PASS exit 0; `build_docs.py` wired to nothing | VERIFIED | execution; CI/CMake scan |
| 64 | Zero repo hits for `clockless`, `durail`, `nonary` | VERIFIED (grep) | grep |
| 65 | 912.7 ms/step at 1728 waters; no per-component breakdown in repo | VERIFIED (repo data) / INFERRED (absence) | `RUN.md:181-188`; `STATUS.md:442` |
| 66 | LJ kernel threaded but off by default; `skin = 0` forces per-step repack | CODE-READ | `StreamedLJ.jl:565,833`; `WaterBox.jl:517`; `run_tip3p.jl:322` |
| 67 | Skin-cached neighbour list exists but unused in the force path | CODE-READ | `Neighbors.jl:50,181` |
| 68 | All 12 example/regression programs exit 0 under `run` | VERIFIED | loop over corpus |
| 69 | **`--test` is vacuous in Release**: assertion strings absent from binary, still prints PASSED | VERIFIED | `strings` grep = 0; `flags.make` `-DNDEBUG`; `CMakeLists.txt:157` vs `:193` |
| 70 | `--native` overflow: rc=1, **empty stderr**, prints "Execution completed" | VERIFIED | `/tmp/xbt_neg/ovf2.bt` on all three lanes |
| 71 | Unchecked `add i64` fallthrough for unhandled `BinaryOp` | CODE-READ | `ternary_llvm_codegen.hpp:1018` |
| 72 | `Lexer::diagnostics()` never consumed; dead `LexerDiagnostic::to_string` | CODE-READ | `ternary_frontend_ast.hpp:230,234,370` |
| 73 | Lexer: 51 token kinds; `/` has **no token path** (comment handling only, `:289,295,301`); `[` lexes but is never parsed | CODE-READ | full read, `ternary_frontend_ast.hpp:66-127,798-805` |
| 74 | Parser: 24 `parse_*`, 18 AST classes; precedence chain at `:2146-2273` | CODE-READ | full read |
| 75 | Dead code: `parse_comparison` alias `:2139`, `LoopNode` `:1055`, `FunctionCallNode` `:1147`, ~50 dup lines `:1795-1859` vs `:2281-2344`, `KwSparseMatrix` fallthrough `:1787-1791` | CODE-READ | full read |
| 76 | `PackedWord` holds up to 64 trits; `TritMatrix`/`SparseMatrix` are real types with matrix ops | CODE-READ | `ternary_frontend_ast.hpp:2618,826-828,2580-2586` |
| 77 | FFT is decimation-in-time with `trit_reverse`; per-stage twiddle vectors, so the "O(1) aux memory" doc claim is false | CODE-READ | `libbt.hpp:1211,1224,1248-1254` |
| 78 | `libbt.hpp` matrix `/` on integral `T` truncates; pivots via `double` lose precision >2^53 | CODE-READ | `:544,620,701,511,582,665` |
| 79 | `TernaryGemmGpu.hpp`: GLSL strings + CPU emulator only; `ternary_gemm_gpu()` never allocates device memory or launches a kernel; `gmacs` is a CPU number | CODE-READ | `:1001-1019,1016-1017` |
| 80 | `bt_backend_emitters.hpp`: 4 real emitters, 4 fixed templates that ignore input (`(void)cfg` `:341`), 0 throwing | CODE-READ | `:51-238,255-328,340-608` |
| 81 | ISA selector: `native-scalar-mul` emits the same `GROUP2/MUL` as lane-mul; "scalar multiply: no" enforced only by literal-only `is_single_trit`; rd/rs masked silently; constants >40 trits truncated | CODE-READ | `XbtInstructionSelector.hpp:473-487,492-508,357-359,117-127,828` |
| 82 | Micro-kernel SIMD is real (AVX2/AVX-512/NEON) with correct `__builtin_cpu_init`; forcing `Isa` bypasses CPUID | CODE-READ | `TernaryMicroKernel.hpp:116-131,156-243,505-516` |
| 83 | `ternary_jit_engine.hpp` is a tree-walk over AST; Kokkos threading is real but `task_error_slot()` is process-global | CODE-READ | `:312,399-1092,204-226,601-628` |
| 84 | `as_int64()` drops trits 39–63 while `make_scalar_int` encodes 64 — out-of-range literals wrap before any check | CODE-READ | `ternary_jit_engine.hpp:162,115,776` |
| 85 | Bigint silently discards carries/terms past capacity; only `to_int64` guards | CODE-READ | `ternary_bigint.hpp:190-197,225,146,820-824` |
| 86 | **12 agents actually run**; chronology 6 → 4 → 2, not two sequential waves of six | VERIFIED | artifact mtimes §1.1; 2 sessionIDs from this session |
| 87 | Deep read = **26 files / ≈34,600 lines (≈44%)** of 77,940 code lines; 26 named files inventoried-only | VERIFIED | `wc -l` per file, §1.2 |
| 88 | Three `.md` and three `.html` guide copies are byte-identical (≈25,950 duplicate lines) | VERIFIED | `shasum -a 256` |

Rows 6, 7, 18–21, 23, and the CLI probes come from **this session's** sources —
`~/.cache/biology-mm/trium-xbt-eval/probe/` (`fft_cmp.cpp`, `fft3d.cpp`, `fp_bench.cpp`, `dlprobe/`)
plus `/tmp/rt.cpp`, `/tmp/fp2.cpp`, `/tmp/h2.cpp`, `/tmp/harness.cpp`, `/tmp/dlt.cpp`, `/tmp/sec/`.
They are not repo artifacts. Everything tagged DOCUMENTED-CLAIM is a repo statement, unverified
because the required hardware or toolchain does not exist on this host.

---

## 16. What was NOT tested or audited

**Not executable on this host (no path exists):**

- GPU execution of any kind — Metal, Vulkan, CUDA. No device; the portable bench finds 0 devices and
  the emit lanes throw before reaching a driver.
- HDL synthesis, place-and-route, timing closure, any FPGA result — no toolchain, no constraints, no
  testbench in the repo.
- Rocq proof *execution* — wired and CI-configured but exit 77 locally.
- FFTW itself — no Julia/FFTW toolchain, so §5.2 compares against a scalar radix-2 proxy.
- Julia on the `foundation/` side — §13 was derived by reading the Julia sources, not by re-running
  the 912.7 ms/step trajectory.
- Absolute timings (§5.1, §5.4, §5.2) are host-specific macOS arm64 numbers. The **ratios** are the
  transferable claims, not the milliseconds.
- Ubuntu CI parity — CI runs ubuntu-24.04; everything measured here is macOS arm64. The macOS-only
  findings (`#32`, the CWD hijack, `LC_RPATH`) are precisely what CI cannot see.

**Coverage is not behavioural coverage.** All 428 tracked files appear in some manifest, but that
means a file was **opened**, not that its behaviour was characterized. The precise split:

| Level | Count | What it supports |
|---|---|---|
| Inventoried / opened | **428 files (100%)** | "was looked at"; no logic claim |
| Skimmed (read partially, or read as a doc) | ~120 doc-ish artifacts; ~56 of W2 A1's 98 at skim depth | doc/claim context only |
| **Deep, line-by-line** | **26 code files, ≈34,600 lines (≈44% of 77,940)** | every §5.5 and §7 code finding |
| Inventoried only, no deep read | **26 named ≥400-line files** (§1.2 list) | nothing beyond a shallow impression |

- Three lanes (W2 A4 build/security, W2 A5 end-to-end, W2 A6 coverage/SDK) were **concern-scoped**
  and produced no file slice of their own.
- W2 A1 read ~140 doc-ish artifacts, only 13 deeply, so roughly **56 of its 98 manifest files were
  skimmed at best**. Its claim findings are therefore weaker than its confident tone.
- Every number in §0 and §4–§6 was re-executed by the orchestrating session rather than taken from a
  worker report. Nine such checks overturned earlier claims — three of them the orchestrator's own
  (§2 #1, #2, #3) — which is why §2 keeps that table rather than presenting a clean narrative.
- The deep lanes this pass **added findings that no manifest-level audit could have produced**:
  the vacuous `--test` (§2 #9), the silent native overflow (§2 #11), the non-clearing arena
  `reset()` (§2 #10), the unchecked `add i64` fallthrough, the unconsumed lexer diagnostics, and the
  fixed-template GPU emitters (§5.5, §7). That is the concrete argument for deep reading over
  file-counting.

**Manifest caveat, restated:** `lib/SDK/SDK.cpp` in `audit/w1_a4.txt:4` is a case-insensitive-APFS
phantom; the real file is `lib/SDK/sdk.cpp`. And `audit/w2_a3.txt` (a file slice) is not the lane
"W2 A3" (claim re-verification) — do not conflate them when citing.

**Process caveat, restated:** the brief asked for 12 agents in two sequential waves of six. Twelve
agents ran, but the chronology was 6 → 4 → 2 (§1.1). This report does **not** claim the strict
protocol was followed, and no amount of re-reading changes the order in which the artifacts were
written.

---

## Appendix — file → lane map (all 428 tracked files)

Wave 1 lanes own the substantive implementation surface; Wave 2 owns docs, the upload dossier, and
validation concerns. Paths are relative to the repo root. Counts: W1 265, W2 163, total 428.
Lane labels record **ownership of the manifest entry**, not depth of read — see §1.2 for which files
were actually read line-by-line.

**W1/A1 — front end (24).** `ternary_frontend_ast.hpp`, `bt_compiler.cpp`, `bt_process.hpp`,
`sample_program.bt`, `ternary_compiler_prototype.py`, `ternary_compiler_solutions.py`,
`test_frontend_ast.cpp`, `test/Unit/test_unit_ast.cpp`,
`test/Unit/test_unit_frontend_fp_sparse.cpp`, `test/Sema/{diagnostic_error,valid_declarations}.bt`,
`test/Regression/{csa_reduce_test,fp_sparse_test,matmul_check,sample_exec,select_spaceship}.bt`,
`test/WholeProgram/matrix_pipeline.bt`, `examples/01..07_*.bt` (7 files).

**W1/A2 — IR, dialect, MLIR (39).** `ternary_ir_lowering.hpp`, `ternary_mlir_lowering.hpp`,
`ternary_linalg_lowering.mlir`, root `{TernaryDialect,TernaryOps}.td`,
`include/Ternary/{TernaryDialect,TernaryOps}.td`, `include/Ternary/TernaryPasses.h`,
`include/Ternary/TernaryDialect.h`, `include/Ternary/IR/{Core,Execute,Optimize,Program,Layout,
Tensor}.hpp`, `include/Ternary/MLIR/CoreToBtDialect.hpp`, `lib/Ternary/{TernaryDialect,
TernaryPasses}.cpp`, `tools/bt-opt/bt-opt.cpp`, `test/Dialect/bt_opt_{input_variance,linalg,
lowering,parse_errors}.mlir`, `test/Dialect/ternary_ops.mlir`,
`test/Unit/{test_unit_mlir,test_unit_bt_opt,test_unit_core_lowering,test_unit_ir_program,
test_unit_layouts,test_unit_tensor,test_bt_dialect_bridge,test_translation_validation}.cpp`,
`verification/{translation_validation,verify_layouts,verify_tensor_shapes,verify_scalar_mul,
engine}.py`, `verify_z3_core_lowering.py`, `verify_z3_phase1.py`.

**W1/A3 — back ends, targets, ISA (44).** `ternary_llvm_codegen.hpp`, `bt_backend_emitters.hpp`,
`include/Ternary/Target/{Backend,GeneratedTargets,Simulator,TargetDescription,XbtBackend,
XbtInstructionSelector}.hpp`, `include/Ternary/IR/{LLVM,GLSL}.hpp`,
`include/Ternary/Kernel/{Map,Vulkan}.hpp`, `include/Ternary/Blas/TernaryGemmGpu.hpp`,
`targets/{generic-cpu,portable-gpu,riscv64-xbt,ternary-torus}.yaml`,
`examples/minimal-target/minimal.yaml`, `schema/target.schema.json`, `specs/core-profile-v0.1.json`,
`tools/{gen_backend,gen_targets,check_generated_backends}.py`, `tools/tile_survey.cpp`,
`tools/{cuda_map,run_cuda_map}.py`, `test/CodeGen/target_{arm64,cross_compile,dwarf_lto,llvm,metal,
vulkan,x86}.test`, `test/Unit/{test_generated_targets,test_unit_kernel_isa_paths,test_xbt_selector,
test_xbt_conformance,test_unit_simulator,test_unit_kernel_contract}.cpp`,
`test/{cuda_map_test,gpu_map_test}.py`, `verification/{xbt_isa,verify_isa}.py`,
`docs/XBT_ISA_SPEC.md`.

**W1/A4 — SDK, FFI, bindings, examples (49, incl. 1 phantom).** `include/Ternary/SDK/{SDK.hpp,
sdk.h}`, `include/Ternary/ternary_c_abi.h`, `lib/SDK/{sdk.cpp,NativeRuntime.cpp,NativeRuntime.hpp,
NumericalRuntime.cpp,NumericalRuntime.hpp,VulkanRuntime.cpp,VulkanRuntime.hpp,FloatRuntime.hpp}`,
`lib/Ternary/ternary_c_abi.cpp`, `ffi/python/{bt_ffi.py,pyproject.toml,trium/__init__.py}`,
`ffi/rust/{Cargo.toml,build.rs,src/lib.rs,src/sdk.rs}`, `examples/sdk/{main.c,main.cpp,
CMakeLists.txt}`, `test/SDK/{contracts,integration,native,vulkan,package}.py`,
`test/SDK/{numerical,gemm_layout,host,math}.cpp`, `test/Unit/{test_unit_c_abi_ffi,
test_unit_math_core,test_unit_microkernel,test_unit_numeric_oracle,test_unit_phase1,test_unit_phase3,
test_unit_universal,test_unit_audit_regressions}.cpp`, `test/{runtime_abi_regression,
rust_ffi_abi_check,core_backend_regression,phase1_differential,check_rocq}.py`,
`tools/{check_sdk_parity.py,run_rust_tests.sh}`, `docs/{SDK_PARITY,SDK_1_1,
ADVANCED_NUMERICS_SDK_1_2}.md`. *Phantom:* `lib/SDK/SDK.cpp` (does not exist; APFS case-fold of
`sdk.cpp`).

**W1/A5 — numerics, hardware, proofs (66).** `ternary_fp.hpp`, `include/Ternary/Detail/ternary_fp.hpp`,
`ternary_{core,wide,carry_free,bigint,universal_codec,sparse,dense_algebra,xbt_blas,xbt_foundation,
numeric_contract,nonlinear_engine}.hpp`, `symplectic_ternary_engine.hpp`,
`hamiltonian_ternary_simulator.py`, `libbt.hpp`, `balanced_ternary_simd.cpp`,
`include/Ternary/Math/{Float,Integer,Interchange,Numerical,Rational}.hpp`,
`include/Ternary/Blas/TernaryMicroKernel.hpp`, `include/Ternary/Memory/Memory.hpp`,
`ternary_memory_arena.hpp`, `hw/riscv_xbt_extension.sv`, `hw/xbt_alu_model.hpp`, `gemm_kernel.metal`,
`gemm_shader.comp`, `*.lean` ×5, `proofs/rocq/*.v` ×4, `specs/{ternary-float-v1,
layout_conformance}.json`, `verify_z3_*.py` ×12, `verify_ternary.py`,
`verification/{verify_memory,verify_redundant}.py`, `test/Unit/{test_unit_fp_foundation,
test_unit_fp_sparse,test_unit_libbt,test_unit_dense,test_unit_bigint_symplectic,
test_unit_carry_free,test_unit_memory,test_unit_memory_model,test_unit_scalar_mul,
test_unit_xbt_hw}.cpp`, `test_multiprecision_oracle.py`.

**W1/A6 — build, tests, CI, benches (43).** `CMakeLists.txt`, `cmake/TriumXBTConfig.cmake.in`,
`.github/workflows/core-foundation.yml`, `run_all_tests.sh`, `requirements/{verification.txt,
portable_gpu_tools.json}`, `.gitignore`, `tools/{build_portable_gpu_tools.sh,install_toolchains.sh,
qualify_portable_gpu.py,verify_cordic_constants.py,heavy_cpu_benchmark.cpp,
heavy_python_benchmark.py,bench_harness.hpp,ternary_bench.cpp}`, `bench_{comparative_binary_vs_ternary,
cycle_accurate,dense_algebra,ternary_core}.cpp`, `bench_metal_gpu.mm`,
`test/Sanitizers/run_sanitizers.sh`, `test/{lit_runner,frontend_dialect_pipeline,
compiler_test_support,core_program_differential,profile_contract,require_tool,audit_regressions}.py`,
`test/benchmark_{arithmetic.cpp,wide_math.cpp,core_optimization.py,foundation.py}`,
`test/{cordic_survey.cpp,floating_foundation.py,fp_reference.py,math_reference.py,
numeric_reference.py,numerical_reference.py,gemm_layout.py}`,
`test/Unit/{test_backend_registry,test_unit_core}.cpp`,
`test/Unit/{legacy_ffi_loader.py,legacy_ffi_hijack.cpp}`.

**W2/A1 — docs and claims (98).** 29 `docs/*.md` (incl. `DEEP_AUDIT_2026_11_02.md`,
`AUDIT_FIXES_2026_11_02.md`, `CHIP_HARDWARE_RESEARCH_AND_PLAN.md`, `CORE_SEMANTICS_V0_1.md`),
53 `docs/evidence/*`, 9 `docs/chapters/*`, `docs/benchmarks/*.json` ×3, `build_docs.py`.

**W2/A2 — upload dossier vs tree (36).** `upload trium core files/00..10_*.md` (11) and
`upload trium core files/core_sources/*` (25).

**W2/A3 — auxiliary scripts and root docs (30).** Root `*.md` (architecture/contributing/licence
guides, `NEXT_GEN_FOUNDATION_COMPILER_COMPLETE_GUIDE.{md,html}`, `CODE_OF_CONDUCT.md`,
`LICENSE`), plus `docs/tile_results/*`, `docs/research/*`, `verification/__init__.py`.

**W2/A4 — build & security (concern-scoped, 0 new files).** Re-checked `CMakeLists.txt`,
`.github/workflows/`, `bt_process.hpp`, `lib/SDK/{NativeRuntime,VulkanRuntime}.cpp`,
`test/Unit/{legacy_ffi_loader.py,test_unit_xbt_hw.cpp,test_unit_frontend_fp_sparse.cpp}`.

**W2/A5 — end-to-end lanes (concern-scoped, 0 new files).** Exercised `bt-compiler` and `bt-opt`
against `examples/03,04,06`, `test/Regression/*`, and traced `bt_compiler.cpp` emit dispatch,
`Execute.hpp` budget ticks, and `TernaryDialect.cpp` validation.

**W2/A6 — test coverage and SDK (0 new files).** `test/SDK/*`, `test/Unit/*` weak-test scan,
`tools/check_sdk_parity.py`, `test/rust_ffi_abi_check.py`, `ffi/python/trium/__init__.py`.

**Known asymmetry in this appendix:** the twenty largest implementation headers
(`ternary_frontend_ast.hpp`, `ternary_fp.hpp`, `libbt.hpp`, `CMakeLists.txt`, `bt_compiler.cpp`, …)
appear under one lane each, but several were re-read by the orchestrator for verification. The map
records *ownership*, not exclusive readership.

---

*Evaluation complete. Only this file changed in Biology MM; the audit clone remains clean at
`313f669e9e605c09dc84fe2db8d200a5aee6f302` with 0 modified and 0 untracked paths.*
