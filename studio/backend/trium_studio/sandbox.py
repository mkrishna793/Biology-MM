"""AST-validated restricted Python execution sandbox for Biology-MM-Studio Code Lab."""

from __future__ import annotations

import ast
import contextlib
import io
import math
import threading
import time
import traceback
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from .engine_core import SimulationSession

ALLOWED_ROOT_MODULES: Set[str] = {
    "biology_mm",
    "numpy",
    "math",
    "trium_studio",
}

ALLOWED_TRIUM_SUBMODULES: Set[str] = {
    "trium_studio.engine_core",
    "trium_studio.presets",
}

BLOCKED_BUILTINS: Set[str] = {
    "open",
    "__import__",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "input",
    "memoryview",
}

_SAFE_BUILTIN_NAMES: Tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "bool",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
)


class SandboxSecurityError(ValueError):
    """Raised when user code violates AST or import sandbox rules."""


def _is_allowed_module(module_name: str) -> bool:
    if not module_name:
        return False
    root = module_name.split(".")[0]
    if root not in ALLOWED_ROOT_MODULES:
        return False
    if root == "trium_studio":
        return module_name == "trium_studio" or module_name in ALLOWED_TRIUM_SUBMODULES
    return True


def validate_user_ast(code: str) -> ast.Module:
    """Parse `code` and reject disallowed imports, builtins, or dunder attribute access."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxSecurityError(f"SyntaxError at line {exc.lineno}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_module(alias.name):
                    raise SandboxSecurityError(f"Disallowed import: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level != 0 or not _is_allowed_module(mod):
                raise SandboxSecurityError(f"Disallowed import from: '{mod}'")
            if mod == "trium_studio":
                for alias in node.names:
                    if alias.name not in {"engine_core", "SimulationSession", "PHYSICS_CHANNEL_NAMES"}:
                        raise SandboxSecurityError(f"Disallowed import '{alias.name}' from 'trium_studio'")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SandboxSecurityError(f"Disallowed dunder attribute access: '{node.attr}'")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_BUILTINS:
                raise SandboxSecurityError(f"Disallowed builtin: '{node.id}'")
            if node.id.startswith("__") and node.id != "__name__":
                raise SandboxSecurityError(f"Disallowed dunder identifier: '{node.id}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                raise SandboxSecurityError(f"Disallowed builtin call: '{node.func.id}'")

    return tree


def _safe_import(
    name: str,
    globals: Optional[Dict[str, Any]] = None,
    locals: Optional[Dict[str, Any]] = None,
    fromlist: Tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    if level != 0 or not _is_allowed_module(name):
        raise SandboxSecurityError(f"Disallowed runtime import: '{name}'")
    return __import__(name, globals, locals, fromlist, level)


def _build_restricted_builtins() -> Dict[str, Any]:
    import builtins as _py_builtins

    safe: Dict[str, Any] = {}
    for bname in _SAFE_BUILTIN_NAMES:
        if hasattr(_py_builtins, bname):
            safe[bname] = getattr(_py_builtins, bname)
    safe["__import__"] = _safe_import
    return safe


def execute_user_code(
    code: str,
    timeout_s: float = 10.0,
    active_structure: Optional[Any] = None,
    active_structure_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[SimulationSession]]:
    """
    Validate and execute user Python code in a restricted sandbox with timeout.

    Returns `(CodeRunResult_dict, optional_SimulationSession)`.
    """
    t0 = time.perf_counter()
    if not isinstance(code, str) or not code.strip():
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "Code snippet is empty.",
            "elapsed_ms": 0.0,
        }, None

    try:
        tree = validate_user_ast(code)
        compiled = compile(tree, "<studio_code_lab>", "exec")
    except SandboxSecurityError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "elapsed_ms": round(elapsed_ms, 3),
        }, None

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exec_globals: Dict[str, Any] = {
        "__builtins__": _build_restricted_builtins(),
        "__name__": "__studio_lab__",
        "np": np,
        "math": math,
        "active_structure": active_structure,
        "active_structure_metadata": active_structure_metadata,
    }
    exec_locals: Dict[str, Any] = {}
    err_holder: Dict[str, Optional[str]] = {"error": None}

    def _runner() -> None:
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compiled, exec_globals, exec_locals)  # noqa: S102 - AST & builtin restricted sandbox
        except Exception as exc:
            err_holder["error"] = f"{type(exc).__name__}: {exc}"
            stderr_buf.write(traceback.format_exc())

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=max(0.05, float(timeout_s)))

    elapsed_ms = (time.perf_counter() - t0) * 1e3
    if worker.is_alive():
        return {
            "ok": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": f"Execution timed out after {float(timeout_s):.1f}s",
            "elapsed_ms": round(elapsed_ms, 3),
        }, None

    if err_holder["error"] is not None:
        return {
            "ok": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": err_holder["error"],
            "elapsed_ms": round(elapsed_ms, 3),
        }, None

    found_session: Optional[SimulationSession] = None
    if isinstance(exec_locals.get("session"), SimulationSession):
        found_session = exec_locals["session"]
    elif isinstance(exec_globals.get("session"), SimulationSession):
        found_session = exec_globals["session"]
    else:
        for val in list(exec_locals.values())[::-1]:
            if isinstance(val, SimulationSession):
                found_session = val
                break

    result: Dict[str, Any] = {
        "ok": True,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "elapsed_ms": round(elapsed_ms, 3),
    }
    if found_session is not None:
        result["metrics"] = found_session.get_metrics()
    return result, found_session
