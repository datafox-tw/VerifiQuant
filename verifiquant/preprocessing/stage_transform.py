"""stage_transform.py — Atomic, verifiable transforms for I-class repair.

Two patch types
---------------
**result_postprocess**  (V1, already implemented)
    new_result = eval(result_expr, {"result": original_result, **bound_inputs})
    Verified by: AST guard on result_expr + SymPy static audit.

**code_patch**  (V2, new)
    Replace `target_pattern` in FIC code with `replacement`.
    Verified by three independent layers:
    1. Occurrence guard — target_pattern appears exactly once in the code.
    2. AST diff guard   — # changed nodes ≤ max_changed_nodes.
    3. Cross-verify     — patched_code(inputs) ≈ eval(cross_verify_expr,
                           {result: original_code(inputs), **inputs})
       This links the code change to its declared algebraic meaning.
       If both paths agree numerically, the patch does exactly what it claims.

SymPy is optional and used only for the static algebraic audit.
"""
from __future__ import annotations

import ast
import math
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import sympy
except ImportError:
    sympy = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared safe builtins
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Set[str] = {
    "abs", "round", "min", "max", "sum", "len", "pow",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransformSpec:
    """result_postprocess: adjust the final result arithmetically."""
    patch_type: str = "result_postprocess"
    result_expr: str = ""
    max_expr_nodes: int = 99
    sympy_invariant: str = ""
    affected_inputs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformSpec":
        return cls(
            patch_type=str(d.get("patch_type", "result_postprocess")),
            result_expr=str(d.get("result_expr", "")),
            max_expr_nodes=int(d.get("max_expr_nodes", 99)),
            sympy_invariant=str(d.get("sympy_invariant", "")),
            affected_inputs=list(d.get("affected_inputs") or []),
        )


@dataclass
class CodePatchSpec:
    """code_patch: replace one pattern in FIC code with a declared replacement.

    Fields
    ------
    target_pattern:
        Exact substring to find in execution.code. Must appear exactly once.
        Keep it as narrow as possible — one keyword argument, one constant.
        Example: "start=1"
    replacement:
        Replacement string. Example: "start=0"
    max_changed_nodes:
        Expected number of AST nodes that differ between old and patched code.
        Acts as a blast radius bound. Example: 1 (just the Constant 1→0).
    cross_verify_result_expr:
        A result_postprocess-style expression. After patching, running the
        patched code must agree numerically with:
            eval(cross_verify_result_expr, {"result": original_result, **inputs})
        This is the mathematical proof that the patch does what it claims.
        Example: "(result + initial_investment) * (1 + discount_rate) - initial_investment"
    sympy_invariant:
        Algebraic statement for documentation / static audit.
    affected_inputs:
        Input names that appear in cross_verify_result_expr.
    """
    patch_type: str = "code_patch"
    target_pattern: str = ""
    replacement: str = ""
    max_changed_nodes: int = 4
    cross_verify_result_expr: str = ""
    cross_verify_max_nodes: int = 30
    sympy_invariant: str = ""
    affected_inputs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CodePatchSpec":
        return cls(
            patch_type=str(d.get("patch_type", "code_patch")),
            target_pattern=str(d.get("target_pattern", "")),
            replacement=str(d.get("replacement", "")),
            max_changed_nodes=int(d.get("max_changed_nodes", 4)),
            cross_verify_result_expr=str(d.get("cross_verify_result_expr", "")),
            cross_verify_max_nodes=int(d.get("cross_verify_max_nodes", 30)),
            sympy_invariant=str(d.get("sympy_invariant", "")),
            affected_inputs=list(d.get("affected_inputs") or []),
        )


def patch_spec_from_dict(d: Dict[str, Any]) -> Union[TransformSpec, CodePatchSpec]:
    """Factory: route to the right spec class based on patch_type."""
    pt = str(d.get("patch_type", "result_postprocess"))
    if pt == "code_patch":
        return CodePatchSpec.from_dict(d)
    return TransformSpec.from_dict(d)


# ---------------------------------------------------------------------------
# Verify result types
# ---------------------------------------------------------------------------

@dataclass
class TransformVerifyResult:
    passed: bool
    ast_node_count: int
    max_allowed_nodes: int
    forbidden_names: List[str]
    numerical_ratio_ok: Optional[bool]
    numerical_samples: int
    sympy_check: Optional[str]
    error: Optional[str] = None

    def summary(self) -> str:
        parts = [
            f"ast_nodes={self.ast_node_count}/{self.max_allowed_nodes}",
            f"forbidden={self.forbidden_names or 'none'}",
            f"numerical={'ok' if self.numerical_ratio_ok else 'FAIL' if self.numerical_ratio_ok is False else 'skip'}",
            f"sympy={self.sympy_check or 'skip'}",
        ]
        return f"[result_postprocess {'PASS' if self.passed else 'FAIL'}] " + " | ".join(parts)


@dataclass
class CodePatchVerifyResult:
    passed: bool
    occurrence_count: int           # how many times target_pattern was found
    ast_changed_nodes: int          # actual AST diff
    max_changed_nodes: int          # declared bound
    cross_verify_ok: Optional[bool] # patched_code ≈ cross_verify_expr?
    cross_verify_max_delta: Optional[float]  # max abs deviation across samples
    numerical_samples: int
    sympy_check: Optional[str]
    error: Optional[str] = None

    def summary(self) -> str:
        parts = [
            f"occurrences={self.occurrence_count}",
            f"ast_diff={self.ast_changed_nodes}/{self.max_changed_nodes}",
            f"cross_verify={'ok' if self.cross_verify_ok else 'FAIL' if self.cross_verify_ok is False else 'skip'}",
            f"max_delta={self.cross_verify_max_delta:.2e}" if self.cross_verify_max_delta is not None else "max_delta=n/a",
            f"sympy={self.sympy_check or 'skip'}",
        ]
        return f"[code_patch {'PASS' if self.passed else 'FAIL'}] " + " | ".join(parts)


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------

def count_ast_nodes(expr: str) -> int:
    try:
        tree = ast.parse(expr, mode="eval")
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return -1


def _shallow_node_key(node: ast.AST) -> str:
    """Produce a per-node key using only the node's OWN primitive fields.

    Child AST nodes and lists are excluded so that changing one Constant
    does NOT ripple through all parent node keys.

    Example: keyword(arg='start', value=Constant(1))
      → "keyword|arg=start"   (value is an AST child, excluded)
    Example: Constant(value=1)
      → "Constant|value=1"
    """
    parts = [type(node).__name__]
    for field_name, field_value in ast.iter_fields(node):
        if isinstance(field_value, ast.AST):
            continue
        if isinstance(field_value, list):
            # Check if it's a list of AST nodes; if so, skip.
            if any(isinstance(v, ast.AST) for v in field_value):
                continue
            parts.append(f"{field_name}={field_value!r}")
        else:
            parts.append(f"{field_name}={field_value!r}")
    return "|".join(parts)


def ast_diff_count(old_code: str, new_code: str) -> int:
    """Count changed AST nodes using shallow (per-node) key comparison.

    Only each node's own primitive attributes are hashed — child subtrees
    are not included.  This means changing `start=1` to `start=0` counts
    as 1 changed node (Constant), not 6 (entire parent chain).

    Returns max(removed, added) as the change count.
    """
    try:
        old_nodes = Counter(_shallow_node_key(n) for n in ast.walk(ast.parse(old_code)))
        new_nodes = Counter(_shallow_node_key(n) for n in ast.walk(ast.parse(new_code)))
        removed = sum((old_nodes - new_nodes).values())
        added = sum((new_nodes - old_nodes).values())
        return max(removed, added)
    except SyntaxError:
        return -1


def _collect_names(tree: ast.AST) -> Set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _collect_calls(tree: ast.AST) -> Set[str]:
    calls: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def audit_expr_safety(
    result_expr: str,
    allowed_names: Set[str],
) -> Tuple[int, List[str]]:
    """Return (node_count, forbidden_names_list)."""
    try:
        tree = ast.parse(result_expr, mode="eval")
    except SyntaxError as exc:
        return -1, [f"SyntaxError: {exc}"]

    node_count = sum(1 for _ in ast.walk(tree))
    used_names = _collect_names(tree)
    used_calls = _collect_calls(tree)
    forbidden = sorted((used_names | used_calls) - allowed_names - _SAFE_BUILTINS)
    return node_count, forbidden


def audit_patched_code_safety(patched_code: str, original_code: str) -> List[str]:
    """Return list of safety violations introduced by the patch.

    Checks: no new `import` statements, no new `__dunder__` name references,
    no new `exec` / `eval` / `open` / `os` calls.
    """
    _FORBIDDEN_IN_PATCH = {"__import__", "exec", "eval", "open", "os",
                            "subprocess", "sys", "shutil", "importlib"}
    violations: List[str] = []
    try:
        orig_calls = _collect_calls(ast.parse(original_code))
        new_calls = _collect_calls(ast.parse(patched_code))
        new_dangerous = (new_calls - orig_calls) & _FORBIDDEN_IN_PATCH
        if new_dangerous:
            violations.append(f"new dangerous calls: {sorted(new_dangerous)}")

        # Check for new import nodes
        orig_imports = {ast.dump(n) for n in ast.walk(ast.parse(original_code))
                        if isinstance(n, (ast.Import, ast.ImportFrom))}
        new_imports = {ast.dump(n) for n in ast.walk(ast.parse(patched_code))
                       if isinstance(n, (ast.Import, ast.ImportFrom))}
        added_imports = new_imports - orig_imports
        if added_imports:
            violations.append(f"new import statements introduced")
    except SyntaxError as exc:
        violations.append(f"SyntaxError in patched code: {exc}")
    return violations


# ---------------------------------------------------------------------------
# Execute FIC code
# ---------------------------------------------------------------------------

def _run_fic_code(code: str, entrypoint: str, inputs: Dict[str, Any]) -> Any:
    """Execute FIC code. Supports both call styles:
    - fn(**inputs)   for functions like def compute_npv(cash_flow, rate, ...):
    - fn(inputs)     for functions like def compute(inputs):
    """
    ns: Dict[str, Any] = {}
    exec(textwrap.dedent(code), ns)  # noqa: S102
    fn = ns.get(entrypoint)
    if fn is None:
        raise RuntimeError(f"Entrypoint '{entrypoint}' not found in FIC code.")
    try:
        return fn(**inputs)
    except TypeError:
        return fn(inputs)


# ---------------------------------------------------------------------------
# Apply transforms
# ---------------------------------------------------------------------------

def apply_result_postprocess(
    original_result: Any,
    spec: TransformSpec,
    bound_inputs: Dict[str, Any],
) -> Any:
    """Apply a result_postprocess spec to original_result."""
    if spec.patch_type != "result_postprocess":
        raise ValueError(f"Expected result_postprocess, got {spec.patch_type!r}")
    namespace: Dict[str, Any] = {"result": original_result}
    namespace.update(bound_inputs)
    builtins_obj = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins = {k: builtins_obj[k] for k in _SAFE_BUILTINS if k in builtins_obj}
    namespace["__builtins__"] = safe_builtins
    return eval(spec.result_expr, namespace)  # noqa: S307


# Keep the old name as an alias for backward compatibility.
apply_transform = apply_result_postprocess


def apply_code_patch(code: str, spec: CodePatchSpec) -> str:
    """Apply a code_patch spec: replace target_pattern with replacement.

    Raises ValueError if target_pattern is not found or appears > 1 time.
    """
    if spec.patch_type != "code_patch":
        raise ValueError(f"Expected code_patch, got {spec.patch_type!r}")
    count = code.count(spec.target_pattern)
    if count == 0:
        raise ValueError(
            f"target_pattern {spec.target_pattern!r} not found in code.\n"
            f"Code preview:\n{code[:300]}"
        )
    if count > 1:
        raise ValueError(
            f"target_pattern {spec.target_pattern!r} appears {count} times; "
            "must appear exactly once for a safe patch."
        )
    return code.replace(spec.target_pattern, spec.replacement, 1)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_transform(
    spec: TransformSpec,
    fic_core: Dict[str, Any],
    sample_inputs: List[Dict[str, Any]],
    *,
    ratio_tolerance: float = 1e-6,
) -> TransformVerifyResult:
    """Verify a result_postprocess TransformSpec."""
    fic_input_names: Set[str] = {inp["name"] for inp in fic_core.get("inputs", [])}
    code = fic_core.get("execution", {}).get("code", "")
    entrypoint = fic_core.get("execution", {}).get("entrypoint", "")

    allowed_names = {"result"} | fic_input_names
    node_count, forbidden = audit_expr_safety(spec.result_expr, allowed_names)

    ast_ok = node_count >= 0 and node_count <= spec.max_expr_nodes and not forbidden
    if not ast_ok:
        return TransformVerifyResult(
            passed=False, ast_node_count=node_count,
            max_allowed_nodes=spec.max_expr_nodes,
            forbidden_names=forbidden, numerical_ratio_ok=None,
            numerical_samples=0, sympy_check=None,
            error=f"AST guard failed: nodes={node_count}, forbidden={forbidden}",
        )

    numerical_ratio_ok: Optional[bool] = None
    numerical_samples = 0
    if sample_inputs and code and entrypoint:
        try:
            for inp in sample_inputs:
                original = _run_fic_code(code, entrypoint, inp)
                t1 = apply_result_postprocess(original, spec, inp)
                t2 = apply_result_postprocess(original, spec, inp)
                if t1 != t2:
                    raise ValueError(f"Non-deterministic: {t1} != {t2}")
                if not math.isfinite(float(t1)):
                    raise ValueError(f"Non-finite output: {t1}")
                numerical_samples += 1
            numerical_ratio_ok = True
        except Exception as exc:
            return TransformVerifyResult(
                passed=False, ast_node_count=node_count,
                max_allowed_nodes=spec.max_expr_nodes,
                forbidden_names=forbidden, numerical_ratio_ok=False,
                numerical_samples=numerical_samples, sympy_check=None,
                error=f"Numerical check error: {exc}",
            )

    sympy_check: Optional[str] = None
    if sympy is not None and spec.sympy_invariant:
        sympy_check = _sympy_audit_postprocess(spec)

    passed = ast_ok and (numerical_ratio_ok is not False)
    return TransformVerifyResult(
        passed=passed, ast_node_count=node_count,
        max_allowed_nodes=spec.max_expr_nodes,
        forbidden_names=forbidden, numerical_ratio_ok=numerical_ratio_ok,
        numerical_samples=numerical_samples, sympy_check=sympy_check,
    )


def verify_code_patch(
    spec: CodePatchSpec,
    fic_core: Dict[str, Any],
    sample_inputs: List[Dict[str, Any]],
    *,
    cross_verify_tolerance: float = 1e-4,
) -> CodePatchVerifyResult:
    """Verify a code_patch spec against FIC core.

    Three-layer verification:
    1. Occurrence guard: target_pattern appears exactly once.
    2. AST diff guard: changed nodes ≤ max_changed_nodes.
       Also checks no new dangerous calls / imports.
    3. Cross-verify: patched_code output ≈ cross_verify_result_expr output.
       This is the mathematical proof layer.
    """
    code = fic_core.get("execution", {}).get("code", "")
    entrypoint = fic_core.get("execution", {}).get("entrypoint", "")
    fic_input_names: Set[str] = {inp["name"] for inp in fic_core.get("inputs", [])}

    # --- 1. Apply patch + occurrence guard ---
    occurrence_count = code.count(spec.target_pattern)
    if occurrence_count != 1:
        return CodePatchVerifyResult(
            passed=False,
            occurrence_count=occurrence_count,
            ast_changed_nodes=-1,
            max_changed_nodes=spec.max_changed_nodes,
            cross_verify_ok=None,
            cross_verify_max_delta=None,
            numerical_samples=0,
            sympy_check=None,
            error=(
                f"target_pattern {spec.target_pattern!r} must appear exactly once; "
                f"found {occurrence_count}."
            ),
        )

    try:
        patched_code = apply_code_patch(code, spec)
    except ValueError as exc:
        return CodePatchVerifyResult(
            passed=False, occurrence_count=occurrence_count,
            ast_changed_nodes=-1, max_changed_nodes=spec.max_changed_nodes,
            cross_verify_ok=None, cross_verify_max_delta=None,
            numerical_samples=0, sympy_check=None, error=str(exc),
        )

    # --- 2. AST diff guard ---
    changed_nodes = ast_diff_count(code, patched_code)
    safety_violations = audit_patched_code_safety(patched_code, code)

    ast_ok = (
        changed_nodes >= 0
        and changed_nodes <= spec.max_changed_nodes
        and not safety_violations
    )
    if not ast_ok:
        return CodePatchVerifyResult(
            passed=False, occurrence_count=occurrence_count,
            ast_changed_nodes=changed_nodes,
            max_changed_nodes=spec.max_changed_nodes,
            cross_verify_ok=None, cross_verify_max_delta=None,
            numerical_samples=0, sympy_check=None,
            error=(
                f"AST diff guard failed: changed={changed_nodes} (max={spec.max_changed_nodes}), "
                f"safety violations: {safety_violations}"
            ),
        )

    # --- 3. Cross-verify numerically ---
    cross_verify_ok: Optional[bool] = None
    cross_verify_max_delta: Optional[float] = None
    numerical_samples = 0

    if sample_inputs and code and entrypoint and spec.cross_verify_result_expr:
        # Audit the cross_verify_result_expr for safety too.
        allowed_names = {"result"} | fic_input_names
        cv_node_count, cv_forbidden = audit_expr_safety(
            spec.cross_verify_result_expr, allowed_names
        )
        if cv_forbidden or cv_node_count > spec.cross_verify_max_nodes:
            return CodePatchVerifyResult(
                passed=False, occurrence_count=occurrence_count,
                ast_changed_nodes=changed_nodes,
                max_changed_nodes=spec.max_changed_nodes,
                cross_verify_ok=False, cross_verify_max_delta=None,
                numerical_samples=0, sympy_check=None,
                error=(
                    f"cross_verify_result_expr safety check failed: "
                    f"nodes={cv_node_count}, forbidden={cv_forbidden}"
                ),
            )

        builtins_obj = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        safe_builtins = {k: builtins_obj[k] for k in _SAFE_BUILTINS if k in builtins_obj}

        try:
            deltas: List[float] = []
            for inp in sample_inputs:
                # Path A: patched code
                patched_result = _run_fic_code(patched_code, entrypoint, inp)
                # Path B: original code + cross_verify_result_expr
                original_result = _run_fic_code(code, entrypoint, inp)
                ns: Dict[str, Any] = {"result": original_result, "__builtins__": safe_builtins}
                ns.update(inp)
                expected = eval(spec.cross_verify_result_expr, ns)  # noqa: S307
                # Check they agree.
                delta = abs(float(patched_result) - float(expected))
                deltas.append(delta)
                numerical_samples += 1

            cross_verify_max_delta = max(deltas) if deltas else 0.0
            cross_verify_ok = cross_verify_max_delta <= cross_verify_tolerance

        except Exception as exc:
            return CodePatchVerifyResult(
                passed=False, occurrence_count=occurrence_count,
                ast_changed_nodes=changed_nodes,
                max_changed_nodes=spec.max_changed_nodes,
                cross_verify_ok=False,
                cross_verify_max_delta=None,
                numerical_samples=numerical_samples, sympy_check=None,
                error=f"Cross-verify error: {exc}",
            )

    # --- 4. Optional SymPy audit on cross_verify invariant ---
    sympy_check: Optional[str] = None
    if sympy is not None and spec.sympy_invariant:
        sympy_check = _sympy_audit_code_patch(spec, fic_input_names)

    passed = ast_ok and (cross_verify_ok is not False)
    return CodePatchVerifyResult(
        passed=passed, occurrence_count=occurrence_count,
        ast_changed_nodes=changed_nodes,
        max_changed_nodes=spec.max_changed_nodes,
        cross_verify_ok=cross_verify_ok,
        cross_verify_max_delta=cross_verify_max_delta,
        numerical_samples=numerical_samples,
        sympy_check=sympy_check,
    )


# ---------------------------------------------------------------------------
# SymPy audit helpers
# ---------------------------------------------------------------------------

def _sympy_audit_postprocess(spec: TransformSpec) -> str:
    if sympy is None:
        return "skipped_no_sympy"
    try:
        invariant = spec.sympy_invariant
        result_old, result_new = sympy.symbols("result_old result_new", positive=True)
        local_syms: Dict[str, Any] = {
            name: sympy.Symbol(name, positive=True)
            for name in spec.affected_inputs
        }
        local_syms.update({"result_old": result_old, "result_new": result_new})
        result_expr_sym = sympy.sympify(
            spec.result_expr.replace("result", "result_old"), locals=local_syms
        )
        if "==" in invariant:
            lhs_str, rhs_str = invariant.split("==", 1)
            lhs = sympy.sympify(lhs_str.strip(), locals=local_syms)
            rhs = sympy.sympify(rhs_str.strip(), locals=local_syms)
            diff = sympy.simplify(lhs.subs(result_new, result_expr_sym) - rhs)
            return "verified" if diff == 0 else f"counterexample: {diff}"
        return "skipped_no_equality"
    except Exception as exc:
        return f"skipped_error: {exc}"


def _sympy_audit_code_patch(spec: CodePatchSpec, fic_input_names: Set[str]) -> str:
    if sympy is None:
        return "skipped_no_sympy"
    try:
        local_syms: Dict[str, Any] = {
            name: sympy.Symbol(name, positive=True)
            for name in spec.affected_inputs
        }
        result_old, result_new = sympy.symbols("result_old result_new", positive=True)
        local_syms.update({"result_old": result_old, "result_new": result_new})
        cv_expr_sym = sympy.sympify(
            spec.cross_verify_result_expr.replace("result", "result_old"),
            locals=local_syms,
        )
        invariant = spec.sympy_invariant
        if "==" in invariant:
            lhs_str, rhs_str = invariant.split("==", 1)
            lhs = sympy.sympify(lhs_str.strip(), locals=local_syms)
            rhs = sympy.sympify(rhs_str.strip(), locals=local_syms)
            diff = sympy.simplify(lhs.subs(result_new, cv_expr_sym) - rhs)
            return "verified" if diff == 0 else f"counterexample: {diff}"
        return "skipped_no_equality"
    except Exception as exc:
        return f"skipped_error: {exc}"


# ---------------------------------------------------------------------------
# Convenience: extract spec from repair rule
# ---------------------------------------------------------------------------

def get_transform_spec_for_choice(
    repair_rule: Dict[str, Any],
    chosen_value: str,
) -> Optional[Union[TransformSpec, CodePatchSpec]]:
    """Return the spec (result_postprocess or code_patch) for the chosen option.
    Returns None if the choice is the default (no transform needed).
    """
    transform_map = repair_rule.get("repair_action", {}).get("transform_map", {})
    raw_spec = transform_map.get(chosen_value)
    if raw_spec is None:
        return None
    return patch_spec_from_dict(raw_spec)
