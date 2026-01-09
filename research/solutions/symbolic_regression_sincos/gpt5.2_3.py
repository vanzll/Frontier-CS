import numpy as np
from itertools import combinations
from fractions import Fraction

class _Term:
    __slots__ = ("expr", "values", "unary_ops", "binary_ops")

    def __init__(self, expr: str, values: np.ndarray, unary_ops: int, binary_ops: int):
        self.expr = expr
        self.values = values
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(d * d))


def _snap_float(a: float, abs_tol: float = 1e-10, rel_tol: float = 1e-8):
    if not np.isfinite(a):
        return a
    if abs(a) <= abs_tol:
        return 0.0

    # Prefer integers and simple rationals if extremely close.
    nearest_int = float(np.round(a))
    if abs(a - nearest_int) <= max(abs_tol, rel_tol * max(1.0, abs(a))):
        return nearest_int

    fr = Fraction(a).limit_denominator(12)
    rf = fr.numerator / fr.denominator
    if abs(a - rf) <= max(abs_tol, rel_tol * max(1.0, abs(a))):
        return rf

    # Also try common constants
    common = [0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 1.5, -1.5, 0.25, -0.25]
    for c in common:
        if abs(a - c) <= max(abs_tol, rel_tol * max(1.0, abs(a))):
            return c

    return a


def _fmt_float(a: float) -> str:
    if a == 0.0:
        return "0"
    if float(a).is_integer() and abs(a) <= 1e12:
        return str(int(a))
    s = "{:.12g}".format(float(a))
    if s == "-0":
        s = "0"
    return s


def _build_library(x1: np.ndarray, x2: np.ndarray):
    n = x1.shape[0]
    terms = []

    def add(expr, values, unary_ops, binary_ops):
        values = np.asarray(values, dtype=np.float64).reshape(n)
        terms.append(_Term(expr, values, unary_ops, binary_ops))

    # Linear terms (sometimes help if dataset includes offsets/slopes)
    add("x1", x1, 0, 0)
    add("x2", x2, 0, 0)

    ks = (1, 2, 3)
    for k in ks:
        if k == 1:
            add("sin(x1)", np.sin(x1), 1, 0)
            add("cos(x1)", np.cos(x1), 1, 0)
            add("sin(x2)", np.sin(x2), 1, 0)
            add("cos(x2)", np.cos(x2), 1, 0)
        else:
            kk = float(k)
            add(f"sin({k}*x1)", np.sin(kk * x1), 1, 1)
            add(f"cos({k}*x1)", np.cos(kk * x1), 1, 1)
            add(f"sin({k}*x2)", np.sin(kk * x2), 1, 1)
            add(f"cos({k}*x2)", np.cos(kk * x2), 1, 1)

    s = x1 + x2
    d = x1 - x2
    for k in ks:
        if k == 1:
            add("sin(x1+x2)", np.sin(s), 1, 1)
            add("cos(x1+x2)", np.cos(s), 1, 1)
            add("sin(x1-x2)", np.sin(d), 1, 1)
            add("cos(x1-x2)", np.cos(d), 1, 1)
        else:
            kk = float(k)
            add(f"sin({k}*(x1+x2))", np.sin(kk * s), 1, 2)  # + inside and * outside
            add(f"cos({k}*(x1+x2))", np.cos(kk * s), 1, 2)
            add(f"sin({k}*(x1-x2))", np.sin(kk * d), 1, 2)  # - inside and * outside
            add(f"cos({k}*(x1-x2))", np.cos(kk * d), 1, 2)

    # Products (k=1 only, to keep library modest)
    sinx1 = np.sin(x1)
    cosx1 = np.cos(x1)
    sinx2 = np.sin(x2)
    cosx2 = np.cos(x2)
    add("sin(x1)*cos(x2)", sinx1 * cosx2, 2, 1)
    add("cos(x1)*sin(x2)", cosx1 * sinx2, 2, 1)
    add("sin(x1)*sin(x2)", sinx1 * sinx2, 2, 1)
    add("cos(x1)*cos(x2)", cosx1 * cosx2, 2, 1)

    return terms


def _complexity_from_parts(constant, parts):
    # parts: list of (coef, term_unary, term_binary, coef_mul_binary(0/1), expr_str)
    # Count binary ops:
    # - term internal binary ops
    # - multiplications by coefficient (if coef not +/-1, and not 0)
    # - additions between nonzero components (including constant if present)
    # Unary ops: term unary ops
    bin_ops = 0
    unary_ops = 0
    component_count = 0

    if constant is not None and constant != 0.0:
        component_count += 1

    for coef, tu, tb, cmul, _ in parts:
        if coef == 0.0:
            continue
        component_count += 1
        unary_ops += tu
        bin_ops += tb
        bin_ops += cmul

    if component_count >= 2:
        bin_ops += (component_count - 1)

    return 2 * bin_ops + unary_ops


def _build_expression(constant, used_terms):
    # used_terms: list of (coef, term_expr)
    parts = []

    c = constant if constant is not None else 0.0
    if c != 0.0:
        parts.append(_fmt_float(c))

    for coef, texpr in used_terms:
        if coef == 0.0:
            continue
        if coef == 1.0:
            parts.append(texpr)
        elif coef == -1.0:
            parts.append(f"-{texpr}")
        else:
            coef_str = _fmt_float(abs(coef))
            if coef > 0:
                parts.append(f"{coef_str}*{texpr}")
            else:
                parts.append(f"-{coef_str}*{texpr}")

    if not parts:
        return "0"

    expr = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            expr += " - " + p[1:]
        else:
            expr += " + " + p
    return expr


class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 4))
        self.coef_prune = float(kwargs.get("coef_prune", 1e-10))
        self.snap = bool(kwargs.get("snap", True))
        self.random_state = int(kwargs.get("random_state", 0))

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = y.shape[0]

        if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] != n:
            raise ValueError("X must have shape (n, 2) matching y shape (n,)")

        # Remove non-finite rows (shouldn't happen, but safe)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if not np.all(mask):
            X = X[mask]
            y = y[mask]
            n = y.shape[0]
            if n == 0:
                return {"expression": "0", "predictions": [], "details": {}}

        x1 = X[:, 0]
        x2 = X[:, 1]

        terms = _build_library(x1, x2)
        m = len(terms)
        lib = np.column_stack([t.values for t in terms])  # (n, m)

        ones = np.ones((n, 1), dtype=np.float64)

        # Fast baseline candidates (common in SinCos benchmarks)
        quick_exprs = [
            (0.0, [(1.0, "sin(x1)"), (1.0, "cos(x2)")]),
            (0.0, [(1.0, "sin(x1)"), (1.0, "sin(x2)")]),
            (0.0, [(1.0, "cos(x1)"), (1.0, "cos(x2)")]),
            (0.0, [(1.0, "sin(x1+x2)")]),
            (0.0, [(1.0, "sin(x1-x2)")]),
            (0.0, [(1.0, "sin(x1)*cos(x2)")]),
            (0.0, [(1.0, "cos(x1)*sin(x2)")]),
            (0.0, [(1.0, "cos(x1)*cos(x2)")]),
            (0.0, [(1.0, "sin(x1)*sin(x2)")]),
        ]

        # Map expr string to values for quick eval
        val_map = {
            "x1": x1,
            "x2": x2,
            "sin(x1)": np.sin(x1),
            "cos(x1)": np.cos(x1),
            "sin(x2)": np.sin(x2),
            "cos(x2)": np.cos(x2),
            "sin(x1+x2)": np.sin(x1 + x2),
            "cos(x1+x2)": np.cos(x1 + x2),
            "sin(x1-x2)": np.sin(x1 - x2),
            "cos(x1-x2)": np.cos(x1 - x2),
            "sin(x1)*cos(x2)": np.sin(x1) * np.cos(x2),
            "cos(x1)*sin(x2)": np.cos(x1) * np.sin(x2),
            "sin(x1)*sin(x2)": np.sin(x1) * np.sin(x2),
            "cos(x1)*cos(x2)": np.cos(x1) * np.cos(x2),
        }

        best = {
            "mse": float("inf"),
            "complexity": float("inf"),
            "constant": 0.0,
            "used": [],
            "pred": None,
            "expr": "0",
        }

        def consider_solution(constant, used_terms, pred, complexity_override=None):
            mse = _mse(y, pred)
            parts_for_complexity = []
            for coef, texpr in used_terms:
                # find term stats if possible
                tu = 0
                tb = 0
                if texpr in val_map:
                    if texpr.startswith("sin(") or texpr.startswith("cos("):
                        tu = 1
                        if ("+" in texpr or "-" in texpr or "*" in texpr) and texpr not in ("sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)"):
                            # rough but OK for tie-breaking
                            tb = texpr.count("+") + texpr.count("-") + texpr.count("*")
                    elif "*sin" in texpr or "*cos" in texpr or ")*sin" in texpr or ")*cos" in texpr:
                        tu = 2
                        tb = texpr.count("*")
                cmul = 0 if abs(coef) == 1.0 else 1
                parts_for_complexity.append((coef, tu, tb, cmul, texpr))

            complexity = complexity_override
            if complexity is None:
                complexity = _complexity_from_parts(constant, parts_for_complexity)

            if (mse < best["mse"] - 1e-14) or (abs(mse - best["mse"]) <= 1e-14 and complexity < best["complexity"]):
                best["mse"] = mse
                best["complexity"] = complexity
                best["constant"] = float(constant)
                best["used"] = [(float(c), str(t)) for c, t in used_terms if c != 0.0]
                best["pred"] = pred

        # Try quick candidates
        for constant, used in quick_exprs:
            pred = np.zeros(n, dtype=np.float64) + float(constant)
            ok = True
            for coef, texpr in used:
                v = val_map.get(texpr, None)
                if v is None:
                    ok = False
                    break
                pred = pred + float(coef) * v
            if ok:
                expr = _build_expression(constant, used)
                # Very rough complexity override to encourage these if they fit:
                complexity = 1  # placeholder; computed later if selected by search anyway
                consider_solution(constant, used, pred, complexity_override=None)
                if best["mse"] <= 1e-24:
                    best["expr"] = expr
                    return {
                        "expression": best["expr"],
                        "predictions": best["pred"].tolist(),
                        "details": {"complexity": int(best["complexity"])},
                    }

        # Exhaustive sparse linear regression over library
        # Always include intercept (constant)
        max_terms = max(1, min(self.max_terms, m))
        idxs = list(range(m))

        # Precompute norms for quick pruning (optional)
        # But keep simple: full search up to max_terms with combinations
        for k in range(0, max_terms + 1):
            # k terms + intercept; allow k=0 => constant only
            if k == 0:
                A = ones
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                c0 = float(coef[0])
                pred = c0 + np.zeros(n, dtype=np.float64)
                consider_solution(c0, [], pred)
                continue

            for comb in combinations(idxs, k):
                Phi = lib[:, comb]
                A = np.hstack([ones, Phi])
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

                c0 = float(coef[0])
                coefs = coef[1:].astype(np.float64, copy=False)

                # prune tiny coefficients
                used = []
                pred = c0 + np.zeros(n, dtype=np.float64)
                unary_ops = 0
                binary_ops = 0
                component_count = 1 if c0 != 0.0 else 0

                for j, ti in enumerate(comb):
                    a = float(coefs[j])
                    if abs(a) <= self.coef_prune:
                        continue
                    term = terms[ti]
                    used.append((a, term.expr, term.unary_ops, term.binary_ops))
                    pred = pred + a * term.values
                    unary_ops += term.unary_ops
                    binary_ops += term.binary_ops
                    component_count += 1
                    if abs(a) != 1.0:
                        binary_ops += 1  # coef*term

                if component_count == 0:
                    continue
                if component_count >= 2:
                    binary_ops += (component_count - 1)  # additions/subtractions

                complexity = 2 * binary_ops + unary_ops

                used_terms = [(a, expr) for (a, expr, _, _) in used]
                consider_solution(c0, used_terms, pred, complexity_override=complexity)

                if best["mse"] <= 1e-24 and best["complexity"] <= 8:
                    # early exit if essentially perfect and very simple
                    break
            else:
                continue
            break

        # Post-process: snap coefficients if it doesn't hurt fit much
        c0 = best["constant"]
        used = best["used"]

        if self.snap and best["pred"] is not None and len(used) > 0:
            # reconstruct selected terms from library or eval map
            # We'll compute predictions again with snapped coefficients
            snapped_c0 = _snap_float(c0)
            snapped_used = []
            pred = np.zeros(n, dtype=np.float64) + snapped_c0

            # Build a lookup for term expr -> values and ops
            lib_map = {t.expr: t for t in terms}

            for a, texpr in used:
                sa = _snap_float(a)
                if abs(sa) <= self.coef_prune:
                    continue
                snapped_used.append((sa, texpr))
                t = lib_map.get(texpr, None)
                if t is not None:
                    pred = pred + sa * t.values
                else:
                    v = val_map.get(texpr, None)
                    if v is None:
                        # fallback: keep original predictions
                        pred = None
                        break
                    pred = pred + sa * v

            if pred is not None:
                mse_before = best["mse"]
                mse_after = _mse(y, pred)
                if mse_after <= mse_before * (1.0 + 1e-8) + 1e-12:
                    best["constant"] = float(snapped_c0)
                    best["used"] = [(float(a), str(t)) for a, t in snapped_used]
                    best["pred"] = pred

        best["expr"] = _build_expression(best["constant"], best["used"])

        # Recompute a reasonable complexity for details
        # Use term stats where available
        lib_map = {t.expr: t for t in terms}
        parts = []
        for a, texpr in best["used"]:
            t = lib_map.get(texpr, None)
            tu = t.unary_ops if t is not None else 0
            tb = t.binary_ops if t is not None else 0
            cmul = 0 if abs(a) == 1.0 else 1
            parts.append((a, tu, tb, cmul, texpr))
        best["complexity"] = _complexity_from_parts(best["constant"], parts)

        return {
            "expression": best["expr"],
            "predictions": None if best["pred"] is None else best["pred"].tolist(),
            "details": {"complexity": int(best["complexity"])},
        }