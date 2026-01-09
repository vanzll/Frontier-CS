import os
import math
import numpy as np
import sympy as sp

np.seterr(all="ignore")


def _safe_exp(x):
    return np.exp(np.clip(x, -80.0, 80.0))


def _linear_baseline_mse(X, y):
    n = X.shape[0]
    A = np.column_stack([X, np.ones(n, dtype=X.dtype)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    err = pred - y
    return float(np.mean(err * err))


def _generate_exponents(nvars, degmax):
    exps = []

    def rec(i, remaining, cur):
        if i == nvars:
            exps.append(tuple(cur))
            return
        for d in range(remaining + 1):
            cur[i] = d
            rec(i + 1, remaining - d, cur)

    for total in range(degmax + 1):
        cur = [0] * nvars
        rec(0, total, cur)
    return exps


def _monomial_expr(exp_tuple, varnames):
    parts = []
    for v, p in zip(varnames, exp_tuple):
        if p == 0:
            continue
        if p == 1:
            parts.append(v)
        else:
            parts.append(f"{v}**{p}")
    if not parts:
        return "1"
    return "*".join(parts)


def _compute_monomial_matrix(X, exps, maxdeg):
    n, d = X.shape
    x_pows = [[None] * (maxdeg + 1) for _ in range(d)]
    for i in range(d):
        x = X[:, i]
        x_pows[i][0] = np.ones(n, dtype=X.dtype)
        if maxdeg >= 1:
            x_pows[i][1] = x
        for p in range(2, maxdeg + 1):
            x_pows[i][p] = x_pows[i][p - 1] * x

    Phi = np.empty((n, len(exps)), dtype=X.dtype)
    for k, e in enumerate(exps):
        col = np.ones(n, dtype=X.dtype)
        for i, p in enumerate(e):
            if p:
                col = col * x_pows[i][p]
        Phi[:, k] = col
    return Phi


def _ridge_solve(A, y, lam=1e-10):
    m = A.shape[1]
    ATA = A.T @ A
    ATy = A.T @ y
    diag_add = lam * (np.trace(ATA) / max(m, 1) + 1.0)
    ATA.flat[:: m + 1] += diag_add
    try:
        c = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def _least_squares(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def _snap_float_str(x):
    if not np.isfinite(x):
        return "0.0"
    ax = abs(x)
    if ax < 1e-14:
        return "0.0"
    r = round(x)
    if abs(x - r) < 1e-10:
        return str(int(r))
    common = [0.5, 0.25, 0.75, 0.3333333333333333, 0.6666666666666666, 0.2, 0.4, 0.6, 0.8]
    sgn = -1.0 if x < 0 else 1.0
    for v in common:
        if abs(x - sgn * v) < 1e-10:
            return format(sgn * v, ".12g")
    return format(float(x), ".12g")


def _build_poly_expr(coefs, exps, varnames, coef_prune_rel=1e-12):
    coefs = np.asarray(coefs, dtype=np.float64)
    maxabs = float(np.max(np.abs(coefs))) if coefs.size else 0.0
    tol = max(1e-14, coef_prune_rel * maxabs)

    terms = []
    for c, e in zip(coefs, exps):
        if not np.isfinite(c) or abs(c) <= tol:
            continue
        mon = _monomial_expr(e, varnames)
        cs = _snap_float_str(c)
        if mon == "1":
            terms.append(f"({cs})")
        else:
            if abs(c - 1.0) < 1e-12:
                terms.append(f"({mon})")
            elif abs(c + 1.0) < 1e-12:
                terms.append(f"(-({mon}))")
            else:
                terms.append(f"({cs})*({mon})")

    if not terms:
        return "0.0"
    return " + ".join(terms)


def _eval_expr_numpy(expr_str, X):
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    env = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "sin": np.sin,
        "cos": np.cos,
        "exp": _safe_exp,
        "log": np.log,
    }
    try:
        pred = eval(expr_str, {"__builtins__": {}}, env)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != (X.shape[0],):
            pred = np.reshape(pred, (X.shape[0],))
        return pred
    except Exception:
        return None


def _allowed_sympy_expr(expr):
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")
    allowed_syms = {x1, x2, x3, x4}
    if not set(expr.free_symbols).issubset(allowed_syms):
        return False
    if expr.has(sp.Piecewise, sp.Abs, sp.sign, sp.Max, sp.Min, sp.Heaviside, sp.floor, sp.ceiling):
        return False
    for f in expr.atoms(sp.Function):
        fn = f.func
        if fn not in (sp.exp, sp.log, sp.sin, sp.cos):
            return False
    return True


def _sanitize_sympy_constants(expr):
    expr = expr.xreplace({sp.pi: sp.Float(np.pi), sp.E: sp.Float(np.e)})
    return expr


class Solution:
    def __init__(self, **kwargs):
        self.random_state = int(kwargs.get("random_state", 42))
        self.max_deg = int(kwargs.get("max_deg", 4))
        self.try_pysr = bool(kwargs.get("try_pysr", True))
        self.pysr_time_limit = float(kwargs.get("pysr_time_limit", 35.0))

    def _fit_structured(self, X, y):
        varnames = ["x1", "x2", "x3", "x4"]
        n = X.shape[0]
        maxdeg = max(2, min(6, self.max_deg))
        exps = _generate_exponents(4, maxdeg)
        Phi = _compute_monomial_matrix(X, exps, maxdeg)

        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        r2_all = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4
        r2_12 = x1 * x1 + x2 * x2
        r2_34 = x3 * x3 + x4 * x4
        r2_pm = (x1 - x2) * (x1 - x2) + (x3 - x4) * (x3 - x4)
        r2_pp = (x1 + x2) * (x1 + x2) + (x3 + x4) * (x3 + x4)

        variants = [
            ("1.0", None),
            ("exp(-a*(x1**2+x2**2+x3**2+x4**2))", r2_all),
            ("exp(-a*(x1**2+x2**2))", r2_12),
            ("exp(-a*(x3**2+x4**2))", r2_34),
            ("exp(-a*((x1-x2)**2+(x3-x4)**2))", r2_pm),
            ("exp(-a*((x1+x2)**2+(x3+x4)**2))", r2_pp),
        ]
        a_grid = [0.25, 0.5, 1.0, 2.0]

        best = None
        for base_expr, R in variants:
            if R is None:
                dampings = [(1.0, np.ones(n, dtype=X.dtype), "1.0")]
            else:
                dampings = []
                for a in a_grid:
                    g = _safe_exp(-a * R)
                    a_str = _snap_float_str(a)
                    gexpr = base_expr.replace("a", a_str)
                    dampings.append((a, g, gexpr))
            for _, g, gexpr in dampings:
                A = Phi * g[:, None]
                c = _ridge_solve(A, y, lam=1e-10)
                pred = A @ c
                err = pred - y
                mse = float(np.mean(err * err))
                if not np.isfinite(mse):
                    continue
                if best is None or mse < best[0]:
                    poly_expr = _build_poly_expr(c, exps, varnames, coef_prune_rel=1e-12)
                    if gexpr != "1.0":
                        expr = f"({poly_expr})*({gexpr})"
                    else:
                        expr = poly_expr
                    best = (mse, expr, pred, {"structured": True, "damping": gexpr, "deg": maxdeg})
        return best

    def _fit_multi_damping_sparse(self, X, y, k_keep=16):
        varnames = ["x1", "x2", "x3", "x4"]
        n = X.shape[0]
        deg = 2
        exps = _generate_exponents(4, deg)
        Phi = _compute_monomial_matrix(X, exps, deg)

        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        r2_all = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4
        r2_12 = x1 * x1 + x2 * x2
        r2_34 = x3 * x3 + x4 * x4

        damp_list = [
            ("1.0", np.ones(n, dtype=X.dtype)),
            ("exp(-0.5*(x1**2+x2**2+x3**2+x4**2))", _safe_exp(-0.5 * r2_all)),
            ("exp(-(x1**2+x2**2+x3**2+x4**2))", _safe_exp(-1.0 * r2_all)),
            ("exp(-0.5*(x1**2+x2**2))", _safe_exp(-0.5 * r2_12)),
            ("exp(-(x1**2+x2**2))", _safe_exp(-1.0 * r2_12)),
            ("exp(-0.5*(x3**2+x4**2))", _safe_exp(-0.5 * r2_34)),
            ("exp(-(x3**2+x4**2))", _safe_exp(-1.0 * r2_34)),
        ]

        cols = []
        col_expr = []
        for gexpr, g in damp_list:
            G = Phi * g[:, None]
            cols.append(G)
            for e in exps:
                mon = _monomial_expr(e, varnames)
                if gexpr == "1.0":
                    col_expr.append(mon)
                else:
                    if mon == "1":
                        col_expr.append(gexpr)
                    else:
                        col_expr.append(f"({mon})*({gexpr})")

        A = np.concatenate(cols, axis=1)
        c = _ridge_solve(A, y, lam=1e-9)

        idx = np.argsort(-np.abs(c))[: max(1, min(k_keep, c.size))]
        A2 = A[:, idx]
        c2 = _least_squares(A2, y)
        pred = A2 @ c2
        err = pred - y
        mse = float(np.mean(err * err))
        if not np.isfinite(mse):
            return None

        terms = []
        maxabs = float(np.max(np.abs(c2))) if c2.size else 0.0
        tol = max(1e-14, 1e-12 * maxabs)
        for coef, j in zip(c2, idx):
            if not np.isfinite(coef) or abs(coef) <= tol:
                continue
            cs = _snap_float_str(coef)
            te = col_expr[j]
            if te == "1":
                terms.append(f"({cs})")
            else:
                if abs(coef - 1.0) < 1e-12:
                    terms.append(f"({te})")
                elif abs(coef + 1.0) < 1e-12:
                    terms.append(f"(-({te}))")
                else:
                    terms.append(f"({cs})*({te})")

        expr = " + ".join(terms) if terms else "0.0"
        return (mse, expr, pred, {"multi_damping_sparse": True, "deg": deg, "k_keep": int(k_keep)})

    def _try_pysr(self, X, y):
        try:
            os.environ.setdefault("JULIA_NUM_THREADS", str(max(1, min(8, os.cpu_count() or 1))))
            os.environ.setdefault("SYMBOLIC_REGRESSION_QUIET", "true")
            from pysr import PySRRegressor  # type: ignore
        except Exception:
            return None

        model = PySRRegressor(
            niterations=200,
            timeout_in_seconds=float(self.pysr_time_limit),
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp"],
            populations=12,
            population_size=40,
            maxsize=28,
            verbosity=0,
            progress=False,
            random_state=self.random_state,
            temp_equation_file=True,
            nested_constraints={"exp": {"exp": 0}},
            multithreading=True,
        )

        try:
            model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        except Exception:
            return None

        eqs = getattr(model, "equations_", None)
        if eqs is None or len(eqs) == 0:
            try:
                sym = model.sympy()
                sym = _sanitize_sympy_constants(sym)
                if not _allowed_sympy_expr(sym):
                    return None
                expr = str(sym)
                pred = _eval_expr_numpy(expr, X)
                if pred is None or not np.all(np.isfinite(pred)):
                    return None
                mse = float(np.mean((pred - y) ** 2))
                return (mse, expr, pred, {"pysr": True})
            except Exception:
                return None

        x1s, x2s, x3s, x4s = sp.symbols("x1 x2 x3 x4")
        best = None

        # Try best few (lowest loss)
        try:
            eqs_sorted = eqs.sort_values("loss", ascending=True).head(12)
        except Exception:
            eqs_sorted = eqs.head(12)

        for _, row in eqs_sorted.iterrows():
            s = None
            for col in ("sympy_format", "equation"):
                if col in row and isinstance(row[col], str) and row[col].strip():
                    s = row[col]
                    break
            if not s:
                continue
            try:
                sym = sp.sympify(s)
                sym = _sanitize_sympy_constants(sym)
                if not _allowed_sympy_expr(sym):
                    continue
                f = sp.lambdify((x1s, x2s, x3s, x4s), sym, modules={"sin": np.sin, "cos": np.cos, "exp": _safe_exp, "log": np.log})
                pred = f(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
                pred = np.asarray(pred, dtype=np.float64)
                if pred.shape != (X.shape[0],):
                    pred = np.reshape(pred, (X.shape[0],))
                if not np.all(np.isfinite(pred)):
                    continue
                mse = float(np.mean((pred - y) ** 2))
                if not np.isfinite(mse):
                    continue
                expr = str(sym)
                if best is None or mse < best[0]:
                    best = (mse, expr, pred, {"pysr": True})
            except Exception:
                continue

        return best

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError("X must have shape (n, 4)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y must have shape (n,)")

        mse_lin = _linear_baseline_mse(X, y)
        var_y = float(np.var(y) + 1e-18)

        candidates = []

        structured = self._fit_structured(X, y)
        if structured is not None:
            candidates.append(structured)

        sparse = self._fit_multi_damping_sparse(X, y, k_keep=16)
        if sparse is not None:
            candidates.append(sparse)

        best_current = min(candidates, key=lambda t: t[0]) if candidates else None
        good_enough = False
        if best_current is not None:
            mse = best_current[0]
            if mse <= 1e-10 * max(1.0, var_y):
                good_enough = True
            elif np.isfinite(mse_lin) and mse <= 1e-3 * max(mse_lin, 1e-18):
                good_enough = True

        pysr_res = None
        if self.try_pysr and not good_enough:
            pysr_res = self._try_pysr(X, y)
            if pysr_res is not None:
                candidates.append(pysr_res)

        if not candidates:
            expression = "0.0"
            predictions = np.zeros(X.shape[0], dtype=np.float64)
            return {"expression": expression, "predictions": predictions.tolist(), "details": {"fallback": True}}

        best = min(candidates, key=lambda t: t[0])
        mse, expression, pred, details = best

        if pred is None or not np.all(np.isfinite(pred)):
            pred = _eval_expr_numpy(expression, X)
        if pred is None or not np.all(np.isfinite(pred)):
            pred = np.zeros(X.shape[0], dtype=np.float64)

        details = dict(details or {})
        details["mse"] = float(mse)

        return {"expression": expression, "predictions": pred.tolist(), "details": details}