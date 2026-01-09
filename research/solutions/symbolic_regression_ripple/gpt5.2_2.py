import numpy as np
import sympy as sp


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def _make_w_grid(max_g: float) -> np.ndarray:
        if not np.isfinite(max_g) or max_g <= 1e-12:
            return np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0], dtype=np.float64)

        w_min = max(0.1, 4.0 * np.pi / (max_g + 1e-12))
        w_max = min(400.0, 80.0 * np.pi / (max_g + 1e-12))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max < w_min:
            w_min, w_max = 0.1, 50.0
        if w_max < 1.5 * w_min:
            w_max = min(400.0, w_min * 50.0)

        grid_geom = np.geomspace(w_min, w_max, num=45).astype(np.float64)
        ints = np.array(
            [0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400],
            dtype=np.float64,
        )
        ints = ints[(ints >= w_min) & (ints <= w_max)]
        grid = np.unique(np.concatenate([grid_geom, ints]))
        return grid

    @staticmethod
    def _fmt_float(x: float) -> str:
        if not np.isfinite(x):
            return "0.0"
        ax = abs(x)
        if ax < 1e-15:
            return "0.0"
        return repr(float(x))

    @staticmethod
    def _poly_string(coeffs, basis_expr, coef_tol: float) -> str:
        parts = []
        for c, be in zip(coeffs, basis_expr):
            if not np.isfinite(c) or abs(c) <= coef_tol:
                continue
            cs = Solution._fmt_float(c)
            if be == "1":
                parts.append(f"{cs}")
            else:
                parts.append(f"({cs})*({be})")
        if not parts:
            return "0"
        return " + ".join(parts)

    @staticmethod
    def _count_complexity(expr_str: str) -> int:
        x1, x2 = sp.Symbol("x1"), sp.Symbol("x2")
        locals_map = {
            "x1": x1,
            "x2": x2,
            "sin": sp.sin,
            "cos": sp.cos,
            "exp": sp.exp,
            "log": sp.log,
        }
        try:
            expr = sp.sympify(expr_str, locals=locals_map)
        except Exception:
            return 10**9

        unary_funcs = (sp.sin, sp.cos, sp.exp, sp.log)
        binary_ops = 0
        unary_ops = 0

        def visit(e):
            nonlocal binary_ops, unary_ops
            if isinstance(e, sp.Add) or isinstance(e, sp.Mul):
                args = list(e.args)
                if len(args) >= 2:
                    binary_ops += len(args) - 1
                for a in args:
                    visit(a)
            elif isinstance(e, sp.Pow):
                binary_ops += 1
                visit(e.base)
                visit(e.exp)
            elif isinstance(e, sp.Function):
                if e.func in unary_funcs:
                    unary_ops += 1
                for a in e.args:
                    visit(a)
            else:
                for a in getattr(e, "args", ()):
                    visit(a)

        visit(expr)
        return int(2 * binary_ops + unary_ops)

    @staticmethod
    def _lincomb(coeffs: np.ndarray, basis_arrays: list) -> np.ndarray:
        out = np.zeros_like(basis_arrays[0], dtype=np.float64)
        for c, arr in zip(coeffs, basis_arrays):
            if c != 0.0:
                out += c * arr
        return out

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = X.shape[0]
        if n == 0:
            return {"expression": "0", "predictions": [], "details": {"complexity": 0}}

        x1 = X[:, 0]
        x2 = X[:, 1]
        r = x1 * x1 + x2 * x2
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            s = np.sqrt(np.maximum(r, 0.0) + 1e-15)

        ones = np.ones_like(r)
        r2 = r * r
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / (1.0 + r)

        r_expr = "(x1**2 + x2**2)"
        g_expr_r = r_expr
        g_expr_s = f"({r_expr})**0.5"

        basis_arrays = [ones, r, r2, inv]
        basis_expr = ["1", r_expr, f"({r_expr})**2", f"1/(1 + ({r_expr}))"]

        nb_base = len(basis_arrays)
        nb_amp = len(basis_arrays)

        w_grid_r = self._make_w_grid(float(np.max(r)))
        w_grid_s = self._make_w_grid(float(np.max(s)))

        best = None  # dict

        def eval_candidate(g_arr, g_expr, w, harmonics):
            nonlocal best
            harm = list(harmonics)
            k = nb_base + len(harm) * 2 * nb_amp
            A = np.empty((n, k), dtype=np.float64)
            col = 0
            for arr in basis_arrays:
                A[:, col] = arr
                col += 1

            with np.errstate(over="ignore", invalid="ignore"):
                for h in harm:
                    wh = float(w) * float(h)
                    arg = wh * g_arr
                    sv = np.sin(arg)
                    cv = np.cos(arg)
                    for arr in basis_arrays:
                        A[:, col] = arr * sv
                        col += 1
                    for arr in basis_arrays:
                        A[:, col] = arr * cv
                        col += 1

            try:
                coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            except Exception:
                return

            pred = A @ coefs
            resid = pred - y
            mse = float(np.mean(resid * resid))
            eps = 1e-30
            bic = float(n * np.log(mse + eps) + k * np.log(max(n, 2)))

            if best is None or bic < best["bic"]:
                best = {
                    "g_expr": g_expr,
                    "g_type": "s" if g_expr == g_expr_s else "r",
                    "w": float(w),
                    "harmonics": tuple(harm),
                    "coefs": coefs.astype(np.float64, copy=True),
                    "mse": mse,
                    "bic": bic,
                }

        # Polynomial-only baseline (no trig)
        A0 = np.column_stack(basis_arrays).astype(np.float64, copy=False)
        try:
            c0, _, _, _ = np.linalg.lstsq(A0, y, rcond=None)
            p0 = A0 @ c0
            mse0 = float(np.mean((p0 - y) ** 2))
            bic0 = float(n * np.log(mse0 + 1e-30) + len(c0) * np.log(max(n, 2)))
            best = {
                "g_expr": g_expr_r,
                "g_type": "r",
                "w": 0.0,
                "harmonics": tuple(),
                "coefs": c0.astype(np.float64, copy=True),
                "mse": mse0,
                "bic": bic0,
            }
        except Exception:
            best = {
                "g_expr": g_expr_r,
                "g_type": "r",
                "w": 0.0,
                "harmonics": tuple(),
                "coefs": np.zeros(nb_base, dtype=np.float64),
                "mse": float(np.mean(y * y)),
                "bic": float("inf"),
            }

        harmonics_options = [(1,), (1, 2)]
        for w in w_grid_r:
            for harm in harmonics_options:
                eval_candidate(r, g_expr_r, w, harm)
        for w in w_grid_s:
            for harm in harmonics_options:
                eval_candidate(s, g_expr_s, w, harm)

        # Prune and choose simplest expression not degrading MSE too much.
        best_full = best
        coefs_full = best_full["coefs"]
        mse_full = best_full["mse"]

        def build_from_coefs(coefs: np.ndarray, coef_tol: float):
            idx = 0
            base_c = coefs[idx : idx + nb_base]
            idx += nb_base

            base_str = self._poly_string(base_c, basis_expr, coef_tol)
            parts = []
            if base_str != "0":
                parts.append(f"({base_str})")

            w = best_full["w"]
            g_expr = best_full["g_expr"]
            harm = best_full["harmonics"]
            for h in harm:
                sin_c = coefs[idx : idx + nb_amp]
                idx += nb_amp
                cos_c = coefs[idx : idx + nb_amp]
                idx += nb_amp

                ps = self._poly_string(sin_c, basis_expr, coef_tol)
                pc = self._poly_string(cos_c, basis_expr, coef_tol)
                if ps == "0" and pc == "0":
                    continue
                wh = float(w) * float(h)
                arg_str = f"({self._fmt_float(wh)})*({g_expr})"
                subparts = []
                if ps != "0":
                    subparts.append(f"(({ps}))*sin({arg_str})")
                if pc != "0":
                    subparts.append(f"(({pc}))*cos({arg_str})")
                parts.append("(" + " + ".join(subparts) + ")")

            if not parts:
                expr = "0"
            else:
                expr = " + ".join(parts)
            return expr

        def predict_from_coefs(coefs: np.ndarray, coef_tol: float):
            idx = 0
            base_c = coefs[idx : idx + nb_base].copy()
            idx += nb_base
            base_c[np.abs(base_c) <= coef_tol] = 0.0
            pred = self._lincomb(base_c, basis_arrays)

            w = best_full["w"]
            harm = best_full["harmonics"]
            g_arr = s if best_full["g_type"] == "s" else r

            with np.errstate(over="ignore", invalid="ignore"):
                for h in harm:
                    sin_c = coefs[idx : idx + nb_amp].copy()
                    idx += nb_amp
                    cos_c = coefs[idx : idx + nb_amp].copy()
                    idx += nb_amp
                    sin_c[np.abs(sin_c) <= coef_tol] = 0.0
                    cos_c[np.abs(cos_c) <= coef_tol] = 0.0
                    if np.all(sin_c == 0.0) and np.all(cos_c == 0.0):
                        continue
                    wh = float(w) * float(h)
                    arg = wh * g_arr
                    sv = np.sin(arg)
                    cv = np.cos(arg)
                    amp_s = self._lincomb(sin_c, basis_arrays)
                    amp_c = self._lincomb(cos_c, basis_arrays)
                    pred = pred + amp_s * sv + amp_c * cv
            return pred

        maxabs = float(np.max(np.abs(coefs_full))) if coefs_full.size else 0.0
        yscale = float(np.std(y)) if np.isfinite(np.std(y)) else 1.0
        base_tol = max(1e-14, 1e-10 * max(1.0, yscale), 1e-12 * max(1.0, maxabs))

        tol_multipliers = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 0.0]

        best_choice = None
        for mult in tol_multipliers:
            coef_tol = base_tol * (mult / 1e-7) if mult > 0 else 0.0
            if mult == 0.0:
                coef_tol = 0.0
            expr = build_from_coefs(coefs_full, coef_tol)
            pred = predict_from_coefs(coefs_full, coef_tol)
            mse = float(np.mean((pred - y) ** 2))
            if mse <= mse_full * 1.002 + 1e-18:
                comp = self._count_complexity(expr)
                if best_choice is None or comp < best_choice["complexity"] or (comp == best_choice["complexity"] and mse < best_choice["mse"]):
                    best_choice = {"expression": expr, "pred": pred, "mse": mse, "complexity": comp, "coef_tol": coef_tol}

        if best_choice is None:
            expr = build_from_coefs(coefs_full, 0.0)
            pred = predict_from_coefs(coefs_full, 0.0)
            mse = float(np.mean((pred - y) ** 2))
            comp = self._count_complexity(expr)
            best_choice = {"expression": expr, "pred": pred, "mse": mse, "complexity": comp, "coef_tol": 0.0}

        details = {
            "mse": float(best_choice["mse"]),
            "complexity": int(best_choice["complexity"]),
            "model": {
                "g_type": best_full["g_type"],
                "w": float(best_full["w"]),
                "harmonics": list(best_full["harmonics"]),
                "coef_tol": float(best_choice["coef_tol"]),
            },
        }

        return {
            "expression": best_choice["expression"],
            "predictions": best_choice["pred"].tolist(),
            "details": details,
        }