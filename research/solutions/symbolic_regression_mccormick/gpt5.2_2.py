import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 0)

    @staticmethod
    def _mse(y, yhat):
        r = y - yhat
        return float(np.mean(r * r))

    @staticmethod
    def _snap_value(v):
        if not np.isfinite(v):
            return v
        candidates = [
            0.0,
            1.0, -1.0,
            0.5, -0.5,
            1.5, -1.5,
            2.0, -2.0,
            2.5, -2.5,
            3.0, -3.0,
            4.0, -4.0,
            5.0, -5.0,
        ]
        av = abs(v)
        for c in candidates:
            if abs(v - c) <= 1e-8 * max(1.0, abs(c)):
                return float(c)
        r = np.round(v)
        if abs(v - r) <= 1e-8 * max(1.0, abs(r)):
            return float(r)
        return float(v)

    @staticmethod
    def _format_float(v):
        if not np.isfinite(v):
            return "0.0"
        v = float(v)
        s = f"{v:.12g}"
        if s == "-0":
            s = "0"
        return s

    @staticmethod
    def _build_expression(term_exprs, coeffs, tol=1e-10):
        parts = []
        const = 0.0
        for expr, c in zip(term_exprs, coeffs):
            if expr == "1":
                const += c
                continue
            if not np.isfinite(c) or abs(c) <= tol:
                continue
            c = Solution._snap_value(c)
            if abs(c) <= tol:
                continue
            if abs(c - 1.0) <= 1e-12:
                parts.append(expr)
            elif abs(c + 1.0) <= 1e-12:
                parts.append(f"-({expr})")
            else:
                parts.append(f"{Solution._format_float(c)}*({expr})")

        const = Solution._snap_value(const)
        if np.isfinite(const) and abs(const) > tol:
            parts.append(Solution._format_float(const))

        if not parts:
            return "0", 0.0

        expr = parts[0]
        for p in parts[1:]:
            if p.startswith("-(") or p.startswith("-"):
                expr = f"({expr}) + ({p})"
            else:
                expr = f"({expr}) + ({p})"
        return expr, const

    @staticmethod
    def _stlsq(Theta, y, lam=1e-8, max_iter=20):
        n_features = Theta.shape[1]
        active = np.ones(n_features, dtype=bool)
        coef = np.zeros(n_features, dtype=float)
        for _ in range(max_iter):
            if not np.any(active):
                break
            Th = Theta[:, active]
            try:
                c_active, _, _, _ = np.linalg.lstsq(Th, y, rcond=None)
            except np.linalg.LinAlgError:
                break
            coef_new = np.zeros(n_features, dtype=float)
            coef_new[active] = c_active
            new_active = np.abs(coef_new) > lam
            if np.array_equal(new_active, active):
                coef = coef_new
                break
            active = new_active
            coef = coef_new
        return coef

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        x1 = X[:, 0]
        x2 = X[:, 1]

        # Stage 1: McCormick-structure fit
        t_sin = np.sin(x1 + x2)
        t_sq = (x1 - x2) ** 2
        A = np.column_stack([t_sin, t_sq, x1, x2, np.ones_like(x1)])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.zeros(A.shape[1], dtype=float)

        yhat = A @ coef
        mse1 = self._mse(y, yhat)
        var_y = float(np.var(y)) if y.size else 0.0
        rel1 = mse1 / (var_y + 1e-12)

        term_exprs1 = ["sin(x1 + x2)", "(x1 - x2)**2", "x1", "x2", "1"]
        expr1, _ = self._build_expression(term_exprs1, coef, tol=1e-10)

        # Stage 2: broader sparse library if needed
        if not np.isfinite(mse1) or rel1 > 1e-6:
            terms = []
            exprs = []

            def add_term(val, expr):
                terms.append(val)
                exprs.append(expr)

            add_term(np.sin(x1 + x2), "sin(x1 + x2)")
            add_term(np.cos(x1 + x2), "cos(x1 + x2)")
            add_term(np.sin(x1), "sin(x1)")
            add_term(np.sin(x2), "sin(x2)")
            add_term(np.cos(x1), "cos(x1)")
            add_term(np.cos(x2), "cos(x2)")
            add_term((x1 - x2) ** 2, "(x1 - x2)**2")
            add_term((x1 + x2) ** 2, "(x1 + x2)**2")
            add_term(x1, "x1")
            add_term(x2, "x2")
            add_term(x1 * x2, "(x1*x2)")
            add_term(x1 ** 2, "(x1**2)")
            add_term(x2 ** 2, "(x2**2)")
            add_term(np.ones_like(x1), "1")

            Theta = np.column_stack(terms)
            scale = np.linalg.norm(Theta, axis=0)
            scale[scale == 0] = 1.0
            Theta_s = Theta / scale

            coef_s = self._stlsq(Theta_s, y, lam=1e-9, max_iter=25)
            coef2 = coef_s / scale

            yhat2 = Theta @ coef2
            mse2 = self._mse(y, yhat2)
            rel2 = mse2 / (var_y + 1e-12)

            if np.isfinite(mse2) and (mse2 < mse1 or (rel1 > 1e-4 and rel2 < rel1)):
                expr2, _ = self._build_expression(exprs, coef2, tol=1e-10)
                return {
                    "expression": expr2,
                    "predictions": yhat2.tolist(),
                    "details": {"mse": mse2, "mse_relative": rel2},
                }

        return {
            "expression": expr1,
            "predictions": yhat.tolist(),
            "details": {"mse": mse1, "mse_relative": rel1},
        }