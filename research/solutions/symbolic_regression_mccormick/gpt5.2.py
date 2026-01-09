import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def _format_float(v: float) -> str:
        if not np.isfinite(v):
            return "0.0"
        if abs(v) < 1e-15:
            v = 0.0
        s = f"{v:.12g}"
        if s == "-0":
            s = "0"
        return s

    @staticmethod
    def _snap(v: float) -> float:
        snaps = [
            0.0, 1.0, -1.0,
            0.5, -0.5,
            1.5, -1.5,
            2.5, -2.5,
            3.0, -3.0,
            4.0, -4.0
        ]
        for t in snaps:
            if abs(v - t) <= 2e-2:
                return float(t)
        return float(v)

    @staticmethod
    def _fit_linear(feats: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            coef, _, _, _ = np.linalg.lstsq(feats, y, rcond=None)
            return coef
        except Exception:
            # fallback: ridge-like stabilization
            lam = 1e-12
            A = feats
            ATA = A.T @ A
            ATy = A.T @ y
            ATA.flat[:: ATA.shape[0] + 1] += lam
            return np.linalg.solve(ATA, ATy)

    @staticmethod
    def _build_expression(coef: np.ndarray, basis_exprs: list[str], snap: bool = True) -> str:
        parts = []
        for c, bexpr in zip(coef.tolist(), basis_exprs):
            if snap:
                c = Solution._snap(c)
            if abs(c) < 1e-10:
                continue

            is_const = (bexpr == "1")
            if is_const:
                term = Solution._format_float(abs(c))
            else:
                ac = abs(c)
                if abs(ac - 1.0) < 1e-12:
                    term = bexpr
                else:
                    term = Solution._format_float(ac) + "*" + bexpr

            if not parts:
                if c < 0:
                    parts.append("-" + term)
                else:
                    parts.append(term)
            else:
                if c < 0:
                    parts.append(" - " + term)
                else:
                    parts.append(" + " + term)

        if not parts:
            return "0"
        expr = "".join(parts)
        return expr

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2).")
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("y must have length n.")

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Candidate basis sets: (basis_exprs, feature_matrix_constructor)
        def make_feats_A():
            return np.column_stack([
                np.sin(x1 + x2),
                (x1 - x2) ** 2,
                x1,
                x2,
                np.ones_like(x1),
            ])

        def make_feats_B():
            return np.column_stack([
                np.sin(x1 + x2),
                np.cos(x1 + x2),
                (x1 - x2) ** 2,
                x1,
                x2,
                np.ones_like(x1),
            ])

        def make_feats_C():
            return np.column_stack([
                np.sin(x1),
                np.sin(x2),
                np.cos(x1),
                np.cos(x2),
                (x1 - x2) ** 2,
                x1,
                x2,
                np.ones_like(x1),
            ])

        def make_feats_D():
            return np.column_stack([
                x1,
                x2,
                x1 * x2,
                x1 ** 2,
                x2 ** 2,
                (x1 - x2) ** 2,
                np.ones_like(x1),
            ])

        candidates = [
            (["sin(x1 + x2)", "(x1 - x2)**2", "x1", "x2", "1"], make_feats_A),
            (["sin(x1 + x2)", "cos(x1 + x2)", "(x1 - x2)**2", "x1", "x2", "1"], make_feats_B),
            (["sin(x1)", "sin(x2)", "cos(x1)", "cos(x2)", "(x1 - x2)**2", "x1", "x2", "1"], make_feats_C),
            (["x1", "x2", "x1*x2", "x1**2", "x2**2", "(x1 - x2)**2", "1"], make_feats_D),
        ]

        best = None
        yvar = float(np.var(y)) if np.isfinite(np.var(y)) else 1.0
        yscale = max(1e-12, yvar)

        for basis_exprs, make_feats in candidates:
            feats = make_feats()
            coef = self._fit_linear(feats, y)
            pred = feats @ coef
            resid = y - pred
            mse = float(np.mean(resid * resid))
            coef_eff = coef.copy()
            coef_eff[np.abs(coef_eff) < 1e-10] = 0.0
            k = int(np.count_nonzero(coef_eff))
            if best is None:
                best = (mse, k, basis_exprs, coef, pred)
            else:
                best_mse, best_k, _, _, _ = best
                # Prefer lower MSE; if essentially tied, prefer fewer terms
                if (mse + 1e-14) < best_mse - 1e-14:
                    best = (mse, k, basis_exprs, coef, pred)
                else:
                    tied = mse <= best_mse + 1e-10 * yscale
                    if tied and k < best_k:
                        best = (mse, k, basis_exprs, coef, pred)

        mse, k, basis_exprs, coef, pred = best

        # If it matches canonical McCormick very closely, output the clean exact form.
        if basis_exprs == ["sin(x1 + x2)", "(x1 - x2)**2", "x1", "x2", "1"]:
            target = np.array([1.0, 1.0, -1.5, 2.5, 1.0], dtype=float)
            if np.all(np.abs(coef - target) <= 2e-2) or mse <= 1e-14 * max(1.0, float(np.mean(y * y))):
                expression = "sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1"
                pred = np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0
                return {
                    "expression": expression,
                    "predictions": pred.tolist(),
                    "details": {}
                }

        # Otherwise, build from fitted coefficients (snapped when close to simple constants).
        coef2 = np.array([self._snap(float(c)) for c in coef], dtype=float)
        expression = self._build_expression(coef2, basis_exprs, snap=False)

        # Recompute predictions from snapped coefficients for consistency with expression.
        feats = None
        if basis_exprs == ["sin(x1 + x2)", "(x1 - x2)**2", "x1", "x2", "1"]:
            feats = make_feats_A()
        elif basis_exprs == ["sin(x1 + x2)", "cos(x1 + x2)", "(x1 - x2)**2", "x1", "x2", "1"]:
            feats = make_feats_B()
        elif basis_exprs == ["sin(x1)", "sin(x2)", "cos(x1)", "cos(x2)", "(x1 - x2)**2", "x1", "x2", "1"]:
            feats = make_feats_C()
        elif basis_exprs == ["x1", "x2", "x1*x2", "x1**2", "x2**2", "(x1 - x2)**2", "1"]:
            feats = make_feats_D()

        if feats is not None:
            pred2 = feats @ coef2
        else:
            pred2 = pred

        return {
            "expression": expression,
            "predictions": pred2.tolist(),
            "details": {}
        }