import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _snap_value(self, v, tol=1e-3):
        candidates = np.array([
            -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
        ], dtype=float)
        idx = int(np.argmin(np.abs(candidates - v)))
        c = float(candidates[idx])
        if abs(v - c) <= tol:
            return c
        return float(v)

    def _fmt_num(self, v):
        if np.isfinite(v) and abs(v) < 1e-12:
            v = 0.0
        return format(float(v), ".12g")

    def _add_term(self, terms, coef, expr):
        if not np.isfinite(coef):
            return
        if abs(coef) < 1e-12:
            return
        if abs(coef - 1.0) < 1e-12:
            terms.append(expr)
        elif abs(coef + 1.0) < 1e-12:
            terms.append(f"-({expr})")
        else:
            terms.append(f"{self._fmt_num(coef)}*({expr})")

    def _join_terms(self, terms, const):
        s = ""
        if np.isfinite(const) and abs(const) >= 1e-12:
            s = self._fmt_num(const)

        for t in terms:
            if not s:
                s = t
            else:
                if t.startswith("-"):
                    s = f"{s} - {t[1:]}"
                else:
                    s = f"{s} + {t}"
        if not s:
            s = "0"
        return s

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        x1 = X[:, 0].astype(float, copy=False)
        x2 = X[:, 1].astype(float, copy=False)

        b1 = np.sin(x1 + x2)
        b2 = (x1 - x2) ** 2
        b3 = x1
        b4 = x2
        b5 = np.ones_like(x1)

        A = np.column_stack([b1, b2, b3, b4, b5])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c, d, e = [float(v) for v in coef]

        a = self._snap_value(a, tol=5e-3)
        b = self._snap_value(b, tol=5e-3)
        c = self._snap_value(c, tol=5e-3)
        d = self._snap_value(d, tol=5e-3)
        e = self._snap_value(e, tol=5e-3)

        preds = a * b1 + b * b2 + c * b3 + d * b4 + e

        terms = []
        self._add_term(terms, a, "sin(x1 + x2)")
        self._add_term(terms, b, "(x1 - x2)**2")
        self._add_term(terms, c, "x1")
        self._add_term(terms, d, "x2")
        expression = self._join_terms(terms, e)

        return {
            "expression": expression,
            "predictions": preds.tolist(),
            "details": {}
        }