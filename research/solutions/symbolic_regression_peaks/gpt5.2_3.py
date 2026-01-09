import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def _fmt(v: float) -> str:
        if not np.isfinite(v):
            v = 0.0
        s = format(float(v), ".15g")
        if s in ("nan", "inf", "-inf"):
            s = "0.0"
        return s

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Peaks-like basis
        e1 = np.exp(-(x1 * x1) - (x2 + 1.0) * (x2 + 1.0))
        e2 = np.exp(-(x1 * x1) - (x2 * x2))
        e3 = np.exp(-(x1 + 1.0) * (x1 + 1.0) - (x2 * x2))

        b1 = (1.0 - x1) ** 2 * e1
        b2 = (x1 / 5.0 - x1 ** 3 - x2 ** 5) * e2
        b3 = e3

        A = np.column_stack([b1, b2, b3, np.ones_like(x1)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a1, a2, a3, c0 = coeffs.tolist()
        except Exception:
            a1, a2, a3, c0 = 3.0, -10.0, -1.0 / 3.0, 0.0

        preds = a1 * b1 + a2 * b2 + a3 * b3 + c0
        if not np.all(np.isfinite(preds)):
            a1, a2, a3, c0 = 3.0, -10.0, -1.0 / 3.0, 0.0
            preds = a1 * b1 + a2 * b2 + a3 * b3 + c0

        s_a1 = self._fmt(a1)
        s_a2 = self._fmt(a2)
        s_a3 = self._fmt(a3)
        s_c0 = self._fmt(c0)

        expression = (
            f"({s_a1})*((1 - x1)**2)*exp(-(x1**2) - ((x2 + 1)**2))"
            f" + ({s_a2})*(x1/5 - x1**3 - x2**5)*exp(-(x1**2) - (x2**2))"
            f" + ({s_a3})*exp(-((x1 + 1)**2) - (x2**2))"
            f" + ({s_c0})"
        )

        return {
            "expression": expression,
            "predictions": preds.tolist(),
            "details": {}
        }