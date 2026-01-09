import numpy as np
try:
    from pysr import PySRRegressor
    _HAVE_PYSR = True
except Exception:
    PySRRegressor = None
    _HAVE_PYSR = False


class Solution:
    def __init__(self, use_pysr: bool = True, **kwargs):
        self.use_pysr = use_pysr and _HAVE_PYSR

    def _peaks_base(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        term1 = 3.0 * (1.0 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1.0) ** 2)
        term2 = -10.0 * (x1 / 5.0 - x1 ** 3 - x2 ** 5) * np.exp(-(x1 ** 2) - (x2 ** 2))
        term3 = -(1.0 / 3.0) * np.exp(-(x1 + 1.0) ** 2 - x2 ** 2)
        return term1 + term2 + term3

    def _fit_peaks_linear_combo(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]
        base = self._peaks_base(x1, x2)

        n = x1.shape[0]
        A = np.column_stack([base, x1, x2, np.ones(n, dtype=x1.dtype)])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        c0, c1, c2, c3 = coeffs

        # Threshold small coefficients to zero for simpler expressions
        max_abs = float(np.max(np.abs(coeffs))) if coeffs.size > 0 else 1.0
        tol = 1e-6 * max(1.0, max_abs)
        if abs(c0) < tol:
            c0 = 0.0
        if abs(c1) < tol:
            c1 = 0.0
        if abs(c2) < tol:
            c2 = 0.0
        if abs(c3) < tol:
            c3 = 0.0

        preds = c0 * base + c1 * x1 + c2 * x2 + c3

        base_expr = (
            "3*(1 - x1)**2*exp(-(x1**2) - (x2 + 1)**2)"
            " - 10*(x1/5 - x1**3 - x2**5)*exp(-(x1**2) - (x2**2))"
            " - (1/3)*exp(-(x1 + 1)**2 - x2**2)"
        )

        terms = []
        if c0 != 0.0:
            terms.append(f"({c0:.12g})*({base_expr})")
        if c1 != 0.0:
            terms.append(f"({c1:.12g})*x1")
        if c2 != 0.0:
            terms.append(f"({c2:.12g})*x2")
        if c3 != 0.0 or not terms:
            terms.append(f"({c3:.12g})")

        expression = " + ".join(terms) if terms else "0"

        return expression, preds

    def _fit_pysr(self, X: np.ndarray, y: np.ndarray):
        model = PySRRegressor(
            niterations=40,
            ncycles_per_iteration=200,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=15,
            population_size=33,
            maxsize=25,
            verbosity=0,
            progress=False,
            random_state=42,
        )
        model.fit(X, y, variable_names=["x1", "x2"])
        best_expr = model.sympy()
        expression = str(best_expr)
        preds = model.predict(X)
        return expression, preds

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        best_expression = None
        best_preds = None
        best_mse = float("inf")

        # Candidate 1: PySR (if available)
        if self.use_pysr:
            try:
                expr_pysr, preds_pysr = self._fit_pysr(X, y)
                mse_pysr = float(np.mean((preds_pysr - y) ** 2))
                best_expression = expr_pysr
                best_preds = preds_pysr
                best_mse = mse_pysr
            except Exception:
                self.use_pysr = False

        # Candidate 2: Peaks-based linear combination (always computed)
        expr_peaks, preds_peaks = self._fit_peaks_linear_combo(X, y)
        mse_peaks = float(np.mean((preds_peaks - y) ** 2))

        if best_expression is None or mse_peaks < best_mse:
            best_expression = expr_peaks
            best_preds = preds_peaks

        return {
            "expression": best_expression,
            "predictions": best_preds.tolist(),
            "details": {}
        }