import numpy as np

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    _HAS_PYSR = False


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit_with_pysr(self, X, y):
        if not _HAS_PYSR:
            return None

        n_samples = X.shape[0]

        if n_samples < 1000:
            niterations = 60
        elif n_samples < 5000:
            niterations = 40
        else:
            niterations = 30

        try:
            model = PySRRegressor(
                niterations=niterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "log"],
                populations=15,
                population_size=33,
                maxsize=25,
                ncyclesperiteration=300,
                model_selection="best",
                verbosity=0,
                progress=False,
                random_state=42,
            )

            model.fit(X, y, variable_names=["x1", "x2"])

            try:
                best_expr = model.sympy()
                expression = str(best_expr)
            except Exception:
                eq_df = getattr(model, "equations_", None)
                if eq_df is None or len(eq_df) == 0:
                    return None
                try:
                    row = eq_df.sort_values("loss").iloc[0]
                    expression = str(row.get("sympy_format", row.get("equation")))
                except Exception:
                    return None

            predictions = model.predict(X)
            return expression, predictions
        except Exception:
            return None

    def _fit_fallback(self, X, y):
        x1 = X[:, 0]
        x2 = X[:, 1]
        r2 = x1 * x1 + x2 * x2

        s1 = np.sin(r2)
        c1 = np.cos(r2)
        rs1 = r2 * s1
        rc1 = r2 * c1
        r2_sq = r2 * r2
        r2s1 = r2_sq * s1
        r2c1 = r2_sq * c1
        ones = np.ones_like(r2)

        A = np.column_stack([s1, c1, rs1, rc1, r2s1, r2c1, r2, r2_sq, ones])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        predictions = A @ coeffs

        def fmt(c):
            if abs(c) < 1e-12:
                return None
            return f"{c:.12g}"

        r2_expr = "(x1**2 + x2**2)"
        basis_exprs = [
            f"sin{r2_expr}",
            f"cos{r2_expr}",
            f"{r2_expr}*sin{r2_expr}",
            f"{r2_expr}*cos{r2_expr}",
            f"({r2_expr}**2)*sin{r2_expr}",
            f"({r2_expr}**2)*cos{r2_expr}",
            r2_expr,
            f"({r2_expr}**2)",
            "1",
        ]

        terms = []
        for c, basis in zip(coeffs, basis_exprs):
            s = fmt(c)
            if s is None:
                continue
            if basis == "1":
                terms.append(f"({s})")
            else:
                terms.append(f"({s})*{basis}")

        if not terms:
            expression = "0"
        else:
            expression = " + ".join(terms)

        return expression, predictions

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        if X.shape[1] != 2:
            raise ValueError("X must have exactly 2 columns representing x1 and x2.")

        result = None
        if _HAS_PYSR:
            result = self._fit_with_pysr(X, y)

        if result is None:
            expression, predictions = self._fit_fallback(X, y)
        else:
            expression, predictions = result

        return {
            "expression": expression,
            "predictions": predictions.tolist() if predictions is not None else None,
            "details": {}
        }