import numpy as np

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
except Exception:
    PySRRegressor = None
    _HAS_PYSR = False


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        if _HAS_PYSR:
            try:
                return self._solve_pysr(X, y)
            except Exception:
                pass
        return self._solve_fallback(X, y)

    def _solve_pysr(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        model = PySRRegressor(
            niterations=60,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=20,
            population_size=40,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
        )
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])

        sympy_expr = model.sympy()
        if isinstance(sympy_expr, (list, tuple)):
            if len(sympy_expr) == 0:
                expression = "x1 + x2 + x3 + x4"
            else:
                expression = str(sympy_expr[0])
        else:
            expression = str(sympy_expr)

        predictions = model.predict(X)

        details = {}
        try:
            eqs = getattr(model, "equations_", None)
            if eqs is not None and hasattr(eqs, "iloc") and len(eqs) > 0:
                best_row = eqs.iloc[0]
                if "complexity" in best_row:
                    complexity_val = best_row["complexity"]
                    try:
                        details["complexity"] = int(complexity_val)
                    except Exception:
                        pass
        except Exception:
            pass

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }

    def _solve_fallback(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[1] != 4:
            raise ValueError("Expected X with 4 columns.")

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        feats_arrays = []
        feat_strs = []

        def add_feature(arr, s):
            feats_arrays.append(arr)
            feat_strs.append(s)

        add_feature(np.ones_like(y), "1.0")
        add_feature(x1, "x1")
        add_feature(x2, "x2")
        add_feature(x3, "x3")
        add_feature(x4, "x4")

        x1_2 = x1 ** 2
        x2_2 = x2 ** 2
        x3_2 = x3 ** 2
        x4_2 = x4 ** 2
        add_feature(x1_2, "x1**2")
        add_feature(x2_2, "x2**2")
        add_feature(x3_2, "x3**2")
        add_feature(x4_2, "x4**2")

        add_feature(x1 * x2, "x1*x2")
        add_feature(x1 * x3, "x1*x3")
        add_feature(x1 * x4, "x1*x4")
        add_feature(x2 * x3, "x2*x3")
        add_feature(x2 * x4, "x2*x4")
        add_feature(x3 * x4, "x3*x4")

        add_feature(x1 ** 3, "x1**3")
        add_feature(x2 ** 3, "x2**3")
        add_feature(x3 ** 3, "x3**3")
        add_feature(x4 ** 3, "x4**3")

        r2 = x1_2 + x2_2 + x3_2 + x4_2
        r2_str = "x1**2 + x2**2 + x3**2 + x4**2"
        add_feature(r2, f"({r2_str})")

        for c in (0.1, 0.5, 1.0, 2.0):
            exp_term = np.exp(-c * r2)
            exp_str = f"exp(-{c}*({r2_str}))"
            add_feature(exp_term, exp_str)
            add_feature(x1 * exp_term, f"x1*{exp_str}")
            add_feature(x2 * exp_term, f"x2*{exp_str}")
            add_feature(x3 * exp_term, f"x3*{exp_str}")
            add_feature(x4 * exp_term, f"x4*{exp_str}")

        A = np.column_stack(feats_arrays)
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        terms = []
        y_pred = np.zeros_like(y, dtype=float)
        tol = 1e-6
        for coef, arr, s in zip(coeffs, feats_arrays, feat_strs):
            if abs(coef) < tol:
                continue
            coef_str = f"{coef:.12g}"
            if s == "1.0":
                term_expr = coef_str
            else:
                term_expr = f"{coef_str}*({s})"
            terms.append(term_expr)
            y_pred += coef * arr

        if not terms:
            expression = "0"
            y_pred = np.zeros_like(y, dtype=float)
        else:
            expression = " + ".join(terms)

        return {
            "expression": expression,
            "predictions": y_pred.tolist(),
            "details": {},
        }