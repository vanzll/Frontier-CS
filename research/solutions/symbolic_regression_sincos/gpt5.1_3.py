import numpy as np
from typing import Optional, Tuple

try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None


class Solution:
    def __init__(self, **kwargs):
        pass

    def _fit_simple_trig_models(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[str], Optional[np.ndarray], float]:
        x1 = X[:, 0]
        x2 = X[:, 1]

        candidates = [
            ("sin(x1) + cos(x2)", lambda a, b: np.sin(a) + np.cos(b)),
            ("sin(x1) + sin(x2)", lambda a, b: np.sin(a) + np.sin(b)),
            ("cos(x1) + cos(x2)", lambda a, b: np.cos(a) + np.cos(b)),
            ("sin(x1) * cos(x2)", lambda a, b: np.sin(a) * np.cos(b)),
            ("cos(x1) * sin(x2)", lambda a, b: np.cos(a) * np.sin(b)),
            ("sin(x1) * sin(x2)", lambda a, b: np.sin(a) * np.sin(b)),
            ("cos(x1) * cos(x2)", lambda a, b: np.cos(a) * np.cos(b)),
            ("sin(x1 + x2)", lambda a, b: np.sin(a + b)),
            ("cos(x1 + x2)", lambda a, b: np.cos(a + b)),
            ("sin(x1 - x2)", lambda a, b: np.sin(a - b)),
            ("cos(x1 - x2)", lambda a, b: np.cos(a - b)),
            ("sin(x1 * x2)", lambda a, b: np.sin(a * b)),
            ("cos(x1 * x2)", lambda a, b: np.cos(a * b)),
            ("sin(x1) + cos(x1)", lambda a, b: np.sin(a) + np.cos(a)),
            ("sin(x2) + cos(x2)", lambda a, b: np.sin(b) + np.cos(b)),
            ("sin(x1) - cos(x2)", lambda a, b: np.sin(a) - np.cos(b)),
            ("sin(x1) - sin(x2)", lambda a, b: np.sin(a) - np.sin(b)),
            ("cos(x1) - cos(x2)", lambda a, b: np.cos(a) - np.cos(b)),
            ("sin(x1) + cos(x2) + sin(x2)", lambda a, b: np.sin(a) + np.cos(b) + np.sin(b)),
            ("sin(x1) + cos(x2) + cos(x1)", lambda a, b: np.sin(a) + np.cos(b) + np.cos(a)),
        ]

        best_expr = None
        best_pred = None
        best_mse = np.inf

        for expr_str, func in candidates:
            try:
                preds = func(x1, x2)
                if preds.shape != y.shape:
                    continue
                residuals = preds - y
                mse = float(np.mean(residuals * residuals))
                if np.isnan(mse) or np.isinf(mse):
                    continue
                if mse < best_mse:
                    best_mse = mse
                    best_expr = expr_str
                    best_pred = preds
            except Exception:
                continue

        return best_expr, best_pred, best_mse

    def _py_sr_symbolic_regression(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        if PySRRegressor is None:
            return None, None

        try:
            model = PySRRegressor(
                niterations=60,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos"],
                populations=15,
                population_size=33,
                maxsize=15,
                verbosity=0,
                progress=False,
                random_state=42,
            )
            model.fit(X, y, variable_names=["x1", "x2"])

            best_expr = model.sympy()
            try:
                # Handle case where sympy() returns a list/array
                if isinstance(best_expr, (list, tuple, np.ndarray)):
                    best_expr = best_expr[0]
            except Exception:
                pass
            expression = str(best_expr)

            predictions = model.predict(X)
            predictions = np.asarray(predictions, dtype=float)
            return expression, predictions
        except Exception:
            return None, None

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2 or X.shape[1] < 2:
            expression = "x1 + x2"
            predictions = np.sum(X[:, :2], axis=1) if X.size > 0 else np.array([], dtype=float)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {},
            }

        n_samples = X.shape[0]
        if n_samples == 0:
            expression = "sin(x1) + cos(x2)"
            predictions = np.array([], dtype=float)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {},
            }

        best_expr, best_pred, best_mse = self._fit_simple_trig_models(X, y)

        var_y = float(np.var(y))
        if not np.isfinite(var_y) or var_y <= 0.0:
            var_y = 1.0
        tolerance = max(1e-10, 1e-4 * var_y)

        if best_expr is not None and np.isfinite(best_mse) and best_mse <= tolerance:
            expression = best_expr
            predictions = best_pred
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {},
            }

        expr_sr, preds_sr = self._py_sr_symbolic_regression(X, y)

        if expr_sr is not None and preds_sr is not None and preds_sr.shape == y.shape:
            if np.all(np.isfinite(preds_sr)):
                expression = expr_sr
                predictions = preds_sr
            else:
                expression = best_expr if best_expr is not None else "x1 + x2"
                if best_pred is not None:
                    predictions = best_pred
                else:
                    predictions = X[:, 0] + X[:, 1]
        else:
            expression = best_expr if best_expr is not None else "x1 + x2"
            if best_pred is not None:
                predictions = best_pred
            else:
                predictions = X[:, 0] + X[:, 1]

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {},
        }