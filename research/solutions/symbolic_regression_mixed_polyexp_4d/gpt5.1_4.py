import numpy as np
try:
    from pysr import PySRRegressor
    _PYSR_AVAILABLE = True
except Exception:
    _PYSR_AVAILABLE = False


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit_linear_baseline(self, X, y):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        A = np.column_stack((x1, x2, x3, x4, np.ones_like(x1)))
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c, d, e = coeffs
        expression = (
            f"{a:.12g}*x1 + "
            f"{b:.12g}*x2 + "
            f"{c:.12g}*x3 + "
            f"{d:.12g}*x4 + "
            f"{e:.12g}"
        )
        predictions = A @ coeffs
        return expression, predictions

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[1] != 4:
            # Fallback to using the first 4 features if more are provided
            # (benchmark spec says exactly 4, so this is just a safeguard).
            if X.shape[1] > 4:
                X = X[:, :4]
            else:
                # If fewer than 4 features, pad with zeros to reach 4.
                n_samples = X.shape[0]
                X_padded = np.zeros((n_samples, 4), dtype=float)
                X_padded[:, :X.shape[1]] = X
                X = X_padded

        if not _PYSR_AVAILABLE:
            expr, preds = self._fit_linear_baseline(X, y)
            return {"expression": expr, "predictions": preds.tolist(), "details": {}}

        try:
            n_samples = X.shape[0]
            max_samples = 6000
            if n_samples > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_samples, size=max_samples, replace=False)
                X_train = X[idx]
                y_train = y[idx]
            else:
                X_train = X
                y_train = y

            model = PySRRegressor(
                niterations=60,
                ncycles_per_iteration=60,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp"],
                populations=20,
                population_size=25,
                maxsize=30,
                verbosity=0,
                progress=False,
                random_state=42,
            )

            model.fit(X_train, y_train, variable_names=["x1", "x2", "x3", "x4"])

            best_expr = model.sympy()
            expression = str(best_expr)

            try:
                predictions = model.predict(X)
                predictions_list = predictions.tolist()
            except Exception:
                predictions_list = None

            details = {}
            try:
                eqns = model.equations_
                if eqns is not None and len(eqns) > 0:
                    best_row = eqns.iloc[0]
                    if "complexity" in best_row.index:
                        details["complexity"] = int(best_row["complexity"])
                    if "loss" in best_row.index:
                        details["loss"] = float(best_row["loss"])
            except Exception:
                pass

            return {
                "expression": expression,
                "predictions": predictions_list,
                "details": details,
            }
        except Exception:
            expr, preds = self._fit_linear_baseline(X, y)
            return {"expression": expr, "predictions": preds.tolist(), "details": {}}