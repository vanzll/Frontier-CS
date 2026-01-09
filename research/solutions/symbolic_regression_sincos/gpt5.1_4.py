import numpy as np
try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None


class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit_trig_leastsq(self, X: np.ndarray, y: np.ndarray):
        x1 = X[:, 0]
        x2 = X[:, 1]

        sin_x1 = np.sin(x1)
        cos_x1 = np.cos(x1)
        sin_x2 = np.sin(x2)
        cos_x2 = np.cos(x2)

        sin_x1_plus_x2 = np.sin(x1 + x2)
        cos_x1_plus_x2 = np.cos(x1 + x2)
        sin_x1_minus_x2 = np.sin(x1 - x2)
        cos_x1_minus_x2 = np.cos(x1 - x2)

        sin_x1_cos_x2 = sin_x1 * cos_x2
        cos_x1_sin_x2 = cos_x1 * sin_x2
        sin_x1_sin_x2 = sin_x1 * sin_x2
        cos_x1_cos_x2 = cos_x1 * cos_x2

        ones = np.ones_like(x1)

        A = np.column_stack(
            [
                ones,
                sin_x1,
                cos_x1,
                sin_x2,
                cos_x2,
                sin_x1_plus_x2,
                cos_x1_plus_x2,
                sin_x1_minus_x2,
                cos_x1_minus_x2,
                sin_x1_cos_x2,
                cos_x1_sin_x2,
                sin_x1_sin_x2,
                cos_x1_cos_x2,
            ]
        )

        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        basis_exprs = [
            "1",
            "sin(x1)",
            "cos(x1)",
            "sin(x2)",
            "cos(x2)",
            "sin(x1 + x2)",
            "cos(x1 + x2)",
            "sin(x1 - x2)",
            "cos(x1 - x2)",
            "sin(x1)*cos(x2)",
            "cos(x1)*sin(x2)",
            "sin(x1)*sin(x2)",
            "cos(x1)*cos(x2)",
        ]

        expression_terms = []
        threshold = 1e-4
        for coef, term in zip(coeffs, basis_exprs):
            if abs(coef) < threshold:
                continue
            if term == "1":
                expr_part = f"{coef:.10g}"
            else:
                if abs(coef - 1.0) < 1e-8:
                    expr_part = term
                elif abs(coef + 1.0) < 1e-8:
                    expr_part = f"-{term}"
                else:
                    expr_part = f"{coef:.10g}*{term}"
            expression_terms.append(expr_part)

        if not expression_terms:
            expression = "0"
        else:
            expression = " + ".join(expression_terms)
            expression = expression.replace("+ -", "- ")

        y_pred = A @ coeffs
        return expression, y_pred

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        expression = None
        predictions = None
        details = {}

        if PySRRegressor is not None:
            try:
                niterations = self.kwargs.get("niterations", 40)
                populations = self.kwargs.get("populations", 15)
                population_size = self.kwargs.get("population_size", 33)
                maxsize = self.kwargs.get("maxsize", 20)

                model = PySRRegressor(
                    niterations=niterations,
                    binary_operators=["+", "-", "*"],
                    unary_operators=["sin", "cos"],
                    populations=populations,
                    population_size=population_size,
                    maxsize=maxsize,
                    verbosity=0,
                    progress=False,
                    random_state=42,
                )

                model.fit(X, y, variable_names=["x1", "x2"])
                best_expr = model.sympy()
                expression = str(best_expr)

                try:
                    predictions = model.predict(X)
                except Exception:
                    predictions = None
            except Exception:
                expression = None
                predictions = None

        if expression is None:
            expression, predictions = self._fit_trig_leastsq(X, y)

        if predictions is not None:
            predictions_list = np.asarray(predictions, dtype=float).tolist()
        else:
            predictions_list = None

        return {
            "expression": expression,
            "predictions": predictions_list,
            "details": details,
        }