import numpy as np
import sympy as sp
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model_params = {
            "niterations": 60,
            "populations": 24,
            "population_size": 50,
            "procs": 8,
            "binary_operators": ["+", "-", "*", "/", "**"],
            "unary_operators": ["sin", "cos"],
            "constraints": {"pow": (15, 1)},
            "maxsize": 30,
            "random_state": 42,
            "verbosity": 0,
            "progress": False,
            "model_selection": "best",
            "nested_constraints": {"sin": {"cos": 0}, "cos": {"sin": 0}},
            "complexity_of_operators": {"sin": 2, "cos": 2, "pow": 2},
            "temp_equation_file": True,
            "elementwise_loss": "L2DistLoss()",
        }
        
        model = PySRRegressor(**model_params)

        try:
            model.fit(X, y, variable_names=["x1", "x2"])
        except Exception:
            return self._fallback_solution(X, y)

        if not hasattr(model, 'equations_') or model.equations_.empty:
            return self._fallback_solution(X, y)
        
        best_equation = model.get_best()
        sympy_expr = best_equation.sympy_format
        
        expression_str = str(sympy_expr)
        
        try:
            predictions = model.predict(X)
        except (RuntimeError, ValueError):
            predictions = None

        if predictions is not None:
            predictions = predictions.tolist()

        complexity = best_equation.complexity
        
        return {
            "expression": expression_str,
            "predictions": predictions,
            "details": {"complexity": complexity}
        }

    def _fallback_solution(self, X: np.ndarray, y: np.ndarray) -> dict:
        x1, x2 = X[:, 0], X[:, 1]
        A = np.column_stack([x1, x2, np.ones_like(x1)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c = coeffs
            expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}"
            predictions = A @ coeffs
        except np.linalg.LinAlgError:
            mean_y = np.mean(y)
            expression = str(mean_y)
            predictions = np.full_like(y, mean_y)

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }