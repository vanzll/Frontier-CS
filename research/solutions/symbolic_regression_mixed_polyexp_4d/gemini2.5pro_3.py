import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Args:
            X: Feature matrix of shape (n, 4)
            y: Target values of shape (n,)

        Returns:
            dict with keys:
              - "expression": str, a Python-evaluable expression using x1, x2, x3, x4
              - "predictions": list/array of length n (optional)
              - "details": dict with optional "complexity" int
        """
        model = PySRRegressor(
            niterations=100,
            populations=30,
            population_size=50,
            maxsize=35,
            binary_operators=["+", "-", "*", "/", "pow"],
            unary_operators=["sin", "cos", "exp", "log"],
            procs=8,
            precision=64,
            parsimony=0.001,
            early_stop_condition="loss < 1e-6",
            random_state=42,
            verbosity=0,
            progress=False,
        )

        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])

        expression = "0.0"
        predictions = np.zeros_like(y)
        complexity = 0

        if hasattr(model, 'equations_') and model.equations_ is not None and not model.equations_.empty:
            best_idx = model.equation_
            
            best_expr_sympy = model.sympy(index=best_idx)
            expression = str(best_expr_sympy)
            
            predictions = model.predict(X, index=best_idx)
            
            complexity = model.equations_.iloc[best_idx]['complexity']

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {
                "complexity": int(complexity)
            }
        }