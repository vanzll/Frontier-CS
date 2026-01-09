import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Args:
            X: Feature matrix of shape (n, 2)
            y: Target values of shape (n,)

        Returns:
            dict with keys:
              - "expression": str, a Python-evaluable expression using x1, x2
              - "predictions": list/array of length n (optional)
              - "details": dict with optional "complexity" int
        """
        model = PySRRegressor(
            niterations=60,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=25,
            maxsize=30,
            maxdepth=10,
            verbosity=0,
            progress=False,
            random_state=42,
            multithreading=True,
            ncycles_per_iteration=1000,
            early_stop_condition="stop_if(loss, 1e-6)",
            weight_optimize=0.02,
            turbo=True,
            optimizer_algorithm="BFGS",
            optimizer_iterations=10,
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr).replace(" ", "").replace("**", "^")
        expression = expression.replace("^", "**")
        
        predictions = model.predict(X)
        
        complexity = 0
        if hasattr(model, 'equations_') and model.equations_ is not None:
            best_eq = model.equations_.iloc[model.equations_['loss'].idxmin()]
            complexity = best_eq.get('complexity', 0)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": int(complexity)}
        }