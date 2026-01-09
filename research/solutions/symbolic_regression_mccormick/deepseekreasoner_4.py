import numpy as np
from pysr import PySRRegressor
import sympy as sp

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=30,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=30,
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            early_stop_condition=(
                "stop_if(loss, complexity) = loss < 1e-12 && complexity < 20"
            ),
            constraints={
                "**": (9, 1),
                "/": (-1, 10)
            },
            maxdepth=10,
            warm_start=False,
            deterministic=True,
            model_selection="best",
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        if best_expr is None:
            expression = "x1 + x2"
            predictions = X[:, 0] + X[:, 1]
        else:
            simplified = sp.simplify(best_expr)
            expression = str(simplified)
            predictions = model.predict(X)
        
        complexity = len(str(expression))
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }