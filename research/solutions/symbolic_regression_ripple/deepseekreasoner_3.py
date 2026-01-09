import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=20,
            population_size=50,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            ncyclesperiteration=700,
            early_stop_condition=(
                "stop_if(loss, complexity) = (loss < 1e-9) || (complexity > 30)"
            ),
            temp_annealing=True,
            temp_decay=0.99,
            weight_optimize=0.02,
            optimize_naive=True,
            deterministic=False,
            max_depth=12,
            use_frequency=False,
            constraints={
                "**": (9, 9),
                "log": (9, 9),
                "exp": (9, 9)
            },
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr).replace("x0", "x1").replace("x1", "x2")
        
        predictions = model.predict(X)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }