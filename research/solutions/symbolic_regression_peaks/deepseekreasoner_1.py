import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=60,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=20,
            population_size=35,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            ncycles_per_iteration=700,
            temp_annealing_rate=0.98,
            temperature=0.5,
            constraints={
                "**": (5, 1),
                "log": (-1, 5)
            },
            loss="L2DistLoss()",
            complexity_of_operators={
                "sin": 3, "cos": 3, "exp": 3, "log": 3,
                "**": 3, "+": 1, "-": 1, "*": 2, "/": 2
            }
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        if best_expr is None:
            expression = "0.0"
            predictions = np.zeros_like(y)
        else:
            expression = str(best_expr).replace("**", "^").replace("^", "**")
            predictions = model.predict(X)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "details": {}
        }