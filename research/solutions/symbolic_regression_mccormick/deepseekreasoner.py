import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=25,
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            early_stop_condition=1e-8,
            timeout_in_seconds=300,
            max_evals=100000,
            constraints={
                "**": (9, 1),
                "log": 1,
                "exp": 1
            },
            loss="L2DistLoss()",
            model_selection="best",
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        if best_expr is None:
            expression = "0"
            predictions = np.zeros_like(y)
        else:
            expression = str(best_expr)
            predictions = model.predict(X)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist() if predictions is not None else None,
            "details": {}
        }