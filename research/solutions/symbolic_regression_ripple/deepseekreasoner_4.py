import numpy as np
from pysr import PySRRegressor

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
            nested_constraints={"sin": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                              "cos": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                              "exp": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                              "log": {"sin": 0, "cos": 0, "exp": 0, "log": 0}},
            deterministic=True,
            early_stop_condition=("stop_if(loss, complexity) = (loss < 1e-9) && (complexity < 10)", 5),
            loss="loss(x, y) = (x - y)^2",
            constraints={"**": (9, 1)},
            batching=True,
            batch_size=50,
            warm_start=True,
            turbo=True,
            precision=64
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
            "predictions": predictions.tolist(),
            "details": {"complexity": len(str(best_expr)) if best_expr else 0}
        }