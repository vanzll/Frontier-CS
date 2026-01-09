import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=200,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=10,
            population_size=50,
            maxsize=40,
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            loss="loss(x, y) = (x - y)^2",
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-8 && complexity < 15",
            timeout_in_seconds=60,
            warm_start=False,
            model_selection="accuracy",
        )
        
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        
        if hasattr(model, "sympy") and model.sympy() is not None:
            try:
                best_expr = str(model.sympy())
                predictions = model.predict(X)
            except:
                best_expr = "x1"
                predictions = X[:, 0]
        else:
            best_expr = "x1"
            predictions = X[:, 0]
        
        complexity = 0
        if hasattr(model, "get_best") and model.get_best() is not None:
            best_eq = model.get_best()
            if hasattr(best_eq, 'complexity'):
                complexity = best_eq.complexity
        
        return {
            "expression": best_expr,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }