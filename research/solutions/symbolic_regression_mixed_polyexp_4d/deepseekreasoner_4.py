import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=80,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=35,
            maxsize=35,
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            turbo=True,
            early_stop_condition=(
                "stop_if(loss, complexity) = loss < 1e-9 && complexity < 20"
            ),
            timeout_in_seconds=60,
            max_cycles=100,
            nested_constraints={
                "**": {"**": 0, "sin": 0, "cos": 0, "exp": 0, "log": 0},
                "exp": {"**": 0, "sin": 0, "cos": 0, "exp": 0, "log": 0},
                "log": {"**": 0, "sin": 0, "cos": 0, "exp": 0, "log": 0},
                "sin": {"**": 0, "sin": 0, "cos": 0, "exp": 0, "log": 0},
                "cos": {"**": 0, "sin": 0, "cos": 0, "exp": 0, "log": 0},
            },
            constraints={
                "**": (9, 9),
                "sin": 9,
                "cos": 9,
                "exp": 9,
                "log": 9,
            },
            batching=True,
            batch_size=100,
            warm_start=True,
            weight_optimize=0.02,
            update_test=False,
            precision=64,
        )
        
        model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
        
        best_expr = model.sympy()
        if best_expr is None:
            best_expr = "0"
        expression = str(best_expr)
        
        try:
            predictions = model.predict(X).tolist()
        except:
            x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            expr_eval = expression.replace("sin", "np.sin").replace("cos", "np.cos").replace("exp", "np.exp").replace("log", "np.log")
            predictions = eval(expr_eval).tolist()
        
        complexity = len(model.equations_) if hasattr(model, 'equations_') else 0
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {"complexity": complexity}
        }