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
            populations=8,
            population_size=40,
            maxsize=30,
            max_depth=8,
            verbosity=0,
            progress=False,
            random_state=42,
            multithreading=True,
            ncyclesperiteration=700,
            early_stop_condition=(
                "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
            ),
            loss="loss(x, y) = (x - y)^2",
            constraints={
                "**": (9, 1),
                "log": 9,
                "exp": 9,
                "sin": 9,
                "cos": 9,
            },
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr).replace("**", "^")
        
        binary_ops = expression.count("+") + expression.count("-") + \
                    expression.count("*") + expression.count("/") + \
                    expression.count("^")
        unary_ops = expression.count("sin") + expression.count("cos") + \
                   expression.count("exp") + expression.count("log")
        complexity = 2 * binary_ops + unary_ops
        
        try:
            predictions = model.predict(X).tolist()
        except:
            predictions = None
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {"complexity": complexity}
        }