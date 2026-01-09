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
            population_size=50,
            maxsize=30,
            verbosity=0,
            progress=False,
            random_state=42,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-9 && complexity < 20",
            warm_start=True,
            max_evals=1000000,
            ncycles_per_iteration=500,
            temp_anneal_rate=0.9,
            temp_decay_rate=0.01,
            temp_init=2.0,
            model_selection="accuracy",
            nested_constraints={"exp": ["exp"], "log": ["log"]}
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        if best_expr is None:
            expression = "0"
            predictions = np.zeros_like(y)
        else:
            expression = str(best_expr).replace("**", "^").replace("^", "**")
            predictions = model.predict(X)
        
        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": model.equations_.iloc[-1]["complexity"] if len(model.equations_) > 0 else 0}
        }