import numpy as np
from pysr import PySRRegressor
import sympy
from typing import Dict, Any

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=8,
            population_size=30,
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            ncycles_per_iteration=1000,
            early_stop_condition=1e-8,
            loss="L2DistLoss()",
            complexity_of_operators={"**": 3, "/": 2},
            constraints={"**": (5, 1)},
            nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
            deterministic=True,
            weight_optimize=0.02,
            warm_start=True,
            update=False,
            use_frequency=True,
            use_frequency_in_tournament=True,
            adaptive_parsimony_scaling=20.0,
            parsimony=1e-4,
            annealing=True,
            temperature=2.0,
            temp_annealing_rate=0.99,
            temp_decay_exponent=2,
        )
        
        model.fit(X, y, variable_names=["x1", "x2"])
        
        best_expr = model.sympy()
        expression = str(best_expr)
        
        try:
            predictions = model.predict(X).tolist()
        except:
            x1 = X[:, 0]
            x2 = X[:, 1]
            predictions = [float(best_expr.subs({"x1": xi1, "x2": xi2}))
                          for xi1, xi2 in zip(x1, x2)]
        
        complexity = None
        if hasattr(model, 'get_best') and model.equations_ is not None:
            best_eq = model.equations_.iloc[0]
            if 'complexity' in best_eq:
                complexity = int(best_eq['complexity'])
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {"complexity": complexity} if complexity is not None else {}
        }