import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        try:
            model = PySRRegressor(
                niterations=120,
                populations=32,
                population_size=50,
                binary_operators=["+", "-", "*", "/", "pow"],
                unary_operators=["sin", "cos", "exp", "log"],
                maxsize=35,
                complexity_of_operators={
                    "+": 2,
                    "-": 2,
                    "*": 2,
                    "/": 2,
                    "pow": 2,
                    "sin": 1,
                    "cos": 1,
                    "exp": 1,
                    "log": 1,
                },
                complexity_of_constants=1,
                complexity_of_variables=1,
                procs=8,
                random_state=42,
                verbosity=0,
                progress=False,
                loss="L2DistLoss()",
                nested_constraints={
                    "pow": {"x": 1e-6},
                    "log": {"x": 1e-6},
                    "div": {"y": 1e-6},
                },
                early_stop_condition="stop_if(loss, 1e-6)",
            )

            model.fit(X, y, variable_names=["x1", "x2"])

            if not hasattr(model, "equations_") or model.equations_.empty:
                raise RuntimeError("PySR found no equations.")

            best_equation = model.get_best()
            
            expression_sympy = best_equation.sympy_format
            expression_str = str(expression_sympy)

            predictions = model.predict(X)
            complexity = best_equation.complexity

            return {
                "expression": expression_str,
                "predictions": predictions.tolist(),
                "details": {"complexity": int(complexity)},
            }
        except Exception:
            x1, x2 = X[:, 0], X[:, 1]
            A = np.vstack([x1, x2, np.ones(len(x1))]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a, b, c = coeffs
                expression = f"{a:.8f}*x1 + {b:.8f}*x2 + {c:.8f}"
                predictions = a * x1 + b * x2 + c
                complexity = 8
            except np.linalg.LinAlgError:
                expression = "0.0"
                predictions = np.zeros_like(y)
                complexity = 0

            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {"complexity": complexity},
            }