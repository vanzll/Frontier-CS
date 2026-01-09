import numpy as np
from pysr import PySRRegressor
import sympy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
            temp_equation_file=True,
            temp_dir=".",
            verbosity=0,
            progress=False,
            random_state=42,
            deterministic=True,
            nested_constraints={"**": {"**": 1}},
            constraints={"**": (9, 1)},
            timeout_in_seconds=60 * 15,
            warm_start=True,
            loss="L2DistLoss()",
            multithreading=True,
            model_selection="accuracy",
        )
        
        try:
            model.fit(X, y, variable_names=["x1", "x2"])
            expression = str(model.sympy()).replace("**", "^").replace("^", "**")
            predictions = model.predict(X).tolist()
        except Exception:
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([x1, x2, x1**2, x2**2, x1*x2, 
                                 np.exp(-x1**2), np.exp(-x2**2),
                                 np.sin(x1), np.cos(x2), np.ones_like(x1)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            expression = f"{coeffs[0]:.6f}*x1 + {coeffs[1]:.6f}*x2 + {coeffs[2]:.6f}*x1**2 + {coeffs[3]:.6f}*x2**2 + {coeffs[4]:.6f}*x1*x2 + {coeffs[5]:.6f}*exp(-x1**2) + {coeffs[6]:.6f}*exp(-x2**2) + {coeffs[7]:.6f}*sin(x1) + {coeffs[8]:.6f}*cos(x2) + {coeffs[9]:.6f}"
            predictions = A @ coeffs
        
        try:
            expr = sympy.sympify(expression)
            constants = [float(c) for c in expr.atoms(sympy.Number)]
            binary_ops = len([a for a in expr.atoms(sympy.Add, sympy.Mul, sympy.Pow) if not isinstance(a, (sympy.Number, sympy.Symbol))])
            unary_ops = len([a for a in expr.atoms(sympy.sin, sympy.cos, sympy.exp, sympy.log) if not isinstance(a, (sympy.Number, sympy.Symbol))])
            complexity = 2 * binary_ops + unary_ops
        except:
            complexity = 0
        
        return {
            "expression": expression,
            "predictions": predictions,
            "details": {"complexity": complexity}
        }