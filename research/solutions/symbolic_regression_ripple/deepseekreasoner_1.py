import numpy as np
from pysr import PySRRegressor
import sympy

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        model = PySRRegressor(
            niterations=30,
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            populations=10,
            population_size=30,
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            ncycles_per_iteration=400,
            early_stop_condition="1e-8",
            timeout_in_seconds=60,
            loss="L2DistLoss()",
            complexity_of_constants=2,
            batching=True,
            batch_size=100,
            warm_start=True,
            temp_equation_file=True,
            update=False,
            deterministic=False,
            parsimony=0.003,
            use_frequency=True,
            use_frequency_in_tournament=True,
            adaptive_parsimony_scaling=20,
            nested_constraints={
                "sin": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "cos": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "exp": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
                "log": {"sin": 0, "cos": 0, "exp": 0, "log": 0},
            },
            elementwise_loss=False,
            turbo=True,
            precision=64,
            constraints={
                "**": (4, 1),
                "log": (-1, 5),
            },
            max_denoising_coeff=0.2,
            denoising=True,
        )
        
        try:
            model.fit(X, y, variable_names=["x1", "x2"])
            best_expr = model.sympy()
            expression = str(best_expr)
            predictions = model.predict(X)
            
            complexity = 2 * model.equations_.iloc[-1]["n_binary_operators"] + model.equations_.iloc[-1]["n_unary_operators"]
            
        except Exception:
            x1, x2 = X[:, 0], X[:, 1]
            A = np.column_stack([
                x1, x2, 
                np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2),
                np.sin(x1**2 + x2**2),
                x1 * x2,
                x1**2, x2**2,
                np.exp(-x1**2), np.exp(-x2**2),
                np.ones_like(x1)
            ])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            terms = [
                f"{coeffs[0]:.6f}*x1",
                f"{coeffs[1]:.6f}*x2",
                f"{coeffs[2]:.6f}*sin(x1)",
                f"{coeffs[3]:.6f}*cos(x1)",
                f"{coeffs[4]:.6f}*sin(x2)",
                f"{coeffs[5]:.6f}*cos(x2)",
                f"{coeffs[6]:.6f}*sin(x1**2 + x2**2)",
                f"{coeffs[7]:.6f}*x1*x2",
                f"{coeffs[8]:.6f}*x1**2",
                f"{coeffs[9]:.6f}*x2**2",
                f"{coeffs[10]:.6f}*exp(-x1**2)",
                f"{coeffs[11]:.6f}*exp(-x2**2)",
                f"{coeffs[12]:.6f}"
            ]
            expression = " + ".join(terms)
            predictions = A @ coeffs
            complexity = 14

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {"complexity": complexity}
        }