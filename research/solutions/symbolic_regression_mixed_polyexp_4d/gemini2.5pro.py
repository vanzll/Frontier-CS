import numpy as np
from pysr import PySRRegressor
import sympy
import os
import tempfile

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Learns a symbolic expression for the given data using PySR.
        """
        # Define operators and their complexities according to the scoring rule
        binary_operators = ["+", "-", "*", "/", "**"]
        unary_operators = ["sin", "cos", "exp", "log"]

        # Scoring rule: C = 2 * (#binary ops) + (#unary ops)
        # Align PySR's internal complexity calculation with the scoring metric.
        complexity_of_operators = {op: 2 for op in binary_operators}
        complexity_of_operators.update({op: 1 for op in unary_operators})
        
        # Use a temporary directory for PySR's equation files for robustness
        with tempfile.TemporaryDirectory() as tempdir:
            temp_equation_file = os.path.join(tempdir, "equations.csv")

            model = PySRRegressor(
                # Search Configuration
                niterations=100,        # Increased iterations for complex 4D problem
                populations=24,         # Leverage 8 vCPUs (3*8) for diverse search
                population_size=50,     # Larger populations for more diversity

                # Operators and Complexity
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                complexity_of_operators=complexity_of_operators,
                
                # Constraints
                maxsize=35,             # Allow slightly more complex expressions for 4D

                # Time and Performance Management
                early_stop_condition=f"stop_if(loss, 1e-5)", # Stop if a good fit is found
                timeout_in_seconds=580, # Safety net to ensure timely completion

                # Parallelism and Environment
                procs=8,                # Utilize all 8 vCPUs
                multithreading=False,   # Use multiprocessing for better CPU-bound performance
                
                # Other Settings
                random_state=42,
                verbosity=0,
                progress=False,
                temp_equation_file=temp_equation_file,
            )

            model.fit(X, y, variable_names=["x1", "x2", "x3", "x4"])
            
            if not hasattr(model, 'equations_') or len(model.equations_) == 0:
                # Fallback to a simple linear model if PySR fails or times out early
                x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
                A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    a, b, c, d, e = coeffs
                    expression = f"{a:.6f}*x1 + {b:.6f}*x2 + {c:.6f}*x3 + {d:.6f}*x4 + {e:.6f}"
                    predictions = a * x1 + b * x2 + c * x3 + d * x4 + e
                    # Complexity: 4 mults + 4 adds = 8 binary ops -> C = 2 * 8 = 16
                    complexity = 16 
                except np.linalg.LinAlgError:
                    expression = "0.0"
                    predictions = np.zeros_like(y)
                    complexity = 0
                
                return {
                    "expression": expression,
                    "predictions": predictions.tolist(),
                    "details": {"complexity": complexity}
                }

            # Process Results from PySR
            best_equation = model.get_best()
            
            # Convert the best sympy expression to a string
            expression = str(model.sympy())
            
            # Generate predictions using the internal PySR model for best accuracy
            predictions = model.predict(X)
            
            complexity = best_equation.complexity
            
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {"complexity": int(complexity)}
            }