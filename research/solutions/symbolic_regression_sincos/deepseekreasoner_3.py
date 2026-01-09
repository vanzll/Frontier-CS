import numpy as np
from pysr import PySRRegressor
import sympy
import warnings
warnings.filterwarnings('ignore')

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        # Configure PySR for efficient CPU-only execution
        model = PySRRegressor(
            niterations=30,  # Reduced for CPU efficiency
            populations=8,   # Match vCPU count
            population_size=20,
            maxsize=20,      # Limit complexity
            binary_operators=["+", "-", "*", "/", "**"],
            unary_operators=["sin", "cos", "exp", "log"],
            constraints={
                "**": (4, 1),  # Limit exponentiation complexity
                "log": 1,      # Limit log to positive domain
            },
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},  # Prevent nesting
                "cos": {"sin": 0, "cos": 0},
            },
            annealing=True,    # Better convergence
            turbo=True,        # Faster evaluation
            deterministic=True,
            random_state=42,
            verbosity=0,
            progress=False,
            model_selection="best",  # Select best MSE
            weight_optimize=0.01,    # Lightweight optimization
        )
        
        try:
            # Fit with variable names for proper substitution
            model.fit(X, y, variable_names=["x1", "x2"])
            
            # Get best expression
            best_expr = model.sympy()
            if best_expr is None:
                raise ValueError("No valid expression found")
            
            # Convert to string and clean
            expr_str = str(best_expr)
            
            # Replace sympy function calls with plain names
            expr_str = expr_str.replace("sin", "sin").replace("cos", "cos")
            expr_str = expr_str.replace("exp", "exp").replace("log", "log")
            
            # Ensure x1, x2 variable names
            expr_str = expr_str.replace("x0", "x1").replace("x1", "x1").replace("x2", "x2")
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Compute complexity (binary ops * 2 + unary ops)
            expr_sym = sympy.sympify(expr_str)
            binary_count = sum(1 for node in sympy.preorder_traversal(expr_sym) 
                              if node.is_Add or node.is_Mul or node.is_Pow)
            unary_count = sum(1 for node in sympy.preorder_traversal(expr_sym)
                             if node.is_Function)
            complexity = 2 * binary_count + unary_count
            
            return {
                "expression": expr_str,
                "predictions": predictions.tolist(),
                "details": {"complexity": complexity}
            }
            
        except Exception as e:
            # Fallback to simple model if PySR fails
            x1, x2 = X[:, 0], X[:, 1]
            
            # Try trigonometric combinations common in sin/cos problems
            features = np.column_stack([
                np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2),
                np.sin(x1 + x2), np.cos(x1 + x2),
                x1, x2, np.ones_like(x1)
            ])
            
            coeffs, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
            
            # Build expression string
            terms = []
            coeff_names = [
                "sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)",
                "sin(x1 + x2)", "cos(x1 + x2)",
                "x1", "x2", ""
            ]
            
            for c, name in zip(coeffs, coeff_names):
                if abs(c) > 1e-10:
                    if name == "":
                        terms.append(f"{c:.6f}")
                    else:
                        terms.append(f"{c:.6f}*{name}")
            
            expr_str = " + ".join(terms).replace("+ -", "- ")
            
            # Compute predictions
            predictions = features @ coeffs
            
            # Estimate complexity
            complexity = 2 * len(terms) + 4  # Approximate
            
            return {
                "expression": expr_str,
                "predictions": predictions.tolist(),
                "details": {"complexity": complexity}
            }