import numpy as np
from pysr import PySRRegressor

class Solution:
    def __init__(self, **kwargs):
        """
        Constructor for the solution class.
        """
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Implements the solution for the symbolic regression problem.

        This method uses PySR (Symbolic Regression via Evolutionary Search)
        to find a closed-form symbolic expression that predicts the target `y`
        from the input features `X`.

        The parameters for PySRRegressor are tuned based on the problem
        description and the evaluation environment:
        - The dataset name "SinCos" strongly suggests the underlying function
          involves trigonometric operations. Therefore, the search space for
          unary operators is limited to 'sin' and 'cos' to guide the search
          effectively and speed up convergence.
        - `niterations`, `populations`, and `population_size` are increased
          from the baseline example to allow for a more thorough search,
          leveraging the 8 vCPUs available in the environment.
        - `maxsize` is kept relatively small to encourage simpler, more
          generalizable expressions, which aligns with the scoring metric that
          penalizes high complexity.
        - `turbo=True` is used to accelerate the expression evaluation, which is
          often a bottleneck.

        Args:
            X: numpy.ndarray of shape (n, 2) containing feature values.
            y: numpy.ndarray of shape (n,) containing target values.

        Returns:
            A dictionary containing the symbolic expression, predictions,
            and model details.
        """
        model = PySRRegressor(
            niterations=60,
            populations=24,
            population_size=50,
            unary_operators=["sin", "cos"],
            binary_operators=["+", "-", "*", "/"],
            maxsize=20,
            verbosity=0,
            progress=False,
            random_state=42,
            turbo=True,
            model_selection='best',
        )

        model.fit(X, y, variable_names=["x1", "x2"])

        if len(model.equations) == 0:
            # Fallback solution if PySR fails to find any expression
            expression = "0.0"
            predictions = np.zeros_like(y)
            complexity = 0
        else:
            # Extract the best expression and its properties from the fitted model
            best_sympy_expr = model.sympy()
            expression = str(best_sympy_expr)
            predictions = model.predict(X)
            complexity = model.equations_.iloc[-1]['complexity']

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {
                "complexity": int(complexity)
            }
        }