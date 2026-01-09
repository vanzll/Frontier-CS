import numpy as np
import sympy as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
import warnings

class Solution:
    def __init__(self, **kwargs):
        pass
    
    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            x1 = X[:, 0]
            x2 = X[:, 1]
            
            # Try multiple expression forms with different complexities
            candidates = []
            
            # Base model: quadratic polynomial
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            model = Ridge(alpha=0.1).fit(X_poly, y)
            poly_pred = model.predict(X_poly)
            poly_mse = np.mean((y - poly_pred) ** 2)
            
            # Generate symbolic expression from polynomial
            coeffs = model.coef_
            intercept = model.intercept_
            
            terms = []
            for i, (name, coef) in enumerate(zip(poly.get_feature_names_out(['x1', 'x2']), coeffs)):
                if abs(coef) > 1e-10:
                    if name == '1':
                        terms.append(f"{coef:.6f}")
                    else:
                        terms.append(f"({coef:.6f})*{name}")
            if abs(intercept) > 1e-10 and '1' not in poly.get_feature_names_out(['x1', 'x2']):
                terms.append(f"{intercept:.6f}")
            
            poly_expr = " + ".join(terms).replace(" ", "").replace("x1^2", "x1**2").replace("x2^2", "x2**2")
            candidates.append((poly_mse, poly_expr, poly_pred, 2))
            
            # Try trigonometric forms
            for trig_form in [
                lambda a, b, c, d: a*np.sin(b*x1 + c*x2 + d),
                lambda a, b, c, d: a*np.cos(b*x1 + c*x2 + d),
                lambda a, b, c, d, e: a*np.sin(b*x1)*np.cos(c*x2) + d,
                lambda a, b, c, d, e: a*np.sin(b*x1**2 + c*x2**2) + d,
            ]:
                try:
                    # Parameter fitting
                    if trig_form.__code__.co_argcount == 4:
                        def objective(params):
                            a, b, c, d = params
                            pred = trig_form(a, b, c, d)
                            return np.mean((y - pred) ** 2)
                        
                        res = minimize(objective, [1.0, 1.0, 1.0, 0.0], 
                                     method='L-BFGS-B', bounds=[(0.1, 5), (-5, 5), (-5, 5), (-5, 5)])
                        if res.success:
                            a, b, c, d = res.x
                            pred = trig_form(a, b, c, d)
                            mse = np.mean((y - pred) ** 2)
                            expr = f"{a:.6f}*sin({b:.6f}*x1 + {c:.6f}*x2 + {d:.6f})"
                            candidates.append((mse, expr, pred, 2))
                    elif trig_form.__code__.co_argcount == 5:
                        def objective(params):
                            a, b, c, d, e = params
                            pred = trig_form(a, b, c, d, e)
                            return np.mean((y - pred) ** 2)
                        
                        res = minimize(objective, [1.0, 1.0, 1.0, 0.0, 0.0],
                                     method='L-BFGS-B', bounds=[(0.1, 5), (-3, 3), (-3, 3), (-5, 5), (-5, 5)])
                        if res.success:
                            a, b, c, d, e = res.x
                            pred = trig_form(a, b, c, d, e)
                            mse = np.mean((y - pred) ** 2)
                            expr = f"{a:.6f}*sin({b:.6f}*x1**2 + {c:.6f}*x2**2) + {d:.6f}"
                            candidates.append((mse, expr, pred, 3))
                except:
                    continue
            
            # Try radial basis form (for ripple patterns)
            def radial_form(x1, x2, a, b, c, d, e, f):
                r = np.sqrt(x1**2 + x2**2)
                return a * np.sin(b * r + c) * np.exp(-d * r**2) + e * x1 + f * x2
            
            def objective_radial(params):
                a, b, c, d, e, f = params
                pred = radial_form(x1, x2, a, b, c, d, e, f)
                return np.mean((y - pred) ** 2)
            
            res = minimize(objective_radial, [1.0, 5.0, 0.0, 0.1, 0.0, 0.0],
                         method='L-BFGS-B', 
                         bounds=[(0.1, 5), (1, 20), (-np.pi, np.pi), (0.01, 1), (-2, 2), (-2, 2)])
            
            if res.success:
                a, b, c, d, e, f = res.x
                radial_pred = radial_form(x1, x2, a, b, c, d, e, f)
                radial_mse = np.mean((y - radial_pred) ** 2)
                radial_expr = f"{a:.6f}*sin({b:.6f}*sqrt(x1**2 + x2**2) + {c:.6f})*exp(-{d:.6f}*(x1**2 + x2**2)) + {e:.6f}*x1 + {f:.6f}*x2"
                candidates.append((radial_mse, radial_expr, radial_pred, 4))
            
            # Select best candidate with complexity penalty
            def score_candidate(mse, complexity):
                base_mse = poly_mse * 1.5  # Conservative baseline
                return mse * (1 + 0.1 * max(0, complexity - 2))
            
            best_idx = np.argmin([score_candidate(mse, comp) for mse, _, _, comp in candidates])
            best_mse, best_expr, best_pred, best_comp = candidates[best_idx]
            
            # Simplify expression symbolically if possible
            try:
                sympy_expr = sp.sympify(best_expr)
                simplified = sp.simplify(sympy_expr)
                if simplified.count_ops() < sympy_expr.count_ops():
                    best_expr = str(simplified).replace('sqrt', 'np.sqrt').replace('exp', 'np.exp')
            except:
                pass
            
            # Ensure expression uses allowed functions
            best_expr = best_expr.replace('np.', '')
            
            return {
                "expression": best_expr,
                "predictions": best_pred.tolist(),
                "details": {"complexity": best_comp}
            }