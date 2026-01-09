import numpy as np
import sympy as sp

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be of shape (n, 2)")
        if y.shape[0] != n:
            raise ValueError("y must have length n")

        x1 = X[:, 0]
        x2 = X[:, 1]

        s1 = np.sin(x1)
        c1 = np.cos(x1)
        s2 = np.sin(x2)
        c2 = np.cos(x2)
        s1c2 = s1 * c2
        c1s2 = c1 * s2
        c1c2 = c1 * c2
        s1s2 = s1 * s2
        s_sum = np.sin(x1 + x2)
        s_diff = np.sin(x1 - x2)
        c_sum = np.cos(x1 + x2)
        c_diff = np.cos(x1 - x2)

        var_y = float(np.var(y)) if y.size > 1 else 0.0

        # Helper: compute complexity via sympy
        def compute_complexity(expr_str: str) -> int:
            x1_sym, x2_sym = sp.symbols('x1 x2')
            try:
                expr = sp.sympify(expr_str, locals={'x1': x1_sym, 'x2': x2_sym, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log})
            except Exception:
                # Fallback simple count
                unary = expr_str.count('sin') + expr_str.count('cos') + expr_str.count('exp') + expr_str.count('log')
                binary = expr_str.count('+') + expr_str.count('-') + expr_str.count('*') + expr_str.count('/') + expr_str.count('**')
                return 2 * binary + unary

            def count_ops(e):
                if e.is_Atom:
                    return (0, 0)
                binaries = 0
                unaries = 0
                if isinstance(e, sp.Add):
                    binaries += max(len(e.args) - 1, 0)
                    for a in e.args:
                        b, u = count_ops(a)
                        binaries += b
                        unaries += u
                    return (binaries, unaries)
                if isinstance(e, sp.Mul):
                    binaries += max(len(e.args) - 1, 0)
                    for a in e.args:
                        b, u = count_ops(a)
                        binaries += b
                        unaries += u
                    return (binaries, unaries)
                if isinstance(e, sp.Pow):
                    binaries += 1
                    b1, u1 = count_ops(e.base)
                    b2, u2 = count_ops(e.exp)
                    return (binaries + b1 + b2, unaries + u1 + u2)
                if isinstance(e, sp.Function):
                    # Only count allowed unary functions
                    if isinstance(e, (sp.sin, sp.cos, sp.exp, sp.log)):
                        unaries += 1
                    for a in e.args:
                        b, u = count_ops(a)
                        binaries += b
                        unaries += u
                    return (binaries, unaries)
                # General fallback: traverse args
                for a in e.args:
                    b, u = count_ops(a)
                    binaries += b
                    unaries += u
                return (binaries, unaries)

            b, u = count_ops(expr)
            return 2 * b + u

        # Helper: render linear combination of terms with an optional intercept
        def format_number(val: float) -> str:
            return format(val, ".10g")

        def render_linear_expression(coeffs, term_names, intercept=None) -> str:
            tol = 1e-12
            parts = []

            for coef, name in zip(coeffs, term_names):
                if abs(coef) < tol:
                    continue
                sign_negative = coef < 0
                aval = abs(coef)
                if abs(aval - 1.0) < 1e-10:
                    term_str = name
                else:
                    term_str = f"{format_number(aval)}*{name}"
                if not parts:
                    parts.append(("-" + term_str) if sign_negative else term_str)
                else:
                    parts.append((" - " + term_str) if sign_negative else (" + " + term_str))

            if intercept is not None and abs(intercept) >= tol:
                sign_negative = intercept < 0
                aval = abs(intercept)
                const_str = format_number(aval)
                if not parts:
                    parts.append("-" + const_str if sign_negative else const_str)
                else:
                    parts.append((" - " + const_str) if sign_negative else (" + " + const_str))

            if not parts:
                return "0"
            # Join keeping already added signs/spaces
            expr = parts[0]
            if len(parts) > 1:
                expr += "".join(parts[1:])
            return expr

        # Helper: OLS fit for a set of features
        def fit_ols(feature_arrays, term_names, add_intercept=True):
            A = np.column_stack(feature_arrays) if feature_arrays else np.zeros((n, 0))
            if add_intercept:
                A_aug = np.column_stack([A, np.ones(n)])
            else:
                A_aug = A
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_aug, y, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = np.zeros(A_aug.shape[1])
            preds = A_aug @ coeffs
            if add_intercept:
                term_coeffs = coeffs[:-1]
                intercept = coeffs[-1]
            else:
                term_coeffs = coeffs
                intercept = None
            expr = render_linear_expression(term_coeffs, term_names, intercept)
            return expr, preds

        # Helper: add a candidate
        def mse_of(preds):
            diff = y - preds
            return float(np.mean(diff * diff))

        candidates = []

        def add_candidate(expr_str, preds):
            mse = mse_of(preds)
            comp = compute_complexity(expr_str)
            candidates.append((mse, comp, expr_str, preds))

        # Simple fixed-form candidates
        add_candidate("sin(x1) + cos(x2)", s1 + c2)
        add_candidate("sin(x1) * cos(x2)", s1c2)
        add_candidate("sin(x1) - cos(x2)", s1 - c2)
        add_candidate("sin(x1) + sin(x2)", s1 + s2)
        add_candidate("cos(x1) + cos(x2)", c1 + c2)
        add_candidate("cos(x1) + sin(x2)", c1 + s2)
        add_candidate("sin(x1 + x2)", s_sum)
        add_candidate("sin(x1 - x2)", s_diff)
        add_candidate("cos(x1 + x2)", c_sum)
        add_candidate("cos(x1 - x2)", c_diff)
        add_candidate("sin(x1)", s1)
        add_candidate("cos(x2)", c2)
        add_candidate("cos(x1)", c1)
        add_candidate("sin(x2)", s2)

        # OLS-based candidates with small sets of features
        expr, preds = fit_ols([s1, c2], ["sin(x1)", "cos(x2)"], add_intercept=True)
        add_candidate(expr, preds)

        expr, preds = fit_ols([s1, s2], ["sin(x1)", "sin(x2)"], add_intercept=True)
        add_candidate(expr, preds)

        expr, preds = fit_ols([c1, c2], ["cos(x1)", "cos(x2)"], add_intercept=True)
        add_candidate(expr, preds)

        expr, preds = fit_ols([s1c2], ["sin(x1) * cos(x2)"], add_intercept=True)
        add_candidate(expr, preds)

        expr, preds = fit_ols([s1, c1, s2, c2], ["sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)"], add_intercept=True)
        add_candidate(expr, preds)

        # Select best candidate: prefer simpler expressions within small MSE tolerance
        if candidates:
            candidates.sort(key=lambda t: t[0])
            min_mse = candidates[0][0]
            tol_mse = max(1e-6 * (var_y + 1e-12), 1e-12)
            feasible = [c for c in candidates if c[0] <= min_mse + tol_mse]
            feasible.sort(key=lambda t: (t[1], t[0]))  # sort by complexity then MSE
            best = feasible[0]
        else:
            # Fallback: constant zero
            best = (mse_of(np.zeros_like(y)), compute_complexity("0"), "0", np.zeros_like(y))

        best_mse, best_complexity, best_expr, best_preds = best

        result = {
            "expression": best_expr,
            "predictions": best_preds.tolist(),
            "details": {"complexity": int(best_complexity)}
        }
        return result