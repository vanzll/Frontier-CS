import numpy as np
import sympy as sp


class Solution:
    def __init__(self, **kwargs):
        self.max_terms = kwargs.get("max_terms", 6)
        self.coef_tol = kwargs.get("coef_tol", 1e-6)
        self.abs_improvement_tol = kwargs.get("abs_improvement_tol", 1e-6)
        self.rel_improvement_tol = kwargs.get("rel_improvement_tol", 1e-4)

    def _fit_ls(self, A, y):
        if A.size == 0:
            return np.zeros(0, dtype=float), np.zeros_like(y, dtype=float)
        coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ coefs
        return coefs, y_pred

    def _build_expression(self, basis_exprs, indices, coefs):
        terms = []
        for idx, c in zip(indices, coefs):
            c_float = float(c)
            if abs(c_float) < 1e-12:
                continue
            coeff_str = "{:.10g}".format(c_float)

            expr = basis_exprs[idx]

            if expr == "1":
                term = coeff_str
            else:
                if coeff_str in ("1", "1.0"):
                    term = expr
                elif coeff_str in ("-1", "-1.0"):
                    term = f"-{expr}"
                else:
                    term = f"{coeff_str}*{expr}"

            terms.append(term)

        if not terms:
            return "0"

        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += " - " + term[1:]
            else:
                result += " + " + term
        return result

    def _compute_complexity(self, expr_str):
        try:
            expr = sp.sympify(expr_str)
        except Exception:
            return None

        bin_ops = 0
        unary_ops = 0
        binary_types = (sp.Add, sp.Mul, sp.Pow)
        unary_funcs = (sp.sin, sp.cos, sp.exp, sp.log)

        for node in sp.preorder_traversal(expr):
            if isinstance(node, binary_types):
                bin_ops += 1
            func = getattr(node, "func", None)
            if func in unary_funcs:
                unary_ops += 1

        return 2 * bin_ops + unary_ops

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_samples = X.shape[0]
        if n_samples == 0:
            expression = "0"
            return {
                "expression": expression,
                "predictions": [],
                "details": {"complexity": 0},
            }

        if X.shape[1] < 2:
            if X.shape[1] == 1:
                x1 = X[:, 0]
                x2 = np.zeros_like(x1)
                X = np.column_stack([x1, x2])
            else:
                x1 = np.zeros(n_samples, dtype=float)
                x2 = np.zeros(n_samples, dtype=float)
                X = np.column_stack([x1, x2])

        x1 = X[:, 0]
        x2 = X[:, 1]

        basis_exprs = [
            "1",
            "sin(x1)",
            "cos(x1)",
            "sin(x2)",
            "cos(x2)",
            "sin(x1)*cos(x2)",
            "cos(x1)*sin(x2)",
            "sin(x1)*sin(x2)",
            "cos(x1)*cos(x2)",
            "x1",
            "x2",
            "sin(x1 + x2)",
            "cos(x1 + x2)",
            "sin(x1 - x2)",
            "cos(x1 - x2)",
        ]

        nbasis = len(basis_exprs)
        B = np.empty((n_samples, nbasis), dtype=float)

        ones = np.ones_like(x1)
        s1 = np.sin(x1)
        c1 = np.cos(x1)
        s2 = np.sin(x2)
        c2 = np.cos(x2)
        x1_plus_x2 = x1 + x2
        x1_minus_x2 = x1 - x2
        s_add = np.sin(x1_plus_x2)
        c_add = np.cos(x1_plus_x2)
        s_sub = np.sin(x1_minus_x2)
        c_sub = np.cos(x1_minus_x2)

        B[:, 0] = ones
        B[:, 1] = s1
        B[:, 2] = c1
        B[:, 3] = s2
        B[:, 4] = c2
        B[:, 5] = s1 * c2
        B[:, 6] = c1 * s2
        B[:, 7] = s1 * s2
        B[:, 8] = c1 * c2
        B[:, 9] = x1
        B[:, 10] = x2
        B[:, 11] = s_add
        B[:, 12] = c_add
        B[:, 13] = s_sub
        B[:, 14] = c_sub

        selected = [0]  # start with constant term
        A = B[:, selected]
        coefs_sel, y_pred_sel = self._fit_ls(A, y)
        resid = y - y_pred_sel
        best_mse = float(np.mean(resid * resid))

        remaining = list(range(1, nbasis))
        max_terms = min(self.max_terms, nbasis)

        while len(selected) < max_terms and remaining:
            best_step_mse = best_mse
            best_step_index = None
            best_step_coefs = None
            best_step_pred = None

            for j in remaining:
                trial_indices = selected + [j]
                A_trial = B[:, trial_indices]
                coefs_trial, y_pred_trial = self._fit_ls(A_trial, y)
                resid_trial = y - y_pred_trial
                mse_trial = float(np.mean(resid_trial * resid_trial))

                if mse_trial < best_step_mse:
                    best_step_mse = mse_trial
                    best_step_index = j
                    best_step_coefs = coefs_trial
                    best_step_pred = y_pred_trial

            if best_step_index is None or best_step_mse >= best_mse:
                break

            improvement = best_mse - best_step_mse
            thresh = max(self.abs_improvement_tol, best_mse * self.rel_improvement_tol)

            if improvement <= thresh:
                break

            selected.append(best_step_index)
            remaining.remove(best_step_index)
            coefs_sel = best_step_coefs
            y_pred_sel = best_step_pred
            best_mse = best_step_mse

        abs_coefs = np.abs(coefs_sel)
        active_positions = [i for i, v in enumerate(abs_coefs) if v > self.coef_tol]

        if not active_positions:
            expression = "0"
            y_pred_final = np.zeros_like(y)
            complexity = self._compute_complexity(expression)
            details = {}
            if complexity is not None:
                details["complexity"] = int(complexity)
            return {
                "expression": expression,
                "predictions": y_pred_final.tolist(),
                "details": details,
            }

        final_indices = [selected[i] for i in active_positions]
        final_coefs = coefs_sel[active_positions]

        A_final = B[:, final_indices]
        y_pred_final = A_final @ final_coefs

        expression = self._build_expression(basis_exprs, final_indices, final_coefs)
        complexity = self._compute_complexity(expression)
        details = {}
        if complexity is not None:
            details["complexity"] = int(complexity)

        return {
            "expression": expression,
            "predictions": y_pred_final.tolist(),
            "details": details,
        }