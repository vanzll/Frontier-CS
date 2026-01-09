import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _format_float(self, x):
        # Format with reasonable precision, strip trailing zeros and dot
        s = f"{x:.12g}"
        return s

    def _generate_features(self, x1, x2):
        # Precompute common quantities
        n = x1.shape[0]
        r2 = x1 * x1 + x2 * x2
        r = np.sqrt(np.maximum(r2, 0.0))

        ones = np.ones_like(r)
        # Amplitudes and their expressions (templates using R and R2)
        amps = []
        # ('expr_template', values)
        amps.append(("1", ones))
        amps.append(("1/(1 + R)", 1.0 / (1.0 + r)))
        amps.append(("1/(1 + R2)", 1.0 / (1.0 + r2)))
        amps.append(("1/(1 + 0.5*R2)", 1.0 / (1.0 + 0.5 * r2)))

        # Helper to add a feature
        features_values = []
        features_exprs = []

        def add_feature(values, expr):
            if not np.all(np.isfinite(values)):
                return
            # Skip (near) constant zero features
            if np.allclose(values, 0.0):
                return
            features_values.append(values.astype(float, copy=False))
            features_exprs.append(expr)

        # Polynomial-like features (amplitudes without trig)
        add_feature(r, "(x1**2 + x2**2)**0.5")
        add_feature(r2, "(x1**2 + x2**2)")
        add_feature(1.0 / (1.0 + r2), "1/(1 + (x1**2 + x2**2))")
        add_feature(1.0 / (1.0 + r), "1/(1 + (x1**2 + x2**2)**0.5)")

        # Trig features based on r
        k_r = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        r_amp_indices = [0, 1, 2]  # use 1, 1/(1+r), 1/(1+r2)
        for k in k_r:
            sin_r = np.sin(k * r)
            cos_r = np.cos(k * r)
            for ai in r_amp_indices:
                a_expr, a_val = amps[ai]
                if ai == 0:
                    add_feature(sin_r, f"sin({self._format_float(k)}*((x1**2 + x2**2)**0.5))")
                    add_feature(cos_r, f"cos({self._format_float(k)}*((x1**2 + x2**2)**0.5))")
                else:
                    add_feature(sin_r * a_val, f"sin({self._format_float(k)}*((x1**2 + x2**2)**0.5))*{a_expr.replace('R2', '(x1**2 + x2**2)').replace('R', '((x1**2 + x2**2)**0.5)')}")
                    add_feature(cos_r * a_val, f"cos({self._format_float(k)}*((x1**2 + x2**2)**0.5))*{a_expr.replace('R2', '(x1**2 + x2**2)').replace('R', '((x1**2 + x2**2)**0.5)')}")

        # Trig features based on r^2
        k_r2 = [0.5, 1.0, 2.0, 3.0, 5.0]
        r2_amp_indices = [0, 2, 3]  # use 1, 1/(1+r2), 1/(1+0.5*r2)
        for k in k_r2:
            sin_r2 = np.sin(k * r2)
            cos_r2 = np.cos(k * r2)
            for ai in r2_amp_indices:
                a_expr, a_val = amps[ai]
                if ai == 0:
                    add_feature(sin_r2, f"sin({self._format_float(k)}*(x1**2 + x2**2))")
                    add_feature(cos_r2, f"cos({self._format_float(k)}*(x1**2 + x2**2))")
                else:
                    add_feature(sin_r2 * a_val, f"sin({self._format_float(k)}*(x1**2 + x2**2))*{a_expr.replace('R2', '(x1**2 + x2**2)').replace('R', '((x1**2 + x2**2)**0.5)')}")
                    add_feature(cos_r2 * a_val, f"cos({self._format_float(k)}*(x1**2 + x2**2))*{a_expr.replace('R2', '(x1**2 + x2**2)').replace('R', '((x1**2 + x2**2)**0.5)')}")

        # Trig features based on x1, x2
        k_x = [1.0, 2.0, 3.0]
        # Use amplitude 1/(1+r2) to control growth
        amp_x_expr, amp_x_val = "1/(1 + (x1**2 + x2**2))", 1.0 / (1.0 + r2)
        for k in k_x:
            add_feature(np.sin(k * x1) * amp_x_val, f"sin({self._format_float(k)}*x1)*{amp_x_expr}")
            add_feature(np.cos(k * x1) * amp_x_val, f"cos({self._format_float(k)}*x1)*{amp_x_expr}")
            add_feature(np.sin(k * x2) * amp_x_val, f"sin({self._format_float(k)}*x2)*{amp_x_expr}")
            add_feature(np.cos(k * x2) * amp_x_val, f"cos({self._format_float(k)}*x2)*{amp_x_expr}")

        # Trig of (x1 + x2)
        s = x1 + x2
        for k in [1.0, 2.0, 3.0]:
            add_feature(np.sin(k * s) * amp_x_val, f"sin({self._format_float(k)}*(x1 + x2))*{amp_x_expr}")
            add_feature(np.cos(k * s) * amp_x_val, f"cos({self._format_float(k)}*(x1 + x2))*{amp_x_expr}")

        # Stack features into matrix
        if len(features_values) == 0:
            A = np.zeros((n, 0), dtype=float)
        else:
            A = np.column_stack(features_values)

        return A, features_exprs

    def _omp_select(self, A, y, max_terms=16):
        n, p = A.shape
        if p == 0:
            # Only intercept
            c0 = float(np.mean(y))
            pred = np.full_like(y, c0, dtype=float)
            return [], np.array([c0]), pred

        # Normalize columns for correlation computation
        col_norms = np.sqrt(np.sum(A * A, axis=0))
        col_norms[col_norms == 0] = 1.0

        selected = []
        best_bic = np.inf
        best_state = None

        # Start with intercept model
        ones = np.ones(n, dtype=float)
        # Residual initialized with just intercept fit
        c0 = float(np.mean(y))
        pred = np.full(n, c0, dtype=float)
        residual = y - pred
        rss_prev = float(np.dot(residual, residual))

        # We'll store incremental results to potentially choose earlier K based on BIC
        for t in range(min(max_terms, p)):
            # Compute normalized correlations
            # c_j = A[:, j]^T residual / ||A[:, j]||
            corr = (A.T @ residual) / col_norms
            # Exclude already selected
            if selected:
                corr[selected] = 0.0
            # Pick index with maximum absolute correlation
            j = int(np.argmax(np.abs(corr)))
            if j in selected:
                # Shouldn't happen, but break to avoid infinite loop
                break
            selected.append(j)

            # Solve least squares with intercept + selected features
            A_sel = A[:, selected]
            B = np.column_stack([ones, A_sel])
            # Use lstsq to handle potential collinearity
            coefs, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
            pred = B @ coefs
            residual = y - pred
            rss = float(np.dot(residual, residual))
            k = 1 + len(selected)  # intercept + number of selected features
            # Bayesian Information Criterion
            # Add tiny term to avoid log(0)
            bic = n * np.log(rss / max(n, 1) + 1e-16) + k * np.log(max(n, 2))
            if bic < best_bic:
                best_bic = bic
                best_state = (selected.copy(), coefs.copy(), pred.copy())

            # Early stopping if little improvement
            if rss_prev - rss <= 1e-12 * (1.0 + rss_prev):
                break
            rss_prev = rss

        if best_state is None:
            # Fallback: use last state
            A_sel = A[:, selected] if selected else np.zeros((n, 0), dtype=float)
            B = np.column_stack([ones, A_sel])
            coefs, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
            pred = B @ coefs
            return selected, coefs, pred
        else:
            selected_best, coefs_best, pred_best = best_state
            return selected_best, coefs_best, pred_best

    def _build_expression(self, c0, coefs, selected_indices, exprs):
        # coefs includes [c0, w1, w2, ...]
        terms = []
        # Constant term
        c0_use = c0
        # Build terms for each selected feature
        for w, idx in zip(coefs[1:], selected_indices):
            if abs(w) < 1e-14:
                continue
            expr = exprs[idx]
            mag = abs(w)
            mag_str = self._format_float(mag)
            if np.isclose(mag, 1.0, atol=1e-12):
                term = f"({expr})"
            else:
                term = f"{mag_str}*({expr})"
            if w < 0:
                term = f"- {term}"
            else:
                term = f"+ {term}"
            terms.append(term)

        if abs(c0_use) < 1e-14 and not terms:
            expression = "0"
        elif abs(c0_use) < 1e-14:
            # No constant, only terms
            # Remove leading '+ ' if present in the first term
            if terms:
                first = terms[0]
                if first.startswith("+ "):
                    terms[0] = first[2:]
            expression = " ".join(terms) if terms else "0"
        else:
            c0_str = self._format_float(c0_use)
            if terms:
                expression = c0_str + " " + " ".join(terms)
            else:
                expression = c0_str

        return expression

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Clean NaNs/Infs
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must have shape (n,) and match X")

        X = np.nan_to_num(X, copy=False)
        y = np.nan_to_num(y, copy=False)

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Generate features
        A, exprs = self._generate_features(x1, x2)

        # Choose max number of terms depending on n and p
        p = A.shape[1]
        if p == 0:
            expression = self._format_float(float(np.mean(y)))
            predictions = np.full_like(y, float(np.mean(y)), dtype=float)
            return {"expression": expression, "predictions": predictions.tolist(), "details": {}}

        # Heuristic for max terms
        max_terms = min(20, max(8, int(0.05 * p)))
        max_terms = max_terms if "max_terms" not in self.kwargs else int(self.kwargs["max_terms"])
        selected_indices, coefs, pred = self._omp_select(A, y, max_terms=max_terms)

        # Build expression string
        c0 = float(coefs[0]) if coefs.size > 0 else float(np.mean(y))
        expression = self._build_expression(c0, coefs, selected_indices, exprs)

        # Predictions already computed as pred
        predictions = pred

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }