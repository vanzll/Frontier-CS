import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 42)
        self.max_terms = kwargs.get("max_terms", 12)
        self.ridge_alpha = kwargs.get("ridge_alpha", 1e-6)
        self.validation_ratio = kwargs.get("validation_ratio", 0.2)

    def _fmt(self, v):
        if abs(v) < 1e-14:
            return "0"
        s = format(float(v), ".12g")
        if "nan" in s or "inf" in s or "-inf" in s:
            return "0"
        return s

    def _ridge_fit(self, A, y, lam):
        p = A.shape[1]
        AtA = A.T @ A
        AtY = A.T @ y
        reg = np.diag(np.concatenate([np.full(p - 1, lam), np.array([0.0])]))
        M = AtA + reg
        try:
            w = np.linalg.solve(M, AtY)
        except np.linalg.LinAlgError:
            # Fallback to lstsq or pinv for numerical stability
            try:
                w = np.linalg.lstsq(M, AtY, rcond=None)[0]
            except Exception:
                w = np.linalg.pinv(M) @ AtY
        return w

    def _build_polynomial_terms(self, x1, x2):
        terms = []
        # Degree 1
        terms.append(("x1", x1))
        terms.append(("x2", x2))
        # Degree 2
        terms.append(("x1**2", x1**2))
        terms.append(("x2**2", x2**2))
        terms.append(("x1*x2", x1 * x2))
        # Degree 3
        terms.append(("x1**3", x1**3))
        terms.append(("x2**3", x2**3))
        terms.append(("x1**2*x2", (x1**2) * x2))
        terms.append(("x1*x2**2", x1 * (x2**2)))
        return terms

    def _build_rbf_terms(self, x1, x2):
        terms = []
        # Quantiles for centers
        q_levels = np.array([0.1, 0.5, 0.9])
        c1 = np.quantile(x1, q_levels) if x1.size > 0 else np.array([0.0, 0.0, 0.0])
        c2 = np.quantile(x2, q_levels) if x2.size > 0 else np.array([0.0, 0.0, 0.0])
        # Scales
        q10_x1, q90_x1 = np.quantile(x1, 0.1), np.quantile(x1, 0.9)
        q10_x2, q90_x2 = np.quantile(x2, 0.1), np.quantile(x2, 0.9)
        s1 = max((q90_x1 - q10_x1) / 2.0, np.std(x1), 1e-6)
        s2 = max((q90_x2 - q10_x2) / 2.0, np.std(x2), 1e-6)
        g1_base = 1.0 / (2.0 * (s1 ** 2))
        g2_base = 1.0 / (2.0 * (s2 ** 2))
        width_factors = [0.8, 1.6]
        for wf in width_factors:
            a = g1_base * wf
            b = g2_base * wf
            a_str = self._fmt(a)
            b_str = self._fmt(b)
            for cx in c1:
                cx_str = self._fmt(cx)
                for cy in c2:
                    cy_str = self._fmt(cy)
                    expr = f"exp(-{a_str}*(x1 - ({cx_str}))**2 - {b_str}*(x2 - ({cy_str}))**2)"
                    vals = np.exp(-a * (x1 - cx) ** 2 - b * (x2 - cy) ** 2)
                    terms.append((expr, vals))
        return terms

    def _build_radial_terms(self, x1, x2):
        terms = []
        # Isotropic radial Gaussian terms centered at origin, possibly multiplied by x1 or x2
        # Scale based on typical spread
        q10_x1, q90_x1 = np.quantile(x1, 0.1), np.quantile(x1, 0.9)
        q10_x2, q90_x2 = np.quantile(x2, 0.1), np.quantile(x2, 0.9)
        s1 = max((q90_x1 - q10_x1) / 2.0, np.std(x1), 1e-6)
        s2 = max((q90_x2 - q10_x2) / 2.0, np.std(x2), 1e-6)
        s = max((s1 + s2) / 2.0, 1e-6)
        g_base = 1.0 / (2.0 * (s ** 2))
        factors = [0.5, 1.0, 2.0]
        r2 = x1 ** 2 + x2 ** 2
        for f in factors:
            k = g_base * f
            k_str = self._fmt(k)
            expr = f"exp(-{k_str}*(x1**2 + x2**2))"
            phi = np.exp(-k * r2)
            terms.append((expr, phi))
            terms.append((f"x1*({expr})", x1 * phi))
            terms.append((f"x2*({expr})", x2 * phi))
        return terms

    def _build_all_terms(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        terms = []
        # Polynomials
        terms += self._build_polynomial_terms(x1, x2)
        # 2D RBF grid
        terms += self._build_rbf_terms(x1, x2)
        # Radial Gaussians and modulations
        terms += self._build_radial_terms(x1, x2)
        return terms

    def _train_val_split(self, n, ratio, rng):
        if n <= 5:
            # too few, keep all for training
            train_idx = np.arange(n)
            val_idx = np.arange(0)
            return train_idx, val_idx
        n_val = max(1, int(n * ratio))
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    def _forward_stepwise(self, Z_train, y_train, Z_val, y_val, max_terms, lam):
        n_train, m = Z_train.shape
        selected = []
        current_best = np.inf
        # Start metric with intercept-only model
        if Z_val is not None and y_val is not None and y_val.size > 0:
            c0 = float(np.mean(y_train)) if n_train > 0 else 0.0
            current_best = float(np.mean((y_val - c0) ** 2))
        else:
            current_best = np.inf
        for _ in range(max_terms):
            best_j = None
            best_metric = current_best
            for j in range(m):
                if j in selected:
                    continue
                cols = selected + [j]
                A_tr = np.column_stack([Z_train[:, cols], np.ones(n_train)])
                w = self._ridge_fit(A_tr, y_train, lam)
                if Z_val is not None and y_val is not None and y_val.size > 0:
                    A_va = np.column_stack([Z_val[:, cols], np.ones(Z_val.shape[0])])
                    pred_val = A_va @ w
                    mse = float(np.mean((y_val - pred_val) ** 2))
                else:
                    # If no validation, use training MSE (fallback)
                    pred_tr = A_tr @ w
                    mse = float(np.mean((y_train - pred_tr) ** 2))
                if mse + 1e-12 < best_metric:
                    best_metric = mse
                    best_j = j
            if best_j is None:
                break
            selected.append(best_j)
            current_best = best_metric
        return selected

    def _build_expression(self, coeffs, feature_exprs):
        # coeffs: array of length k+1, last is intercept
        terms = []
        k = len(coeffs) - 1
        for i in range(k):
            c = coeffs[i]
            if abs(c) < 1e-12:
                continue
            c_str = self._fmt(c)
            term_expr = feature_exprs[i]
            terms.append(f"({c_str})*({term_expr})")
        c0 = coeffs[-1]
        expr_parts = []
        if abs(c0) >= 1e-12:
            expr_parts.append(self._fmt(c0))
        if terms:
            expr_parts.extend(terms)
        if not expr_parts:
            return "0"
        return " + ".join(expr_parts)

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        # Build features
        term_list = self._build_all_terms(X)
        if not term_list:
            # Fallback: constant mean
            c0 = float(np.mean(y)) if y.size > 0 else 0.0
            expression = self._fmt(c0)
            predictions = np.full(n, c0)
            return {"expression": expression, "predictions": predictions.tolist(), "details": {}}

        # Build feature matrix
        feature_exprs = [expr for expr, _ in term_list]
        Z = np.column_stack([vals for _, vals in term_list])

        # Train/validation split
        train_idx, val_idx = self._train_val_split(n, self.validation_ratio, rng)
        y_train = y[train_idx]
        Z_train = Z[train_idx, :]
        if val_idx.size > 0:
            y_val = y[val_idx]
            Z_val = Z[val_idx, :]
        else:
            y_val = None
            Z_val = None

        # Forward stepwise selection
        selected = self._forward_stepwise(Z_train, y_train, Z_val, y_val, self.max_terms, self.ridge_alpha)
        if len(selected) == 0:
            # Fallback: pick feature with best correlation
            corrs = np.array([abs(np.corrcoef(Z_train[:, j], y_train)[0, 1]) if np.std(Z_train[:, j]) > 0 else 0.0 for j in range(Z_train.shape[1])])
            best_j = int(np.argmax(corrs))
            selected = [best_j]

        # Final fit on all data with selected features + intercept
        A_full = np.column_stack([Z[:, selected], np.ones(n)])
        w_full = self._ridge_fit(A_full, y, self.ridge_alpha)

        # Build final expression
        selected_exprs = [feature_exprs[j] for j in selected]
        expression = self._build_expression(w_full, selected_exprs)

        # Predictions
        predictions = A_full @ w_full

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }