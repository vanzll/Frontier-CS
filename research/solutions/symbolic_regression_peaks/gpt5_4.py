import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 16))
        self.ridge_alpha = float(kwargs.get("ridge_alpha", 1e-6))
        self.rel_tol = float(kwargs.get("rel_tol", 1e-4))
        self.abs_tol = float(kwargs.get("abs_tol", 1e-8))
        self.seed = int(kwargs.get("random_state", 42))
        self.include_derivatives = bool(kwargs.get("include_derivatives", True))

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        x1 = X[:, 0].astype(float, copy=False)
        x2 = X[:, 1].astype(float, copy=False)
        n = X.shape[0]

        # Build features
        feature_exprs, Z = self._build_features(x1, x2, y)

        # Orthogonal Matching Pursuit with ridge refit
        try:
            coefs = self._omp(Z, y, self.max_terms, self.ridge_alpha, self.rel_tol, self.abs_tol)
        except Exception:
            # Fallback: ridge on all features
            coefs = self._ridge_fit(Z, y, self.ridge_alpha)

        # Prune small coefficients and refit
        coefs = self._prune_and_refit(Z, y, coefs, feature_exprs)

        # Build expression string
        expression = self._build_expression(feature_exprs, coefs)

        # Predictions
        y_pred = Z.dot(coefs)

        return {
            "expression": expression,
            "predictions": y_pred.tolist(),
            "details": {}
        }

    def _build_features(self, x1, x2, y):
        n = x1.shape[0]
        features = []
        Z_cols = []

        # Formatting helpers for expressions
        def fmt(v):
            if not np.isfinite(v):
                return "0"
            if abs(v) < 1e-12:
                return "0"
            return f"{v:.12g}"

        # Constant term
        features.append("1")
        Z_cols.append(np.ones(n))

        # Polynomial features
        # Use up to degree 3 with interactions
        features.extend(["x1", "x2"])
        Z_cols.append(x1)
        Z_cols.append(x2)

        x1_2 = x1 * x1
        x2_2 = x2 * x2
        x1x2 = x1 * x2

        features.extend(["x1**2", "x1*x2", "x2**2"])
        Z_cols.append(x1_2)
        Z_cols.append(x1x2)
        Z_cols.append(x2_2)

        x1_3 = x1_2 * x1
        x2_3 = x2_2 * x2
        x1_2x2 = x1_2 * x2
        x1x2_2 = x1 * x2_2

        features.extend(["x1**3", "x1**2*x2", "x1*x2**2", "x2**3"])
        Z_cols.append(x1_3)
        Z_cols.append(x1_2x2)
        Z_cols.append(x1x2_2)
        Z_cols.append(x2_3)

        # RBF features
        x1_min, x1_max = float(np.min(x1)), float(np.max(x1))
        x2_min, x2_max = float(np.min(x2)), float(np.max(x2))
        N = len(x1)
        if N <= 300:
            grid = 5
            width_factors = [0.9, 1.8]
        elif N <= 1500:
            grid = 6
            width_factors = [1.0, 2.0]
        else:
            grid = 7
            width_factors = [1.0, 2.0]

        # grid centers
        if grid <= 1:
            c1s = np.array([0.5 * (x1_min + x1_max)])
            c2s = np.array([0.5 * (x2_min + x2_max)])
        else:
            c1s = np.linspace(x1_min, x1_max, grid)
            c2s = np.linspace(x2_min, x2_max, grid)

        dx = (x1_max - x1_min) / max(grid - 1, 1)
        dy = (x2_max - x2_min) / max(grid - 1, 1)
        base_sigma = 1.25 * max(dx, dy) + 1e-12

        for wi, wf in enumerate(width_factors):
            sigma = base_sigma * wf
            alpha = 1.0 / (2.0 * (sigma * sigma))
            for c1 in c1s:
                for c2 in c2s:
                    # Gaussian RBF
                    g = np.exp(-alpha * ((x1 - c1) ** 2 + (x2 - c2) ** 2))
                    expr_g = f"exp(-({fmt(alpha)})*((x1 - ({fmt(c1)}))**2 + (x2 - ({fmt(c2)}))**2))"
                    features.append(expr_g)
                    Z_cols.append(g)

                    # Derivative-like features for narrower width only
                    if self.include_derivatives and wi == 0:
                        gx = (x1 - c1) * g
                        gy = (x2 - c2) * g
                        expr_gx = f"(x1 - ({fmt(c1)}))*{expr_g}"
                        expr_gy = f"(x2 - ({fmt(c2)}))*{expr_g}"
                        features.append(expr_gx)
                        Z_cols.append(gx)
                        features.append(expr_gy)
                        Z_cols.append(gy)

        Z = np.column_stack(Z_cols).astype(float, copy=False)
        return features, Z

    def _ridge_fit(self, Z, y, alpha):
        # Solve (Z^T Z + alpha*I) w = Z^T y
        m = Z.shape[1]
        G = Z.T @ Z
        s = np.trace(G) / m if m > 0 else 1.0
        lam = alpha * (s + 1e-12)
        try:
            A = G + lam * np.eye(m)
            b = Z.T @ y
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        return w

    def _omp(self, Z, y, k_max, ridge_alpha, rel_tol, abs_tol):
        n, m = Z.shape
        # Precompute norms
        col_norms = np.sqrt(np.sum(Z * Z, axis=0)) + 1e-12

        # Active set initialization with constant term (assume index 0 is '1')
        active = [0]
        inactive = set(range(m))
        inactive.discard(0)

        # Initial fit with constant only
        Za = Z[:, active]
        wa = self._ridge_fit(Za, y, ridge_alpha)
        r = y - Za @ wa
        prev_mse = float(np.mean(r * r))
        y_var = float(np.var(y)) + 1e-12

        for _ in range(k_max - 1):  # already have 1 active term
            # Correlations normalized by feature norm
            c = (Z.T @ r) / col_norms
            # Mask out active indices
            c_idx = np.array(list(inactive))
            if c_idx.size == 0:
                break
            j_rel = int(np.argmax(np.abs(c[c_idx])))
            j = int(c_idx[j_rel])

            # Add new feature
            active.append(j)
            inactive.discard(j)

            # Refit
            Za = Z[:, active]
            wa = self._ridge_fit(Za, y, ridge_alpha)

            # Update residual and check improvement
            r = y - Za @ wa
            mse = float(np.mean(r * r))
            improvement = prev_mse - mse
            if improvement < max(rel_tol * prev_mse, abs_tol * y_var):
                # Revert adding if no improvement
                active.pop()
                break
            prev_mse = mse

        # Final coefficients in full space
        full_w = np.zeros(m)
        if len(active) > 0:
            Za = Z[:, active]
            wa = self._ridge_fit(Za, y, ridge_alpha)
            for idx, j in enumerate(active):
                full_w[j] = wa[idx]
        return full_w

    def _prune_and_refit(self, Z, y, coefs, feature_exprs):
        # Compute contribution amplitude per feature
        n, m = Z.shape
        y_std = float(np.std(y)) + 1e-12

        contrib = np.sqrt(np.mean((Z * coefs.reshape(1, -1)) ** 2, axis=0))
        # Threshold small contributions
        thresh = 0.005 * y_std
        keep_idx = [i for i in range(m) if (i == 0) or (abs(contrib[i]) >= thresh and abs(coefs[i]) > 1e-10)]

        # If too many terms, keep top by contribution
        max_keep = max(1, self.max_terms)
        if len(keep_idx) > max_keep:
            active_nonconst = [i for i in keep_idx if i != 0]
            # sort by contribution desc
            order = sorted(active_nonconst, key=lambda i: -contrib[i])
            keep_idx = [0] + order[:max_keep - 1]

        # Refit on pruned set
        Za = Z[:, keep_idx]
        wa = self._ridge_fit(Za, y, self.ridge_alpha)

        # Rebuild full coef vector
        full_w = np.zeros(m)
        for k, j in enumerate(keep_idx):
            full_w[j] = wa[k]

        # Additional cleanup: set very small weights to zero
        small_mask = np.abs(full_w) < (1e-12 * max(1.0, np.max(np.abs(full_w)) + 1e-12))
        full_w[small_mask] = 0.0

        return full_w

    def _build_expression(self, feature_exprs, coefs):
        def fmt(v):
            if not np.isfinite(v):
                return "0"
            if abs(v) < 1e-12:
                return "0"
            return f"{v:.12g}"

        terms = []
        const = coefs[0] if len(coefs) > 0 else 0.0
        const_str = fmt(const)

        for j in range(1, len(coefs)):
            w = coefs[j]
            if abs(w) < 1e-12:
                continue
            expr = feature_exprs[j]
            terms.append(f"({fmt(w)})*({expr})")

        if len(terms) == 0:
            return const_str

        if const_str != "0":
            return const_str + " + " + " + ".join(terms)
        else:
            return " + ".join(terms)