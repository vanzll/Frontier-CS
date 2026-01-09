import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 42)
        self.max_terms = kwargs.get("max_terms", 18)
        self.alpha_ridge = kwargs.get("alpha_ridge", 1e-3)
        self.scan_alpha = kwargs.get("scan_alpha", 1e-4)
        self.r_w_range = kwargs.get("r_w_range", (0.5, 20.0))
        self.s_w_range = kwargs.get("s_w_range", (0.2, 10.0))
        self.r_w_points = kwargs.get("r_w_points", 60)
        self.s_w_points = kwargs.get("s_w_points", 50)
        self.top_k_r = kwargs.get("top_k_r", 3)
        self.top_k_s = kwargs.get("top_k_s", 3)
        self.min_freq_separation = kwargs.get("min_freq_separation", 0.25)
        self.D_r = kwargs.get("D_r", 2)
        self.D_s = kwargs.get("D_s", 2)
        self.use_poly_terms = kwargs.get("use_poly_terms", True)
        self.poly_degrees = kwargs.get("poly_degrees", [1, 2])  # on s = x1^2 + x2^2

    def fmt(self, c):
        return f"{float(c):.16g}"

    def _ridge_fit_with_intercept(self, X, y, alpha):
        # X: shape (n, p), without intercept column
        n, p = X.shape
        # Standardize columns of X for stability, but return coefficients in original scale.
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xn = (X - mu) / sigma
        # Build augmented system to penalize only features, not intercept
        ones = np.ones((n, 1), dtype=float)
        A = np.hstack([ones, Xn])  # (n, p+1)
        penalty = np.zeros((p, p + 1), dtype=float)
        penalty[:, 1:] = np.sqrt(alpha) * np.eye(p)
        A_aug = np.vstack([A, penalty])
        y_aug = np.concatenate([y, np.zeros(p, dtype=float)])
        coef_aug, _, _, _ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        b0 = coef_aug[0]
        w_norm = coef_aug[1:]
        # Convert to original scale
        w = w_norm / sigma
        b = b0 - np.sum((w_norm * mu) / sigma)
        return b, w, mu, sigma

    def _fit_predict(self, X, y, alpha):
        b, w, _, _ = self._ridge_fit_with_intercept(X, y, alpha)
        yhat = b + X @ w
        return b, w, yhat

    def _scan_freqs(self, y, base, degrees, wmin, wmax, ngrid, top_k):
        # Scan w to find top_k frequencies minimizing MSE of small model with sin/cos and polynomial amplitude
        ws = np.linspace(wmin, wmax, int(ngrid))
        results = []
        n = base.shape[0]
        for w in ws:
            cols = []
            for deg in degrees:
                pow_arr = base ** deg
                cols.append(pow_arr * np.sin(w * base))
                cols.append(pow_arr * np.cos(w * base))
            B = np.column_stack(cols) if cols else np.zeros((n, 0))
            # Fit small ridge with intercept
            try:
                b, v, yhat = self._fit_predict(B, y, self.scan_alpha)
                mse = np.mean((y - yhat) ** 2)
            except Exception:
                mse = np.inf
            results.append((w, mse))
        # Sort by mse ascending and pick unique w with min separation
        results.sort(key=lambda t: t[1])
        selected = []
        for w, _ in results:
            if len(selected) >= top_k:
                break
            if all(abs(w - ws0) >= self.min_freq_separation for ws0 in selected):
                selected.append(w)
        return selected

    def _build_terms(self, X, w_r_list, w_s_list):
        # Build term dictionaries: each term -> dict with 'name' and 'vec'
        x1 = X[:, 0]
        x2 = X[:, 1]
        s_arr = x1 * x1 + x2 * x2
        r_arr = np.sqrt(s_arr)

        s_expr = "(x1**2 + x2**2)"
        r_expr = f"({s_expr}**0.5)"

        terms = []

        # Polynomial terms on s (exclude constant)
        if self.use_poly_terms:
            for deg in self.poly_degrees:
                vec = s_arr ** deg
                name = f"({s_expr})**{deg}" if deg != 1 else f"{s_expr}"
                terms.append({"name": name, "vec": vec})

        # r-based trig terms
        for w in w_r_list:
            w_str = self.fmt(w)
            sin_expr = f"sin({w_str}*{r_expr})"
            cos_expr = f"cos({w_str}*{r_expr})"
            # degrees 0..D_r
            for deg in range(self.D_r + 1):
                if deg == 0:
                    vec_sin = np.sin(w * r_arr)
                    vec_cos = np.cos(w * r_arr)
                    name_sin = sin_expr
                    name_cos = cos_expr
                    terms.append({"name": name_sin, "vec": vec_sin})
                    terms.append({"name": name_cos, "vec": vec_cos})
                else:
                    pow_vec = r_arr ** deg
                    pow_expr = f"({r_expr})**{deg}"
                    vec_sin = pow_vec * np.sin(w * r_arr)
                    vec_cos = pow_vec * np.cos(w * r_arr)
                    name_sin = f"{pow_expr}*{sin_expr}"
                    name_cos = f"{pow_expr}*{cos_expr}"
                    terms.append({"name": name_sin, "vec": vec_sin})
                    terms.append({"name": name_cos, "vec": vec_cos})

        # s-based trig terms
        for w in w_s_list:
            w_str = self.fmt(w)
            sin_expr = f"sin({w_str}*{s_expr})"
            cos_expr = f"cos({w_str}*{s_expr})"
            for deg in range(self.D_s + 1):
                if deg == 0:
                    vec_sin = np.sin(w * s_arr)
                    vec_cos = np.cos(w * s_arr)
                    name_sin = sin_expr
                    name_cos = cos_expr
                    terms.append({"name": name_sin, "vec": vec_sin})
                    terms.append({"name": name_cos, "vec": vec_cos})
                else:
                    pow_vec = s_arr ** deg
                    pow_expr = f"({s_expr})**{deg}"
                    vec_sin = pow_vec * np.sin(w * s_arr)
                    vec_cos = pow_vec * np.cos(w * s_arr)
                    name_sin = f"{pow_expr}*{sin_expr}"
                    name_cos = f"{pow_expr}*{cos_expr}"
                    terms.append({"name": name_sin, "vec": vec_sin})
                    terms.append({"name": name_cos, "vec": vec_cos})

        return terms

    def _build_expression(self, intercept, coeffs, term_names):
        # Build a valid Python expression string
        expr = self.fmt(intercept)
        for c, name in zip(coeffs, term_names):
            if abs(c) < 1e-14:
                continue
            c_abs = self.fmt(abs(c))
            if c >= 0:
                expr += f" + {c_abs}*({name})"
            else:
                expr += f" - {c_abs}*({name})"
        return expr

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        x1 = X[:, 0]
        x2 = X[:, 1]
        s_arr = x1 * x1 + x2 * x2
        r_arr = np.sqrt(s_arr)

        # Default expression fallback in case of failure
        def fallback():
            # Simple baseline: ridge on [1, x1, x2, x1^2, x2^2, x1*x2, sin(5*r), cos(5*r), r*sin(5*r), r*cos(5*r), s, s^2]
            feats = []
            names = []
            # polynomial features
            feats.append(x1)
            names.append("x1")
            feats.append(x2)
            names.append("x2")
            feats.append(x1 * x1)
            names.append("(x1**2)")
            feats.append(x2 * x2)
            names.append("(x2**2)")
            feats.append(x1 * x2)
            names.append("(x1*x2)")
            feats.append(s_arr)
            names.append("(x1**2 + x2**2)")
            feats.append(s_arr ** 2)
            names.append("((x1**2 + x2**2)**2)")
            # ripple-like
            w = 5.0
            sr = np.sin(w * r_arr)
            cr = np.cos(w * r_arr)
            feats.append(sr)
            names.append(f"sin({self.fmt(w)}*((x1**2 + x2**2)**0.5))")
            feats.append(cr)
            names.append(f"cos({self.fmt(w)}*((x1**2 + x2**2)**0.5))")
            feats.append(r_arr * sr)
            names.append(f"((x1**2 + x2**2)**0.5)*sin({self.fmt(w)}*((x1**2 + x2**2)**0.5))")
            feats.append(r_arr * cr)
            names.append(f"((x1**2 + x2**2)**0.5)*cos({self.fmt(w)}*((x1**2 + x2**2)**0.5))")
            F = np.column_stack(feats)
            b, wcoef, yhat = self._fit_predict(F, y, self.alpha_ridge)
            expr = self._build_expression(b, wcoef, names)
            return expr, yhat

        try:
            # 1) Scan frequencies for r and s
            r_w_list = self._scan_freqs(
                y=y,
                base=r_arr,
                degrees=[0, 1],
                wmin=self.r_w_range[0],
                wmax=self.r_w_range[1],
                ngrid=self.r_w_points,
                top_k=self.top_k_r,
            )
            s_w_list = self._scan_freqs(
                y=y,
                base=s_arr,
                degrees=[0, 1],
                wmin=self.s_w_range[0],
                wmax=self.s_w_range[1],
                ngrid=self.s_w_points,
                top_k=self.top_k_s,
            )

            # 2) Build comprehensive term set
            terms = self._build_terms(X, r_w_list, s_w_list)

            # If no terms for some reason, fallback
            if not terms:
                expr, yhat = fallback()
                return {
                    "expression": expr,
                    "predictions": yhat.tolist(),
                    "details": {}
                }

            # 3) Assemble feature matrix
            F = np.column_stack([t["vec"] for t in terms])
            names = [t["name"] for t in terms]

            # 4) Ridge fit with intercept
            b_full, w_full, yhat_full = self._fit_predict(F, y, self.alpha_ridge)

            # 5) Prune by coefficient magnitude (on standardized space implicitly); use absolute weights
            # Rank by absolute coefficient magnitude
            abs_coef = np.abs(w_full)
            # Keep indices of largest coefficients up to max_terms
            if len(abs_coef) > self.max_terms:
                keep_idx = np.argsort(-abs_coef)[:self.max_terms]
                keep_idx = np.sort(keep_idx)
            else:
                keep_idx = np.arange(len(abs_coef))

            # Refit on pruned set for better coefficients
            F_pruned = F[:, keep_idx]
            names_pruned = [names[i] for i in keep_idx]
            b_pruned, w_pruned, yhat_pruned = self._fit_predict(F_pruned, y, self.alpha_ridge * 0.1)

            # 6) Build final expression
            expr = self._build_expression(b_pruned, w_pruned, names_pruned)

            predictions = yhat_pruned
            return {
                "expression": expr,
                "predictions": predictions.tolist(),
                "details": {}
            }
        except Exception:
            expr, yhat = fallback()
            return {
                "expression": expr,
                "predictions": yhat.tolist(),
                "details": {}
            }