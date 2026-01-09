import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 0)
        self.max_degree = kwargs.get("max_degree", 3)
        self.kmax = kwargs.get("kmax", None)
        self.val_ratio = kwargs.get("val_ratio", 0.2)
        self.patience = kwargs.get("patience", 5)
        self.tol = kwargs.get("tol", 1e-9)

    def _format_number(self, x):
        if not np.isfinite(x):
            return "0"
        if abs(x) < 1e-15:
            return "0"
        s = f"{x:.12g}"
        # Clean up trailing decimal point
        if s.endswith("."):
            s = s[:-1]
        return s

    def _train_val_split(self, n, random_state, val_ratio):
        rng = np.random.RandomState(random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(n * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        if train_idx.size == 0:
            # fallback: use all as train and val identical
            train_idx = idx
            val_idx = idx
        return train_idx, val_idx

    def _build_polynomial_features(self, X):
        n = X.shape[0]
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        P_list = []
        E_list = []

        # Constant
        P_list.append(np.ones(n))
        E_list.append("1")

        # Degree 1
        P_list.extend([x1, x2, x3, x4])
        E_list.extend(["x1", "x2", "x3", "x4"])

        # Degree 2: squares
        P_list.extend([x1 * x1, x2 * x2, x3 * x3, x4 * x4])
        E_list.extend(["x1**2", "x2**2", "x3**2", "x4**2"])

        # Degree 2: cross
        P_list.extend([
            x1 * x2, x1 * x3, x1 * x4,
            x2 * x3, x2 * x4,
            x3 * x4
        ])
        E_list.extend([
            "x1*x2", "x1*x3", "x1*x4",
            "x2*x3", "x2*x4",
            "x3*x4"
        ])

        if self.max_degree >= 3:
            # Degree 3: cubes
            P_list.extend([x1 * x1 * x1, x2 * x2 * x2, x3 * x3 * x3, x4 * x4 * x4])
            E_list.extend(["x1**3", "x2**3", "x3**3", "x4**3"])

            # Degree 3: squared * other
            xs = [x1, x2, x3, x4]
            for i in range(4):
                for j in range(4):
                    if i == j:
                        continue
                    P_list.append(xs[i] * xs[i] * xs[j])
                    E_list.append(f"x{i+1}**2*x{j+1}")

            # Degree 3: triple cross
            P_list.append(x1 * x2 * x3)
            E_list.append("x1*x2*x3")
            P_list.append(x1 * x2 * x4)
            E_list.append("x1*x2*x4")
            P_list.append(x1 * x3 * x4)
            E_list.append("x1*x3*x4")
            P_list.append(x2 * x3 * x4)
            E_list.append("x2*x3*x4")

        P = np.column_stack(P_list)
        return P, E_list

    def _build_gaussians(self, X):
        n = X.shape[0]
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        S = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4

        # Always include identity
        G_list = [np.ones(n)]
        G_exprs = ["1"]

        # Global Gaussian scales
        sscale = float(np.mean(S)) if np.isfinite(np.mean(S)) else 1.0
        sscale = max(sscale, 1e-12)
        base = 1.0 / sscale

        # Candidate factors for alpha
        factors_sum = [0.25, 0.5, 1.0]
        alphas_sum = []
        for f in factors_sum:
            a = f * base
            # clamp to reasonable range
            a = float(np.clip(a, 1e-6, 50.0))
            if a not in alphas_sum:
                alphas_sum.append(a)

        for a in alphas_sum:
            g = np.exp(-a * S)
            if np.var(g) > 1e-14:
                G_list.append(g)
                a_str = self._format_number(a)
                G_exprs.append(f"exp(-{a_str}*(x1**2 + x2**2 + x3**2 + x4**2))")

        # Per-dimension Gaussians
        xs = [x1, x2, x3, x4]
        for i, xi in enumerate(xs):
            vi = float(np.mean(xi * xi)) if np.isfinite(np.mean(xi * xi)) else 1.0
            vi = max(vi, 1e-12)
            base_i = 1.0 / vi
            factors_each = [0.5, 1.0]
            used_ai = set()
            for f in factors_each:
                a = f * base_i
                a = float(np.clip(a, 1e-6, 50.0))
                if a in used_ai:
                    continue
                used_ai.add(a)
                g = np.exp(-a * xi * xi)
                if np.var(g) > 1e-14:
                    G_list.append(g)
                    a_str = self._format_number(a)
                    G_exprs.append(f"exp(-{a_str}*x{i+1}**2)")

        G = np.column_stack(G_list)
        return G, G_exprs

    def _build_dictionary(self, X):
        P, P_exprs = self._build_polynomial_features(X)
        G, G_exprs = self._build_gaussians(X)

        n = X.shape[0]
        F_cols = []
        terms = []
        # Construct all products: P_j * G_k
        for k, g_expr in enumerate(G_exprs):
            g_col = G[:, k]
            for j, p_expr in enumerate(P_exprs):
                col = P[:, j] * g_col
                # Remove near-constant zero-variance columns
                col_var = float(np.var(col))
                col_norm = float(np.sqrt(np.sum(col * col)))
                if not np.isfinite(col_norm) or col_norm < 1e-12 or col_var < 1e-18:
                    continue
                if g_expr == "1":
                    expr = p_expr
                else:
                    if p_expr == "1":
                        expr = g_expr
                    else:
                        expr = f"({p_expr})*{g_expr}"
                F_cols.append(col)
                terms.append(expr)

        if len(F_cols) == 0:
            # fallback: constant only
            F = np.ones((n, 1))
            terms = ["1"]
        else:
            F = np.column_stack(F_cols)
        return F, terms

    def _omp_select(self, F, y, kmax, random_state, val_ratio, tol, patience):
        n, p = F.shape
        train_idx, val_idx = self._train_val_split(n, random_state, val_ratio)
        Ft = F[train_idx, :]
        yt = y[train_idx]
        Fv = F[val_idx, :] if val_idx.size > 0 else None
        yv = y[val_idx] if val_idx.size > 0 else None

        # Normalize columns by L2 norm on training set
        norms = np.sqrt(np.sum(Ft * Ft, axis=0)) + 1e-12
        U_train = Ft / norms
        U_val = (Fv / norms) if Fv is not None else None

        residual = yt.copy()
        selected = []
        best_val_mse = np.inf
        best_step = -1
        best_w = None
        no_improve = 0

        # Pre-allocate correlations buffer
        for step in range(1, kmax + 1):
            # Correlation
            corr = U_train.T @ residual
            # Mask already selected indices by zeroing
            if selected:
                corr[selected] = 0.0
            j = int(np.argmax(np.abs(corr)))
            if np.abs(corr[j]) < 1e-12:
                break
            selected.append(j)

            A = U_train[:, selected]
            # Solve least squares
            w, _, _, _ = np.linalg.lstsq(A, yt, rcond=None)
            residual = yt - A @ w

            # Validation error
            if U_val is not None:
                yv_pred = U_val[:, selected] @ w
                val_mse = float(np.mean((yv - yv_pred) ** 2))
            else:
                # If no validation set, use training error as proxy
                val_mse = float(np.mean(residual ** 2))

            if val_mse < best_val_mse - tol:
                best_val_mse = val_mse
                best_step = step
                best_w = w.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_step <= 0:
            # If nothing selected, pick the single best feature
            norms = np.sqrt(np.sum(F * F, axis=0)) + 1e-12
            U_all = F / norms
            corr_all = np.abs(U_all.T @ y)
            j = int(np.argmax(corr_all))
            selected = [j]
            best_w = np.array([float((U_all[:, j].T @ y) / (U_all[:, j].T @ U_all[:, j] + 1e-12))])
        else:
            selected = selected[:best_step]

        # Convert weights from normalized to original scale for later refit
        return selected

    def _refit_on_full(self, F, y, selected, ridge=0.0):
        A = F[:, selected]
        if ridge > 0:
            # Solve (A^T A + lam I) w = A^T y
            ATA = A.T @ A
            ATy = A.T @ y
            ATA.flat[:: ATA.shape[0] + 1] += ridge
            w = np.linalg.solve(ATA, ATy)
        else:
            w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return w

    def _build_expression(self, coefs, terms):
        # Build expression string given coefficients and term strings
        expr_parts = []
        for c, t in zip(coefs, terms):
            if abs(c) < 1e-12:
                continue
            c_str = self._format_number(c)
            if t == "1":
                term_str = f"{c_str}"
            else:
                # If coefficient is 1 or -1, avoid printing "*"
                if c_str == "1":
                    term_str = f"{t}"
                elif c_str == "-1":
                    term_str = f"-({t})"
                else:
                    term_str = f"{c_str}*{t}"
            expr_parts.append(term_str)

        if not expr_parts:
            return "0"
        # Combine, handling signs to avoid "+ -"
        expr = expr_parts[0]
        for part in expr_parts[1:]:
            if part.startswith("-"):
                expr += " - " + part[1:] if len(part) > 1 else " - 0"
            else:
                expr += " + " + part
        return expr

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if d != 4:
            # If not 4D, fallback to simple linear model
            A = np.column_stack([X, np.ones(n)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b, c, d_, e = coeffs
            expression = f"{self._format_number(a)}*x1 + {self._format_number(b)}*x2 + {self._format_number(c)}*x3 + {self._format_number(d_)}*x4 + {self._format_number(e)}"
            predictions = (A @ coeffs)
            return {
                "expression": expression,
                "predictions": predictions.tolist(),
                "details": {}
            }

        # Build dictionary of features
        F, term_exprs = self._build_dictionary(X)

        # Determine kmax adaptively if not provided
        if self.kmax is None:
            if n < 120:
                kmax = 12
            elif n < 600:
                kmax = 18
            else:
                kmax = 24
        else:
            kmax = int(self.kmax)
            if kmax <= 0:
                kmax = 12

        # OMP selection
        selected = self._omp_select(F, y, kmax, self.random_state, self.val_ratio, self.tol, self.patience)

        # Refit on full data for selected features
        coefs = self._refit_on_full(F, y, selected, ridge=0.0)

        # Optional pruning of tiny coefficients
        if len(selected) > 1:
            A_sel = F[:, selected]
            contrib = np.abs(coefs) * np.sqrt(np.mean(A_sel ** 2, axis=0))
            # Keep features with sufficient contribution
            if np.any(contrib > 0):
                thresh = max(np.max(contrib) * 1e-4, 1e-12)
                keep_mask = contrib >= thresh
                if not np.all(keep_mask):
                    selected = [selected[i] for i, k in enumerate(keep_mask) if k]
                    if len(selected) == 0:
                        # keep the strongest one
                        idx_max = int(np.argmax(contrib))
                        selected = [selected[idx_max]]
                    coefs = self._refit_on_full(F, y, selected, ridge=0.0)

        # Build expression
        terms_final = [term_exprs[j] for j in selected]
        expression = self._build_expression(coefs, terms_final)

        # Predictions
        predictions = F[:, selected] @ coefs

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": {}
        }