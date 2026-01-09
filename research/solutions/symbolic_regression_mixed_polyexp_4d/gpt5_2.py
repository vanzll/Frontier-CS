import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 42)
        self.max_terms = kwargs.get("max_terms", 12)
        self.val_fraction = kwargs.get("val_fraction", 0.25)
        self.ridge_alpha = kwargs.get("ridge_alpha", 1e-12)

    def _build_exponents(self):
        # Generate monomials up to degree 3 with specified subset
        exps = []
        seen = set()

        def add_exp(e):
            t = tuple(e)
            if t not in seen:
                seen.add(t)
                exps.append(t)

        # Degree 0
        add_exp((0, 0, 0, 0))

        # Degree 1
        for i in range(4):
            e = [0, 0, 0, 0]
            e[i] = 1
            add_exp(e)

        # Degree 2: squares
        for i in range(4):
            e = [0, 0, 0, 0]
            e[i] = 2
            add_exp(e)
        # Degree 2: cross
        for i in range(4):
            for j in range(i + 1, 4):
                e = [0, 0, 0, 0]
                e[i] = 1
                e[j] = 1
                add_exp(e)

        # Degree 3: cubes
        for i in range(4):
            e = [0, 0, 0, 0]
            e[i] = 3
            add_exp(e)
        # Degree 3: x_i^2 * x_j
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                e = [0, 0, 0, 0]
                e[i] = 2
                e[j] = 1
                add_exp(e)
        # Degree 3: triple cross
        for i in range(4):
            for j in range(i + 1, 4):
                for k in range(j + 1, 4):
                    e = [0, 0, 0, 0]
                    e[i] = 1
                    e[j] = 1
                    e[k] = 1
                    add_exp(e)

        return exps

    def _monomial_strings(self, exps):
        # Returns list of strings representing each monomial
        var_names = ["x1", "x2", "x3", "x4"]
        mon_strs = []
        for e in exps:
            parts = []
            for i, p in enumerate(e):
                if p == 1:
                    parts.append(var_names[i])
                elif p == 2:
                    parts.append(f"{var_names[i]}**2")
                elif p == 3:
                    parts.append(f"{var_names[i]}**3")
            if len(parts) == 0:
                mon_strs.append("1")
            else:
                mon_strs.append("*".join(parts))
        return mon_strs

    def _compute_polynomial_features(self, X, exps):
        # Compute values of monomials for all samples
        n = X.shape[0]
        m = len(exps)
        P = np.ones((n, m), dtype=float)
        # Precompute powers for efficiency
        X1 = X
        X2 = X ** 2
        X3 = X ** 3
        for j, e in enumerate(exps):
            val = np.ones(n, dtype=float)
            for i, p in enumerate(e):
                if p == 1:
                    val *= X1[:, i]
                elif p == 2:
                    val *= X2[:, i]
                elif p == 3:
                    val *= X3[:, i]
                # p == 0: multiply by 1, no change
            P[:, j] = val
        return P

    def _safe_lstsq_ridge(self, A, y, alpha):
        # Solve (A^T A + alpha I)x = A^T y using lstsq via augmentation for numerical stability
        k = A.shape[1]
        if k == 0:
            return np.zeros(0, dtype=float)
        if alpha > 0:
            sqrt_alpha = np.sqrt(alpha)
            A_aug = np.vstack([A, sqrt_alpha * np.eye(k)])
            y_aug = np.concatenate([y, np.zeros(k, dtype=float)])
            coef, _, _, _ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        else:
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coef

    def _forward_selection(self, F_train, y_train, F_val, y_val, max_terms, alpha):
        n_features = F_train.shape[1]
        remaining = list(range(n_features))
        selected = []
        best_overall = {
            "indices": [],
            "coefs": np.zeros(0, dtype=float),
            "mse_val": float(np.mean((y_val - 0.0) ** 2))
        }
        current_best_val = best_overall["mse_val"]
        # Early stopping parameters
        tol = 1e-12

        for _ in range(max_terms):
            best_step = None
            best_step_mse = np.inf
            best_step_coefs = None
            for j in remaining:
                trial_idx = selected + [j]
                A_train = F_train[:, trial_idx]
                coefs = self._safe_lstsq_ridge(A_train, y_train, alpha)
                A_val = F_val[:, trial_idx]
                pred_val = A_val @ coefs
                mse_val = float(np.mean((y_val - pred_val) ** 2))
                if mse_val < best_step_mse - 1e-18:
                    best_step_mse = mse_val
                    best_step = j
                    best_step_coefs = coefs
            if best_step is None:
                break
            # Update selected
            selected.append(best_step)
            remaining.remove(best_step)
            # Update best overall
            if best_step_mse < best_overall["mse_val"] - tol:
                best_overall = {
                    "indices": list(selected),
                    "coefs": best_step_coefs.copy(),
                    "mse_val": best_step_mse
                }
                current_best_val = best_step_mse
            else:
                # If no improvement, we still continue a bit but we'll keep best_overall
                # However, to keep complexity low, break if no improvement
                break

        return best_overall

    def _linear_baseline(self, X, y, train_idx, val_idx):
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        A = np.column_stack([x1, x2, x3, x4, np.ones_like(x1)])
        A_train = A[train_idx]
        y_train = y[train_idx]
        coef, _, _, _ = np.linalg.lstsq(A_train, y_train, rcond=None)
        A_val = A[val_idx]
        y_val = y[val_idx]
        pred_val = A_val @ coef
        mse_val = float(np.mean((y_val - pred_val) ** 2))
        return coef, mse_val

    def _num_to_str(self, x):
        # Format number with reasonable precision
        if not np.isfinite(x):
            return "0"
        # Avoid negative zero
        if abs(x) < 1e-15:
            return "0"
        return f"{x:.12g}"

    def _build_polynomial_str(self, coefs, indices, monomial_strs):
        # Build a human-friendly polynomial string sum of terms with proper signs
        if len(indices) == 0:
            return "0"
        terms = []
        for c, idx in zip(coefs, indices):
            if abs(c) < 1e-15:
                continue
            mono = monomial_strs[idx]
            c_abs = abs(c)
            # Attempt to simplify when coef close to 1
            if mono == "1":
                term = self._num_to_str(c)
            else:
                if abs(c - 1.0) < 1e-12:
                    term = mono
                elif abs(c + 1.0) < 1e-12:
                    term = f"-{mono}"
                else:
                    term = f"{self._num_to_str(c_abs)}*{mono}"
            terms.append((c >= 0, term))

        if not terms:
            return "0"

        # Construct string with signs
        first_sign, first_term = terms[0]
        poly = first_term if first_sign else f"-{first_term}"
        for sign, term in terms[1:]:
            if sign:
                poly += f" + {term}"
            else:
                poly += f" - {term}"
        return poly

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)

        # Build monomial exponents and strings
        exps = self._build_exponents()
        monomial_strs = self._monomial_strings(exps)

        # Compute polynomial feature matrix P (n, m)
        P = self._compute_polynomial_features(X, exps)

        # Split into train/val
        idx = np.arange(n)
        rng.shuffle(idx)
        val_size = max(1, int(self.val_fraction * n))
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        if len(train_idx) < 5:  # ensure minimal training points
            train_idx = idx
            val_idx = idx[:0]

        y_train = y[train_idx]
        y_val = y[val_idx] if len(val_idx) > 0 else np.array([], dtype=float)

        # Compute S = sum of squares
        S_all = np.sum(X ** 2, axis=1)
        mean_S = float(np.mean(S_all)) if n > 0 else 1.0
        if mean_S <= 1e-12:
            gamma_base = 1.0
        else:
            gamma_base = 1.0 / mean_S

        gamma_candidates = [0.0] + [gamma_base * g for g in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]]
        # Clamp gammas to sensible range
        gamma_candidates = [g for g in gamma_candidates if 0.0 <= g <= 5.0]
        # Deduplicate with rounding
        gamma_candidates = sorted(list({round(g, 12) for g in gamma_candidates}))

        best_model = {
            "gamma": None,
            "indices": [],
            "coefs": np.zeros(0, dtype=float),
            "mse_val": float("inf")
        }

        # Forward selection for each gamma
        for gamma in gamma_candidates:
            if gamma == 0.0:
                E_all = np.ones(n, dtype=float)
            else:
                E_all = np.exp(-gamma * S_all)
            F_all = P * E_all[:, None]

            F_train = F_all[train_idx]
            F_val = F_all[val_idx] if len(val_idx) > 0 else np.zeros((0, F_all.shape[1]), dtype=float)

            result = self._forward_selection(F_train, y_train, F_val, y_val, self.max_terms, self.ridge_alpha)
            mse_val = result["mse_val"] if len(val_idx) > 0 else float(np.mean((y - (F_all[:, result["indices"]] @ self._safe_lstsq_ridge(F_all[:, result["indices"]], y, self.ridge_alpha))) ** 2)) if len(result["indices"]) > 0 else float(np.mean(y ** 2))

            if mse_val < best_model["mse_val"]:
                best_model = {
                    "gamma": gamma,
                    "indices": result["indices"],
                    "coefs": result["coefs"],
                    "mse_val": mse_val
                }

        # Linear baseline for robustness
        coef_lin, mse_lin = self._linear_baseline(X, y, train_idx, val_idx)
        if mse_lin < best_model["mse_val"]:
            # Build linear expression and predictions
            a, b, c, d_lin, e = coef_lin
            expression = f"{self._num_to_str(a)}*x1 + {self._num_to_str(b)}*x2 + {self._num_to_str(c)}*x3 + {self._num_to_str(d_lin)}*x4 + {self._num_to_str(e)}"
            predictions = (X @ np.array([a, b, c, d_lin]) + e).tolist()
            return {
                "expression": expression,
                "predictions": predictions,
                "details": {}
            }

        # Refit best model on full data to get final coefficients
        gamma = best_model["gamma"]
        if gamma == 0.0:
            E_all = np.ones(n, dtype=float)
        else:
            E_all = np.exp(-gamma * S_all)
        F_all = P * E_all[:, None]

        indices = best_model["indices"]
        if len(indices) == 0:
            # Fallback to constant mean model if selection failed
            c0 = float(np.mean(y))
            expression = self._num_to_str(c0)
            predictions = (np.ones(n) * c0).tolist()
            return {
                "expression": expression,
                "predictions": predictions,
                "details": {}
            }

        A_full = F_all[:, indices]
        coefs_full = self._safe_lstsq_ridge(A_full, y, self.ridge_alpha)

        # Prune very small coefficients to reduce complexity
        keep_mask = np.array([abs(c) >= 1e-12 for c in coefs_full])
        if not np.any(keep_mask):
            # If everything is pruned, keep largest magnitude term
            max_idx = int(np.argmax(np.abs(coefs_full)))
            keep_mask[max_idx] = True

        indices = [indices[i] for i in range(len(indices)) if keep_mask[i]]
        coefs_full = coefs_full[keep_mask]

        # Build expression string
        poly_str = self._build_polynomial_str(coefs_full, indices, monomial_strs)
        if gamma == 0.0:
            expression = poly_str
            predictions = (P[:, indices] @ coefs_full).tolist()
        else:
            S_expr = "x1**2 + x2**2 + x3**2 + x4**2"
            gamma_str = self._num_to_str(gamma)
            # If polynomial simplifies to single term, we can avoid parentheses
            if poly_str.strip() == "0":
                expression = "0"
            else:
                expression = f"exp(-{gamma_str}*({S_expr})) * ({poly_str})"
            predictions = (np.exp(-gamma * S_all) * (P[:, indices] @ coefs_full)).tolist()

        return {
            "expression": expression,
            "predictions": predictions,
            "details": {}
        }