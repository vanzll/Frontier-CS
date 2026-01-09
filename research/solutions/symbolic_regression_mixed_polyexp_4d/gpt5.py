import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.deg_p = kwargs.get("deg_p", 3)
        self.max_iters = kwargs.get("max_iters", 40)
        self.pruned_iters = kwargs.get("pruned_iters", 20)
        self.kp_keep = kwargs.get("kp_keep", 14)
        self.kq_keep = kwargs.get("kq_keep", 10)
        self.random_state = kwargs.get("random_state", 42)
        self._rng = np.random.default_rng(self.random_state)

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if d != 4:
            raise ValueError("X must have shape (n, 4)")
        # Build polynomial basis for P up to degree self.deg_p
        p_exps = self._generate_monomial_exponents(d=4, deg=self.deg_p)
        phi = self._evaluate_monomials(X, p_exps, self.deg_p)
        # Build Q basis (linear + quadratic: xi, xi**2, xi*xj)
        Q_mat, Q_names = self._build_Q_basis(X)
        # Initial optimization with full bases
        a0, c0 = self._optimize(phi, Q_mat, y, max_iters=self.max_iters)
        # Prune terms for compact expression
        p_keep_idx = self._select_top_indices_linear(phi, c0, k_keep=min(self.kp_keep, phi.shape[1]))
        q_keep_idx = self._select_top_indices_linear(Q_mat, a0, k_keep=min(self.kq_keep, Q_mat.shape[1]))
        # Ensure constant term present in P
        const_idx = self._get_constant_index(p_exps)
        if const_idx not in p_keep_idx:
            p_keep_idx = np.unique(np.concatenate([p_keep_idx, [const_idx]])).astype(int)
        # Restrict to pruned sets and refine
        phi_small = phi[:, p_keep_idx]
        Q_small = Q_mat[:, q_keep_idx]
        a_small, c_small = self._optimize(phi_small, Q_small, y, max_iters=self.pruned_iters, a_init=a0[q_keep_idx], c_init=c0[p_keep_idx])
        # Compose final expression string
        p_terms = [p_exps[i] for i in p_keep_idx]
        p_coefs = c_small
        q_names_sel = [Q_names[j] for j in q_keep_idx]
        q_coefs = a_small
        expression = self._compose_expression(p_terms, p_coefs, q_names_sel, q_coefs)
        # Compute predictions from final pruned model
        y_pred = self._predict_from_bases(phi_small, Q_small, c_small, a_small)
        return {
            "expression": expression,
            "predictions": y_pred.tolist(),
            "details": {}
        }

    def _generate_monomial_exponents(self, d: int, deg: int):
        exps = []
        def rec(idx, remaining, curr):
            if idx == d - 1:
                exps.append(tuple(curr + [remaining]))
                return
            for k in range(remaining + 1):
                rec(idx + 1, remaining - k, curr + [k])
        for total in range(deg + 1):
            rec(0, total, [])
        return exps

    def _evaluate_monomials(self, X, exps, max_deg):
        n, d = X.shape
        pow_cache = [[None]*(max_deg+1) for _ in range(d)]
        for j in range(d):
            xj = X[:, j]
            pow_cache[j][0] = np.ones(n, dtype=float)
            for k in range(1, max_deg+1):
                pow_cache[j][k] = pow_cache[j][k-1] * xj
        cols = []
        for e in exps:
            col = np.ones(n, dtype=float)
            for j in range(d):
                ej = e[j]
                if ej != 0:
                    col = col * pow_cache[j][ej]
            cols.append(col)
        if len(cols) == 0:
            return np.empty((n, 0), dtype=float)
        return np.column_stack(cols)

    def _build_Q_basis(self, X):
        # Linear: x1..x4
        # Quadratic: x_i**2, x_i*x_j for i<j
        n, d = X.shape
        cols = []
        names = []
        var_names = ["x1", "x2", "x3", "x4"]
        # linear
        for i in range(d):
            cols.append(X[:, i])
            names.append(var_names[i])
        # squares
        for i in range(d):
            cols.append(X[:, i] * X[:, i])
            names.append(f"{var_names[i]}**2")
        # cross terms
        for i in range(d):
            for j in range(i+1, d):
                cols.append(X[:, i] * X[:, j])
                names.append(f"{var_names[i]}*{var_names[j]}")
        Q_mat = np.column_stack(cols) if len(cols) > 0 else np.empty((n, 0), dtype=float)
        return Q_mat, names

    def _ridge_lstsq(self, A, b, lam=0.0):
        # Solve (A^T A + lam I) x = A^T b using augmented least squares for stability
        m = A.shape[1]
        if lam > 0:
            A_aug = np.vstack([A, np.sqrt(lam) * np.eye(m)])
            b_aug = np.concatenate([b, np.zeros(m)])
            x, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            return x
        else:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return x

    def _optimize(self, Phi, Q, y, max_iters=30, a_init=None, c_init=None):
        n = y.shape[0]
        k_p = Phi.shape[1]
        k_q = Q.shape[1]
        # Initial a, c
        a = np.zeros(k_q, dtype=float) if a_init is None else a_init.astype(float).copy()
        # Fit c with current a (initially zeros -> Q=0)
        Qv = Q @ a if k_q > 0 else np.zeros(n, dtype=float)
        yexp = y * np.exp(np.clip(Qv, -30.0, 30.0))
        # Ridge parameter for c scaled by column energy
        col_var = np.mean(Phi * Phi, axis=0) if k_p > 0 else np.array([1.0])
        lam_c = 1e-6 * (np.mean(col_var) * n + 1e-12)
        c = self._ridge_lstsq(Phi, yexp, lam=lam_c) if c_init is None else c_init.astype(float).copy()
        # Track best
        y_pred = self._predict_from_bases(Phi, Q, c, a)
        r = y_pred - y
        mse = float(np.mean(r * r))
        best = (mse, a.copy(), c.copy())
        no_improve = 0
        for it in range(max_iters):
            # Update c given a
            Qv = Q @ a if k_q > 0 else np.zeros(n, dtype=float)
            yexp = y * np.exp(np.clip(Qv, -30.0, 30.0))
            c = self._ridge_lstsq(Phi, yexp, lam=lam_c)
            # Evaluate current fit
            f = self._predict_from_bases(Phi, Q, c, a)
            r = f - y
            mse_prev = mse
            mse = float(np.mean(r * r))
            if mse < best[0] - 1e-14:
                best = (mse, a.copy(), c.copy())
                no_improve = 0
            else:
                no_improve += 1
            # Update a using Gauss-Newton with damping and line search
            if k_q > 0:
                # Compute Jacobian approximations
                # f = exp(-Qv) * (Phi @ c)
                P_lin = Phi @ c
                exp_neg_Q = np.exp(np.clip(-Qv, -50.0, 50.0))
                f = P_lin * exp_neg_Q
                r = f - y
                w = f * f  # weights for J^T J
                # Build B = Q^T (w * Q)
                WQ = Q * w[:, None]
                B_raw = Q.T @ Q if n < 1 else Q.T @ WQ
                # Regularize B
                traceB = float(np.trace(B_raw)) + 1e-12
                lam_a = 1e-8 * traceB / max(k_q, 1) + 1e-12
                B = B_raw + lam_a * np.eye(k_q)
                # s = Q^T (f * r)
                s = Q.T @ (f * r)
                try:
                    da = np.linalg.solve(B, s)
                except np.linalg.LinAlgError:
                    # Fall back to least squares
                    da, _, _, _ = np.linalg.lstsq(B, s, rcond=None)
                # Line search
                alpha = 1.0
                accepted = False
                for _ in range(8):
                    a_new = a + alpha * da
                    f_new = self._predict_from_bases(Phi, Q, c, a_new)
                    mse_new = float(np.mean((f_new - y) ** 2))
                    if mse_new <= mse + 1e-12:
                        a = a_new
                        mse = mse_new
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    # Small random jitter to escape flat regions
                    a = a + 1e-6 * self._rng.standard_normal(size=a.shape)
            # Early stopping
            rel_improve = (mse_prev - mse) / (mse_prev + 1e-12)
            if rel_improve < 1e-9:
                no_improve += 1
            if no_improve >= 5:
                break
        # Use best encountered
        _, a_best, c_best = best
        return a_best, c_best

    def _predict_from_bases(self, Phi, Q, c, a):
        n = Phi.shape[0]
        if Q.shape[1] == 0:
            Qv = np.zeros(n, dtype=float)
        else:
            Qv = Q @ a
        P_lin = Phi @ c
        exp_neg_Q = np.exp(np.clip(-Qv, -50.0, 50.0))
        return P_lin * exp_neg_Q

    def _select_top_indices_linear(self, A, coef, k_keep=10):
        # Score by |coef| * sqrt(mean(col^2))
        if A.shape[1] == 0:
            return np.array([], dtype=int)
        col_scale = np.sqrt(np.mean(A * A, axis=0) + 1e-18)
        scores = np.abs(coef) * col_scale
        if k_keep >= len(scores):
            idx = np.arange(len(scores))
        else:
            idx = np.argpartition(-scores, k_keep - 1)[:k_keep]
        # Sort by score descending for consistency
        idx = idx[np.argsort(-scores[idx])]
        return np.array(sorted(idx), dtype=int)

    def _get_constant_index(self, exps):
        for i, e in enumerate(exps):
            if all(v == 0 for v in e):
                return i
        return None

    def _compose_expression(self, p_terms, p_coefs, q_names, q_coefs):
        # Compose P(x) as sum_i c_i * mon_i
        p_expr = self._compose_poly_expr(p_terms, p_coefs)
        # Compose Q(x) as sum_j a_j * q_j
        q_expr = self._compose_q_expr(q_names, q_coefs)
        if q_expr.strip() == "":
            q_expr = "0"
        # Final expression f = P * exp(-(Q))
        expr = f"({p_expr})*exp(-({q_expr}))"
        return expr

    def _compose_poly_expr(self, terms, coefs):
        parts = []
        for e, c in zip(terms, coefs):
            if not np.isfinite(c) or abs(c) < 1e-14:
                continue
            mon = self._monomial_str(e)
            coef_str = self._format_number(c)
            if mon == "1":
                term_str = coef_str
            else:
                # optimize multiplication by 1 or -1
                if abs(abs(c) - 1.0) < 1e-12:
                    sign = "-" if c < 0 else ""
                    term_str = f"{sign}{mon}"
                else:
                    term_str = f"{coef_str}*{mon}"
            parts.append(term_str)
        if not parts:
            return "0"
        # Combine with +/-
        expr = parts[0]
        for t in parts[1:]:
            if t.startswith("-"):
                expr += " - " + t[1:]
            else:
                expr += " + " + t
        return expr

    def _compose_q_expr(self, names, coefs):
        parts = []
        for name, a in zip(names, coefs):
            if not np.isfinite(a) or abs(a) < 1e-14:
                continue
            a_str = self._format_number(a)
            if abs(abs(a) - 1.0) < 1e-12:
                sign = "-" if a < 0 else ""
                parts.append(f"{sign}{name}")
            else:
                parts.append(f"{a_str}*{name}")
        if not parts:
            return ""
        expr = parts[0]
        for t in parts[1:]:
            if t.startswith("-"):
                expr += " - " + t[1:]
            else:
                expr += " + " + t
        return expr

    def _monomial_str(self, e):
        vars_names = ["x1", "x2", "x3", "x4"]
        pieces = []
        for power, name in zip(e, vars_names):
            if power == 0:
                continue
            elif power == 1:
                pieces.append(name)
            else:
                pieces.append(f"{name}**{int(power)}")
        if not pieces:
            return "1"
        return "*".join(pieces)

    def _format_number(self, x):
        # Format with reasonable precision and avoid scientific notation for small integers
        return f"{float(x):.12g}"