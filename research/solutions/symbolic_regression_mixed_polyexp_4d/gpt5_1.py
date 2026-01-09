import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 42)
        self.max_poly_deg = kwargs.get("max_poly_deg", 3)  # for plain polynomial
        self.iso_max_deg = kwargs.get("iso_max_deg", 2)    # degree for isotropic-exp terms
        self.gamma_iso = kwargs.get("gamma_iso", [0.5, 1.0, 2.0])
        self.gamma_pair = kwargs.get("gamma_pair", [0.5, 1.5])
        self.gamma_single = kwargs.get("gamma_single", [0.7, 1.5])
        self.lambda_grid = kwargs.get("lambda_grid", np.logspace(-8, 2, 15))
        self.k_candidates = kwargs.get("k_candidates", [6, 8, 10, 12, 15, 20, 25, 30, 35])
        self.max_terms = kwargs.get("max_terms", 35)
        self.zero_tol = 1e-12

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if d != 4:
            # Expect 4 features; if not, truncate or pad with zeros
            if d > 4:
                X = X[:, :4]
            else:
                X = np.column_stack([X, np.zeros((n, 4 - d))])
        # Standardize features to build z
        mu = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        sigma[sigma == 0] = 1.0
        Z = (X - mu) / sigma
        z = Z
        z2 = z ** 2
        z3 = z2 * z

        # Build features and specs
        features, specs = self._build_features(z, z2, z3)

        # Convert features to design matrix
        Phi = np.column_stack(features) if features else np.ones((n, 1))
        # Remove near-constant columns except the constant term (assumed at index 0)
        # Determine constant index (first feature is constant by construction)
        const_idx = 0
        col_std = np.std(Phi, axis=0)
        keep_mask = np.ones(Phi.shape[1], dtype=bool)
        for j in range(Phi.shape[1]):
            if j != const_idx and col_std[j] < 1e-14:
                keep_mask[j] = False
        Phi = Phi[:, keep_mask]
        specs = [specs[i] for i in range(len(specs)) if keep_mask[i]]

        # Ridge regression with holdout for lambda selection
        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = max(1, int(0.8 * n))
        train_idx = idx[:cut]
        val_idx = idx[cut:] if cut < n else idx[:1]  # ensure non-empty
        Phi_tr, y_tr = Phi[train_idx], y[train_idx]
        Phi_va, y_va = Phi[val_idx], y[val_idx]

        G = Phi_tr.T @ Phi_tr
        t = Phi_tr.T @ y_tr

        best_lam = None
        best_mse = np.inf
        for lam in self.lambda_grid:
            try:
                w = np.linalg.solve(G + lam * np.eye(G.shape[0]), t)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(G + lam * np.eye(G.shape[0]), t, rcond=None)[0]
            y_pred_va = Phi_va @ w
            mse = np.mean((y_va - y_pred_va) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_lam = lam

        # Fit ridge on all data with best lambda
        G_full = Phi.T @ Phi
        t_full = Phi.T @ y
        lam = best_lam if best_lam is not None else 1e-6
        try:
            w_full = np.linalg.solve(G_full + lam * np.eye(G_full.shape[0]), t_full)
        except np.linalg.LinAlgError:
            w_full = np.linalg.lstsq(G_full + lam * np.eye(G_full.shape[0]), t_full, rcond=None)[0]

        # Feature selection by importance and validation for K
        col_std_full = np.sqrt(np.mean(Phi ** 2, axis=0))
        importance = np.abs(w_full) * col_std_full
        # Always include constant term
        importance_with_const = importance.copy()
        importance_with_const[const_idx] = np.max(importance_with_const) + 1.0

        # Order indices by importance
        order = np.argsort(-importance_with_const)

        # Evaluate k candidates
        best_k = None
        best_k_mse = np.inf
        for k in self.k_candidates:
            k_eff = min(k, Phi.shape[1])
            sel_idx = np.unique(np.concatenate(([const_idx], order[:k_eff])))
            Phi_tr_k = Phi_tr[:, sel_idx]
            Phi_va_k = Phi_va[:, sel_idx]
            # Fit OLS on training subset
            w_k = np.linalg.lstsq(Phi_tr_k, y_tr, rcond=None)[0]
            y_pred_va_k = Phi_va_k @ w_k
            mse_k = np.mean((y_va - y_pred_va_k) ** 2)
            if mse_k < best_k_mse:
                best_k_mse = mse_k
                best_k = k_eff

        # Final selected indices and refit on all data
        best_k = min(best_k if best_k is not None else 20, self.max_terms, Phi.shape[1])
        sel_idx = np.unique(np.concatenate(([const_idx], order[:best_k])))
        Phi_sel = Phi[:, sel_idx]
        w_sel = np.linalg.lstsq(Phi_sel, y, rcond=None)[0]
        selected_specs = [specs[i] for i in sel_idx]
        selected_coefs = w_sel

        # Build expression string
        expr = self._build_expression(selected_specs, selected_coefs, mu, sigma)

        # Predictions
        y_pred = Phi_sel @ w_sel

        return {
            "expression": expr,
            "predictions": y_pred.tolist(),
            "details": {}
        }

    def _build_features(self, z, z2, z3):
        n, d = z.shape
        # z variables expected of length 4
        features = []
        specs = []

        # Helper to compute monomial values for specified variables and exponents
        def monomial_val(exps, var_idx=None):
            if var_idx is None:
                var_idx = list(range(z.shape[1]))
            res = np.ones(z.shape[0])
            for e, i in zip(exps, var_idx):
                if e == 0:
                    continue
                elif e == 1:
                    res = res * z[:, i]
                elif e == 2:
                    res = res * z2[:, i]
                elif e == 3:
                    res = res * z3[:, i]
                else:
                    # Higher degree not expected; compute via power
                    res = res * (z[:, i] ** e)
            return res

        # Generate exponent tuples
        def gen_exponents(nvars, max_deg, min_deg=0):
            result = []
            def rec(pos, deg_left, cur):
                if pos == nvars:
                    result.append(tuple(cur))
                    return
                for e in range(deg_left + 1):
                    cur.append(e)
                    rec(pos + 1, deg_left - e, cur)
                    cur.pop()
            for total in range(min_deg, max_deg + 1):
                rec(0, total, [])
            return result

        # z variable strings placeholders for specs
        z_varnames = ["z1", "z2", "z3", "z4"]

        # Constant term (none group)
        features.append(np.ones(z.shape[0]))
        specs.append({
            "group": ("none",),
            "monomial_vars": [],
            "monomial_exps": [],
            "monomial_str": "1"
        })

        # Plain polynomial terms up to degree 3 excluding constant
        poly_exps = gen_exponents(4, self.max_poly_deg, min_deg=1)
        for exps in poly_exps:
            col = monomial_val(exps, var_idx=[0, 1, 2, 3])
            mon_str = self._monomial_str_from_exps(exps, z_varnames)
            features.append(col)
            specs.append({
                "group": ("none",),
                "monomial_vars": [0, 1, 2, 3],
                "monomial_exps": list(exps),
                "monomial_str": mon_str
            })

        # Isotropic Gaussian factors: exp(-gamma * (sum zi^2))
        r2_iso = z2.sum(axis=1)
        for gamma in self.gamma_iso:
            E = np.exp(-gamma * r2_iso)
            # monomials up to degree 2 including constant
            iso_exps = gen_exponents(4, self.iso_max_deg, min_deg=0)
            for exps in iso_exps:
                mval = monomial_val(exps, var_idx=[0, 1, 2, 3])
                col = mval * E
                mon_str = self._monomial_str_from_exps(exps, z_varnames)
                features.append(col)
                specs.append({
                    "group": ("iso", float(gamma)),
                    "monomial_vars": [0, 1, 2, 3],
                    "monomial_exps": list(exps),
                    "monomial_str": mon_str
                })

        # Pairwise Gaussian factors: for each pair
        pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        pair_exps = self._gen_pair_exponents(max_deg=3)
        for (i, j) in pairs:
            r2_pair = z2[:, i] + z2[:, j]
            for gamma in self.gamma_pair:
                E = np.exp(-gamma * r2_pair)
                for ex_i, ex_j in pair_exps:
                    mval = np.ones(z.shape[0])
                    if ex_i == 1:
                        mval = mval * z[:, i]
                    elif ex_i == 2:
                        mval = mval * z2[:, i]
                    elif ex_i == 3:
                        mval = mval * z3[:, i]
                    # ex_i == 0: pass
                    if ex_j == 1:
                        mval = mval * z[:, j]
                    elif ex_j == 2:
                        mval = mval * z2[:, j]
                    elif ex_j == 3:
                        mval = mval * z3[:, j]
                    col = mval * E
                    mon_str = self._monomial_str_from_exps_pair(ex_i, ex_j, z_varnames[i], z_varnames[j])
                    features.append(col)
                    specs.append({
                        "group": ("pair", i, j, float(gamma)),
                        "monomial_vars": [i, j],
                        "monomial_exps": [ex_i, ex_j],
                        "monomial_str": mon_str
                    })

        # Single-variable Gaussian factors
        for i in range(4):
            zi = z[:, i]
            zi2 = z2[:, i]
            zi3 = z3[:, i]
            for gamma in self.gamma_single:
                E = np.exp(-gamma * zi2)
                # three forms: 1*E, zi*E, zi**2 * E
                # 1*E
                features.append(E.copy())
                specs.append({
                    "group": ("single", i, float(gamma)),
                    "monomial_vars": [i],
                    "monomial_exps": [0],
                    "monomial_str": "1"
                })
                # zi*E
                features.append(zi * E)
                specs.append({
                    "group": ("single", i, float(gamma)),
                    "monomial_vars": [i],
                    "monomial_exps": [1],
                    "monomial_str": z_varnames[i]
                })
                # zi**2 * E
                features.append(zi2 * E)
                specs.append({
                    "group": ("single", i, float(gamma)),
                    "monomial_vars": [i],
                    "monomial_exps": [2],
                    "monomial_str": f"{z_varnames[i]}**2"
                })

        return features, specs

    def _gen_pair_exponents(self, max_deg=3):
        # Generate all (e_i, e_j) with e_i, e_j >=0 and e_i+e_j <= max_deg
        pairs = []
        for total in range(0, max_deg + 1):
            for e_i in range(0, total + 1):
                e_j = total - e_i
                pairs.append((e_i, e_j))
        return pairs

    def _monomial_str_from_exps(self, exps, znames):
        # exps length equal to len(znames)
        terms = []
        for e, zn in zip(exps, znames):
            if e == 0:
                continue
            elif e == 1:
                terms.append(zn)
            else:
                terms.append(f"{zn}**{int(e)}")
        if not terms:
            return "1"
        return "*".join(terms)

    def _monomial_str_from_exps_pair(self, ei, ej, zi_name, zj_name):
        terms = []
        if ei == 1:
            terms.append(zi_name)
        elif ei >= 2:
            terms.append(f"{zi_name}**{int(ei)}")
        if ej == 1:
            terms.append(zj_name)
        elif ej >= 2:
            terms.append(f"{zj_name}**{int(ej)}")
        if not terms:
            return "1"
        return "*".join(terms)

    def _fmt(self, x):
        if abs(x) < 1e-15:
            return "0"
        s = f"{float(x):.12g}"
        return s

    def _z_str(self, var_index, mu, sigma):
        xi = f"x{var_index+1}"
        m = self._fmt(mu[var_index])
        s = self._fmt(sigma[var_index])
        return f"(({xi} - {m})/{s})"

    def _r2_iso_str(self, mu, sigma):
        parts = []
        for i in range(4):
            zi = self._z_str(i, mu, sigma)
            parts.append(f"({zi})**2")
        return " + ".join(parts)

    def _r2_pair_str(self, i, j, mu, sigma):
        zi = self._z_str(i, mu, sigma)
        zj = self._z_str(j, mu, sigma)
        return f"({zi})**2 + ({zj})**2"

    def _r2_single_str(self, i, mu, sigma):
        zi = self._z_str(i, mu, sigma)
        return f"({zi})**2"

    def _substitute_z_in_monomial(self, monomial_str, mu, sigma):
        # Replace z1..z4 with their expressions
        out = monomial_str
        for i in range(4):
            zi_token = f"z{i+1}"
            zi_expr = self._z_str(i, mu, sigma)
            out = out.replace(zi_token, f"({zi_expr})")
        return out

    def _build_expression(self, specs, coefs, mu, sigma):
        # Group terms by factor
        groups = {}
        for spec, c in zip(specs, coefs):
            if abs(c) < self.zero_tol:
                continue
            key = spec["group"]
            if key not in groups:
                groups[key] = []
            groups[key].append((c, spec["monomial_str"]))

        # Build strings for each group
        expr_parts = []

        # Helper to construct polynomial string in z substituted by x
        def poly_str(terms):
            # terms: list of (coef, monomial_str in z)
            if not terms:
                return ""
            # Sort by absolute coefficient descending
            terms_sorted = sorted(terms, key=lambda t: -abs(t[0]))
            poly = ""
            first = True
            for c, mstr in terms_sorted:
                if abs(c) < self.zero_tol:
                    continue
                cmag = abs(c)
                cstr = self._fmt(cmag)
                mon_str = mstr
                mon_z = self._substitute_z_in_monomial(mon_str, mu, sigma)
                term = ""
                if mon_str == "1":
                    term = f"{cstr}"
                else:
                    term = f"{cstr}*{mon_z}"
                if first:
                    sign = "-" if c < 0 else ""
                    poly += sign + term
                    first = False
                else:
                    sign = " - " if c < 0 else " + "
                    poly += sign + term
            if poly == "":
                poly = "0"
            return poly

        # Build 'none' group first (pure polynomial)
        none_key = ("none",)
        if none_key in groups:
            poly = poly_str(groups[none_key])
            if poly != "" and poly != "0":
                expr_parts.append(poly)

        # Isotropic groups
        for key in sorted([k for k in groups.keys() if len(k) >= 1 and k[0] == "iso"], key=lambda k: k[1]):
            gamma = key[1]
            poly = poly_str(groups[key])
            if poly == "" or poly == "0":
                continue
            r2 = self._r2_iso_str(mu, sigma)
            gstr = self._fmt(gamma)
            factor = f"exp(-{gstr}*({r2}))"
            expr_parts.append(f"({poly})*{factor}")

        # Pair groups
        pair_keys = [k for k in groups.keys() if len(k) >= 1 and k[0] == "pair"]
        # Sort by (i,j,gamma)
        pair_keys = sorted(pair_keys, key=lambda k: (k[1], k[2], k[3]))
        for key in pair_keys:
            i, j, gamma = key[1], key[2], key[3]
            poly = poly_str(groups[key])
            if poly == "" or poly == "0":
                continue
            r2 = self._r2_pair_str(i, j, mu, sigma)
            gstr = self._fmt(gamma)
            factor = f"exp(-{gstr}*({r2}))"
            expr_parts.append(f"({poly})*{factor}")

        # Single groups
        single_keys = [k for k in groups.keys() if len(k) >= 1 and k[0] == "single"]
        single_keys = sorted(single_keys, key=lambda k: (k[1], k[2]))
        for key in single_keys:
            i, gamma = key[1], key[2]
            poly = poly_str(groups[key])
            if poly == "" or poly == "0":
                continue
            r2 = self._r2_single_str(i, mu, sigma)
            gstr = self._fmt(gamma)
            factor = f"exp(-{gstr}*({r2}))"
            expr_parts.append(f"({poly})*{factor}")

        if not expr_parts:
            # Fallback to zero expression
            return "0"

        # Combine parts
        expression = expr_parts[0]
        for part in expr_parts[1:]:
            # Handle leading sign of part
            if part.startswith("-"):
                expression += " - " + part[1:]
            else:
                expression += " + " + part
        return expression