import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.max_terms = int(kwargs.get("max_terms", 28))
        self.random_state = int(kwargs.get("random_state", 42))
        self.patience = int(kwargs.get("patience", 6))
        self.ridge_lambda = float(kwargs.get("ridge_lambda", 1e-8))
        self.include_exp = bool(kwargs.get("include_exp", True))

    def _freq_lists(self, x1, x2):
        pi = np.pi
        # Radial frequencies capturing a wide set including multiples of pi
        w_rad = [
            0.5, 1.0, 1.5, 2.0, 3.0,
            4.0, 5.0, 6.0, 8.0, 10.0,
            pi/2, pi, 1.5*pi, 2*pi, 3*pi, 4*pi
        ]
        # Axis frequencies (smaller set)
        w_axis = [0.5, 1.0, 2.0, 3.0, pi, 2*pi, 3*pi]
        # Cross frequencies (very small set)
        w_cross = [1.0, 2.0, pi]
        # r^2 frequencies (small set)
        w_r2 = [0.5, 1.0, pi/2, pi]
        # Deduplicate while preserving order
        def unique(seq):
            seen = set()
            out = []
            for v in seq:
                if v not in seen:
                    out.append(v)
                    seen.add(v)
            return out
        return unique(w_rad), unique(w_axis), unique(w_cross), unique(w_r2)

    def _add_feature(self, name, values, names, cols):
        if name in names:
            return
        if not np.all(np.isfinite(values)):
            return
        # Avoid near-constant columns
        if np.nanstd(values) < 1e-12:
            return
        names.append(name)
        cols.append(values)

    def _generate_features(self, X):
        n = X.shape[0]
        x1 = X[:, 0].astype(float, copy=False)
        x2 = X[:, 1].astype(float, copy=False)
        r2 = x1 * x1 + x2 * x2
        r = np.sqrt(np.clip(r2, 0.0, None))
        expr_r = "(x1**2 + x2**2)**0.5"
        expr_r2 = "(x1**2 + x2**2)"
        names = []
        cols = []

        # Baseline polynomial features
        self._add_feature("1.0", np.ones(n), names, cols)
        self._add_feature("x1", x1, names, cols)
        self._add_feature("x2", x2, names, cols)
        self._add_feature("x1**2", x1**2, names, cols)
        self._add_feature("x2**2", x2**2, names, cols)
        self._add_feature("x1*x2", x1 * x2, names, cols)
        self._add_feature(expr_r, r, names, cols)
        self._add_feature(expr_r2, r2, names, cols)

        # Frequencies
        w_rad, w_axis, w_cross, w_r2 = self._freq_lists(x1, x2)

        # Radial trig with amplitude variations
        amp_specs = [
            ("1.0", np.ones(n)),
            (expr_r, r),
            (f"1.0/(1.0 + {expr_r})", 1.0 / (1.0 + r))
        ]
        # Optional exponential envelopes
        if self.include_exp:
            for a in [0.2, 0.5]:
                a_str = f"{a:.12g}"
                amp_expr = f"exp(-{a_str}*{expr_r2})"
                amp_vals = np.exp(-a * r2)
                amp_specs.append((amp_expr, amp_vals))

        for w in w_rad:
            w_str = f"{w:.12g}"
            sin_r_vals = np.sin(w * r)
            cos_r_vals = np.cos(w * r)
            sin_r_expr = f"sin({w_str}*{expr_r})"
            cos_r_expr = f"cos({w_str}*{expr_r})"
            for amp_expr, amp_vals in amp_specs:
                if amp_expr == "1.0":
                    self._add_feature(sin_r_expr, sin_r_vals, names, cols)
                    self._add_feature(cos_r_expr, cos_r_vals, names, cols)
                else:
                    self._add_feature(f"({amp_expr})*{sin_r_expr}", amp_vals * sin_r_vals, names, cols)
                    self._add_feature(f"({amp_expr})*{cos_r_expr}", amp_vals * cos_r_vals, names, cols)

        # r^2 trig (small set)
        for w in w_r2:
            w_str = f"{w:.12g}"
            sin_r2_vals = np.sin(w * r2)
            cos_r2_vals = np.cos(w * r2)
            self._add_feature(f"sin({w_str}*{expr_r2})", sin_r2_vals, names, cols)
            self._add_feature(f"cos({w_str}*{expr_r2})", cos_r2_vals, names, cols)

        # Axis trig terms
        for w in w_axis:
            w_str = f"{w:.12g}"
            sx1 = np.sin(w * x1)
            cx1 = np.cos(w * x1)
            sx2 = np.sin(w * x2)
            cx2 = np.cos(w * x2)
            self._add_feature(f"sin({w_str}*x1)", sx1, names, cols)
            self._add_feature(f"cos({w_str}*x1)", cx1, names, cols)
            self._add_feature(f"sin({w_str}*x2)", sx2, names, cols)
            self._add_feature(f"cos({w_str}*x2)", cx2, names, cols)

        # Combination trig across axes (limited)
        for w in w_cross:
            w_str = f"{w:.12g}"
            sx1 = np.sin(w * x1)
            cx1 = np.cos(w * x1)
            sx2 = np.sin(w * x2)
            cx2 = np.cos(w * x2)
            self._add_feature(f"sin({w_str}*x1)*sin({w_str}*x2)", sx1 * sx2, names, cols)
            self._add_feature(f"cos({w_str}*x1)*cos({w_str}*x2)", cx1 * cx2, names, cols)

        # A couple of mixed sums/differences
        for w in [1.0, 2.0, np.pi]:
            w_str = f"{w:.12g}"
            s_plus = np.sin(w * (x1 + x2))
            c_plus = np.cos(w * (x1 + x2))
            s_minus = np.sin(w * (x1 - x2))
            c_minus = np.cos(w * (x1 - x2))
            self._add_feature(f"sin({w_str}*(x1 + x2))", s_plus, names, cols)
            self._add_feature(f"cos({w_str}*(x1 + x2))", c_plus, names, cols)
            self._add_feature(f"sin({w_str}*(x1 - x2))", s_minus, names, cols)
            self._add_feature(f"cos({w_str}*(x1 - x2))", c_minus, names, cols)

        A = np.column_stack(cols) if cols else np.zeros((n, 0))
        return names, A

    def _ridge(self, A, y, lam):
        # Solve (A^T A + lam I) x = A^T y
        if A.size == 0:
            return np.zeros(0)
        AtA = A.T @ A
        m = AtA.shape[0]
        AtA.flat[::m + 1] += lam
        Aty = A.T @ y
        try:
            coef = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coef

    def _omp_with_validation(self, A, y, train_idx, val_idx, kmax, lam, patience):
        # Data splits
        A_tr = A[train_idx]
        y_tr = y[train_idx]
        A_va = A[val_idx]
        y_va = y[val_idx]
        n, p = A_tr.shape
        if p == 0:
            return [], np.array([])

        # Normalize columns for selection
        col_rms = np.sqrt(np.mean(A_tr * A_tr, axis=0) + 1e-18)
        An_tr = A_tr / col_rms

        selected = []
        best_selected = []
        best_coef = np.array([])
        best_val_mse = np.inf
        no_improve = 0
        current_coef = np.array([])

        # Precompute to speed up correlation
        ytr = y_tr.copy()
        pred_tr = np.zeros_like(ytr)

        for t in range(kmax):
            # Compute residual
            resid = ytr - pred_tr

            # Correlations
            corrs = An_tr.T @ resid
            # Zero out already selected
            if selected:
                corrs[selected] = 0.0

            # Select best feature
            j = int(np.argmax(np.abs(corrs)))
            if j in selected:
                # safeguard
                # If all remaining correlations are zero, stop
                if np.allclose(corrs, 0):
                    break
            selected.append(j)

            # Fit ridge on selected
            A_tr_sel = A_tr[:, selected]
            coef_sel = self._ridge(A_tr_sel, y_tr, lam)
            pred_tr = A_tr_sel @ coef_sel

            # Validation performance
            A_va_sel = A_va[:, selected]
            pred_va = A_va_sel @ coef_sel
            val_mse = float(np.mean((y_va - pred_va) ** 2))

            # Keep best
            if val_mse + 1e-12 < best_val_mse:
                best_val_mse = val_mse
                best_selected = selected.copy()
                best_coef = coef_sel.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_selected, best_coef

    def _format_expression(self, names, coefs):
        # Remove negligible coefficients
        if coefs.size == 0:
            return "0.0"
        # Sort terms by absolute coefficient descending for readability
        idx = np.argsort(-np.abs(coefs))
        terms = []
        for j in idx:
            c = float(coefs[j])
            if not np.isfinite(c):
                continue
            # Threshold on absolute coefficient
            if abs(c) < 1e-12:
                continue
            coeff_str = f"{c:.12g}"
            term_expr = names[j]
            if term_expr == "1.0":
                terms.append(f"({coeff_str})")
            else:
                terms.append(f"({coeff_str})*({term_expr})")
        if not terms:
            return "0.0"
        # Combine with +
        expr = " + ".join(terms)
        # Simplify '+ -' by turning '+ (-a)*x' into ' - a*x' is optional; keep as is to avoid parsing issues
        return expr

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        try:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).ravel()
            n = X.shape[0]

            # Generate features
            names, A = self._generate_features(X)
            p = A.shape[1]

            if p == 0:
                # Fallback trivial
                expression = "0.0"
                preds = np.zeros(n)
                return {"expression": expression, "predictions": preds.tolist(), "details": {}}

            # Train/validation split
            rng = np.random.default_rng(self.random_state)
            idx = np.arange(n)
            rng.shuffle(idx)
            val_frac = 0.2 if n >= 50 else 0.15
            n_val = max(1, int(n * val_frac))
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]

            if train_idx.size == 0:
                train_idx = idx
                val_idx = idx

            kmax = int(min(self.max_terms, p, max(6, np.ceil(np.sqrt(p)).astype(int) + 10)))
            selected, _ = self._omp_with_validation(
                A, y, train_idx, val_idx, kmax=kmax, lam=self.ridge_lambda, patience=self.patience
            )
            if not selected:
                # Fallback to ridge on all features with small L2
                coefs_all = self._ridge(A, y, self.ridge_lambda)
                # Select top-k by magnitude
                k = min(10, p)
                top_idx = np.argsort(-np.abs(coefs_all))[:k]
                selected = list(top_idx)

            # Final refit on full data with selected features
            A_sel = A[:, selected]
            coef_sel = self._ridge(A_sel, y, self.ridge_lambda)

            # Prune tiny coefficients relative to target scale
            y_scale = np.maximum(1e-12, np.std(y))
            keep = []
            for i, c in enumerate(coef_sel):
                # Based on normalized contribution on training set
                col = A_sel[:, i]
                contrib_scale = np.sqrt(np.mean((col * c) ** 2))
                if contrib_scale > 1e-6 * y_scale and abs(c) > 1e-12:
                    keep.append(i)
            if not keep:
                keep = list(range(len(selected)))
            selected = [selected[i] for i in keep]
            coef_sel = coef_sel[keep]
            A_sel = A[:, selected]

            # Predictions
            preds = A_sel @ coef_sel

            # Expression string
            names_sel = [names[j] for j in selected]
            expression = self._format_expression(names_sel, coef_sel)

            return {
                "expression": expression,
                "predictions": preds.tolist(),
                "details": {}
            }
        except Exception:
            # Robust fallback: simple trigonometric-polynomial regression
            try:
                X = np.asarray(X, dtype=float)
                y = np.asarray(y, dtype=float).ravel()
                x1 = X[:, 0]
                x2 = X[:, 1]
                r2 = x1 * x1 + x2 * x2
                r = np.sqrt(np.clip(r2, 0.0, None))
                PI = np.pi
                feats = [
                    np.ones_like(x1),
                    x1, x2, x1 * x1, x2 * x2, x1 * x2,
                    r, r2,
                    np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2),
                    np.sin(PI * r), np.cos(PI * r),
                    np.sin(2 * PI * r) / (1.0 + r), np.cos(2 * PI * r) / (1.0 + r)
                ]
                names = [
                    "1.0",
                    "x1", "x2", "x1**2", "x2**2", "x1*x2",
                    "(x1**2 + x2**2)**0.5", "(x1**2 + x2**2)",
                    "sin(x1)", "cos(x1)", "sin(x2)", "cos(x2)",
                    f"sin({PI:.12g}*(x1**2 + x2**2)**0.5)", f"cos({PI:.12g}*(x1**2 + x2**2)**0.5)",
                    f"sin({(2*PI):.12g}*(x1**2 + x2**2)**0.5)/(1.0 + (x1**2 + x2**2)**0.5)",
                    f"cos({(2*PI):.12g}*(x1**2 + x2**2)**0.5)/(1.0 + (x1**2 + x2**2)**0.5)",
                ]
                A = np.column_stack(feats)
                lam = 1e-8
                AtA = A.T @ A
                m = AtA.shape[0]
                AtA.flat[::m + 1] += lam
                Aty = A.T @ y
                try:
                    coef = np.linalg.solve(AtA, Aty)
                except np.linalg.LinAlgError:
                    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                preds = A @ coef
                # Compose expression
                terms = []
                for c, nm in zip(coef, names):
                    if abs(c) < 1e-12:
                        continue
                    cstr = f"{float(c):.12g}"
                    if nm == "1.0":
                        terms.append(f"({cstr})")
                    else:
                        terms.append(f"({cstr})*({nm})")
                expr = " + ".join(terms) if terms else "0.0"
                return {"expression": expr, "predictions": preds.tolist(), "details": {}}
            except Exception:
                return {"expression": "0.0", "predictions": None, "details": {}}