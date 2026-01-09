import numpy as np
import math

class Solution:
    def __init__(self, **kwargs):
        self.random_state = kwargs.get("random_state", 42)

    def _format_float(self, x):
        if np.isnan(x) or np.isinf(x):
            return "0.0"
        s = f"{float(x):.12g}"
        # Ensure a decimal point for integers to avoid integer power precedence confusion
        if "e" not in s and "." not in s:
            s += ".0"
        return s

    def _safe_eval(self, expression, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        local_dict = {
            "x1": x1,
            "x2": x2,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
        }
        try:
            yhat = eval(expression, {"__builtins__": {}}, local_dict)
            if isinstance(yhat, (list, tuple)):
                yhat = np.array(yhat)
            yhat = np.asarray(yhat, dtype=float)
            if yhat.shape != (X.shape[0],):
                yhat = yhat.reshape(-1)
            return yhat
        except Exception:
            # As a last resort, return zeros to avoid crashing
            return np.zeros(X.shape[0], dtype=float)

    def _build_features(self, X):
        # Build a compact yet expressive basis for ripple-like patterns
        x1 = X[:, 0]
        x2 = X[:, 1]
        r2 = x1 * x1 + x2 * x2
        eps_r = 1e-6
        r = np.sqrt(r2 + eps_r)  # keep numerically stable
        # Prepare feature containers
        feats = []
        feat_strs = []
        # Helper lambda to add feature and its string
        def add_feature(arr, s):
            feats.append(arr)
            feat_strs.append(s)

        r2_str = "x1**2 + x2**2"
        r_str = f"({r2_str} + {self._format_float(eps_r)})**0.5"

        # Base polynomial terms
        add_feature(np.ones_like(r), "1.0")
        add_feature(r2, f"{r2_str}")
        add_feature(r2 * r2, f"({r2_str})**2")

        # Amplitude factors
        amp_forms = [
            ("", 1.0, "1.0"),  # no amplitude scaling
            ("/(1.0 + r2)", None, f"1.0/(1.0 + {r2_str})"),
            ("*exp(-0.5*r2)", None, f"exp(-0.5*({r2_str}))"),
        ]

        # Frequencies for radial oscillations
        freqs = [3.0, 5.0, 7.0, 10.0, 12.0]

        # Radial sin/cos with amplitude factors
        for k in freqs:
            sin_k_r = np.sin(k * r)
            cos_k_r = np.cos(k * r)
            sin_str_base = f"sin({self._format_float(k)}*{r_str})"
            cos_str_base = f"cos({self._format_float(k)}*{r_str})"
            for amp_tag, _, amp_str in amp_forms:
                if amp_tag == "":
                    add_feature(sin_k_r, f"{sin_str_base}")
                    add_feature(cos_k_r, f"{cos_str_base}")
                elif amp_tag == "/(1.0 + r2)":
                    arr_amp = 1.0 / (1.0 + r2)
                    add_feature(sin_k_r * arr_amp, f"{sin_str_base}*{amp_str}")
                    add_feature(cos_k_r * arr_amp, f"{cos_str_base}*{amp_str}")
                elif amp_tag == "*exp(-0.5*r2)":
                    arr_amp = np.exp(-0.5 * r2)
                    add_feature(sin_k_r * arr_amp, f"{sin_str_base}*{amp_str}")
                    add_feature(cos_k_r * arr_amp, f"{cos_str_base}*{amp_str}")

        # Sinc-like terms: sin(k*r)/(r)
        for k in [5.0, 10.0]:
            arr = np.sin(k * r) / (r + eps_r)
            add_feature(arr, f"sin({self._format_float(k)}*{r_str})/({r_str})")

        # A couple of r2-based oscillations
        for k in [3.0, 5.0]:
            add_feature(np.sin(k * r2), f"sin({self._format_float(k)}*({r2_str}))")
            add_feature(np.cos(k * r2), f"cos({self._format_float(k)}*({r2_str}))")

        # Stack features and keep their strings
        Phi = np.column_stack(feats)
        return Phi, feat_strs

    def _ridge_cv(self, Phi, y, lams=None, val_ratio=0.2, seed=42):
        n = Phi.shape[0]
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(val_ratio * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        A_tr = Phi[tr_idx]
        A_val = Phi[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # Normalize columns (L2) for numerical stability; restore later
        col_scale = np.sqrt(np.mean(A_tr ** 2, axis=0))
        col_scale[col_scale == 0] = 1.0
        A_tr_n = A_tr / col_scale
        A_val_n = A_val / col_scale

        if lams is None:
            lams = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2]

        AtA = A_tr_n.T @ A_tr_n
        Aty = A_tr_n.T @ y_tr

        best_mse = np.inf
        best_lam = lams[0]
        best_beta_n = None

        I = np.eye(AtA.shape[0])
        for lam in lams:
            try:
                beta_n = np.linalg.solve(AtA + lam * I, Aty)
            except np.linalg.LinAlgError:
                # Use regularized pseudo-inverse if singular
                beta_n = np.linalg.pinv(AtA + lam * I) @ Aty
            y_pred_val = A_val_n @ beta_n
            mse = float(np.mean((y_pred_val - y_val) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_lam = lam
                best_beta_n = beta_n

        # Refit on full data with chosen lambda
        A_full = Phi
        col_scale_full = np.sqrt(np.mean(A_full ** 2, axis=0))
        col_scale_full[col_scale_full == 0] = 1.0
        A_full_n = A_full / col_scale_full

        AtA_full = A_full_n.T @ A_full_n
        Aty_full = A_full_n.T @ y
        lam = best_lam
        try:
            beta_n_full = np.linalg.solve(AtA_full + lam * np.eye(AtA_full.shape[0]), Aty_full)
        except np.linalg.LinAlgError:
            beta_n_full = np.linalg.pinv(AtA_full + lam * np.eye(AtA_full.shape[0])) @ Aty_full

        # Unscale coefficients
        beta = beta_n_full / col_scale_full
        return beta

    def _prune_and_build_expression(self, beta, feat_strs, kmax=12, thresh=None):
        beta = np.asarray(beta).reshape(-1)
        absb = np.abs(beta)
        if thresh is None:
            # Threshold relative to median absolute coefficient for robustness
            nonzero = absb[absb > 0]
            if nonzero.size == 0:
                thresh = 0.0
            else:
                thresh = np.median(nonzero) * 0.01

        idx_sorted = np.argsort(-absb)
        selected = []
        for idx in idx_sorted:
            if absb[idx] >= thresh:
                selected.append(idx)
            if len(selected) >= kmax:
                break

        if len(selected) == 0:
            # Fallback to zero function
            return "0.0", np.array([], dtype=int), np.array([])

        # Sort selected by contribution magnitude for deterministic ordering
        selected = np.array(selected, dtype=int)
        # Build expression as a sum of c_i * phi_i
        terms = []
        for j in selected:
            c = beta[j]
            s = feat_strs[j]
            c_str = self._format_float(c)
            # Simplify if feature is "1.0"
            if s == "1.0":
                term = f"{c_str}"
            else:
                term = f"{c_str}*({s})"
            terms.append(term)

        # Combine with '+' and handle signs explicitly to avoid "+ -"
        expr = None
        for t in terms:
            if expr is None:
                expr = t
            else:
                # t may start with '-' if c is negative; handle elegantly
                if t.lstrip().startswith("-"):
                    expr += " - " + t.lstrip()[1:]
                else:
                    expr += " + " + t
        if expr is None:
            expr = "0.0"
        return expr, selected, beta[selected]

    def _fit_pysr(self, X, y):
        try:
            from pysr import PySRRegressor
            import sympy as sp
        except Exception:
            return None

        try:
            model = PySRRegressor(
                niterations=60,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "log"],
                populations=8,
                population_size=33,
                maxsize=35,
                verbosity=0,
                progress=False,
                random_state=self.random_state,
                procs=0,  # single-process to reduce overhead in constrained envs
            )
            model.fit(X, y, variable_names=["x1", "x2"])
            sym_expr = model.sympy()

            # Convert sympy string to allowed expression format: replace sqrt(...)
            s = str(sym_expr)

            def replace_sqrt(text):
                res = []
                i = 0
                n = len(text)
                while i < n:
                    if text.startswith("sqrt(", i):
                        # find matching ')'
                        j = i + 5
                        depth = 1
                        while j < n and depth > 0:
                            if text[j] == "(":
                                depth += 1
                            elif text[j] == ")":
                                depth -= 1
                            j += 1
                        inner = text[i + 5 : j - 1]
                        res.append("(")
                        res.append(inner)
                        res.append(")**0.5")
                        i = j
                    else:
                        res.append(text[i])
                        i += 1
                return "".join(res)

            expression = replace_sqrt(s)

            # Ensure only allowed names are used (replace sympy constants if needed)
            expression = expression.replace("pi", self._format_float(math.pi))
            expression = expression.replace("E", self._format_float(math.e))

            # Compute predictions using the expression to ensure consistency
            preds = self._safe_eval(expression, X)
            if np.any(~np.isfinite(preds)):
                preds = model.predict(X)
            return {"expression": expression, "predictions": preds}
        except Exception:
            return None

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Try PySR first
        pysr_result = self._fit_pysr(X, y)

        # Build manual basis and fit ridge regression
        Phi, feat_strs = self._build_features(X)
        beta = self._ridge_cv(Phi, y, lams=None, val_ratio=0.2, seed=self.random_state)
        expr_fallback, idx_sel, beta_sel = self._prune_and_build_expression(beta, feat_strs, kmax=12)

        preds_fallback = self._safe_eval(expr_fallback, X)
        mse_fallback = float(np.mean((preds_fallback - y) ** 2))

        if pysr_result is not None:
            preds_pysr = np.asarray(pysr_result["predictions"], dtype=float).reshape(-1)
            mse_pysr = float(np.mean((preds_pysr - y) ** 2))
            # Choose the better one
            if mse_pysr <= mse_fallback * 0.98:  # prefer PySR if clearly better
                return {
                    "expression": pysr_result["expression"],
                    "predictions": preds_pysr.tolist(),
                    "details": {}
                }

        # Fallback solution
        return {
            "expression": expr_fallback,
            "predictions": preds_fallback.tolist(),
            "details": {}
        }