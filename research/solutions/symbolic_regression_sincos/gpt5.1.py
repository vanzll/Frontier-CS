import numpy as np

class Solution:
    def __init__(self, **kwargs):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be of shape (n, 2).")

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Basis functions
        s1 = np.sin(x1)
        c1 = np.cos(x1)
        s2 = np.sin(x2)
        c2 = np.cos(x2)

        features = [
            s1,                # 0
            c1,                # 1
            s2,                # 2
            c2,                # 3
            s1 * s2,           # 4
            s1 * c2,           # 5
            c1 * s2,           # 6
            c1 * c2,           # 7
        ]

        names = [
            "sin(x1)",
            "cos(x1)",
            "sin(x2)",
            "cos(x2)",
            "sin(x1)*sin(x2)",
            "sin(x1)*cos(x2)",
            "cos(x1)*sin(x2)",
            "cos(x1)*cos(x2)",
        ]

        # Fit linear model y ≈ sum_i a_i * feature_i + intercept
        A = np.column_stack(features + [np.ones_like(x1)])
        coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        feat_coefs = coefs[:-1]
        intercept = coefs[-1]

        # Threshold small coefficients for sparsity
        max_abs = np.max(np.abs(feat_coefs)) if feat_coefs.size > 0 else 0.0
        if not np.isfinite(max_abs):
            max_abs = 0.0

        thr_rel = 0.05
        thr_min = 1e-4
        if max_abs > 0:
            thr = max(thr_rel * max_abs, thr_min)
        else:
            thr = thr_min

        kept_indices = []
        kept_coefs = []
        coef_mult_flags = []  # whether we multiply by coefficient in expression

        unary_per_feature = [1, 1, 1, 1, 2, 2, 2, 2]
        binary_internal_per_feature = [0, 0, 0, 0, 1, 1, 1, 1]

        for i, c in enumerate(feat_coefs):
            if not np.isfinite(c):
                continue
            if abs(c) >= thr:
                c_round = float(np.round(c, 8))
                if abs(c_round) < thr:
                    continue
                kept_indices.append(i)
                kept_coefs.append(c_round)
                # placeholder; actual coef_mult_flag decided when building expression
                coef_mult_flags.append(True)

        # Intercept handling
        intercept_use = False
        intercept_round = 0.0
        if np.isfinite(intercept):
            intercept_round = float(np.round(intercept, 8))
            if abs(intercept_round) >= thr_min:
                intercept_use = True
            else:
                intercept_round = 0.0

        # Build expression string
        terms_str = []
        if intercept_use:
            intercept_str = f"{intercept_round:.8g}"
            terms_str.append(intercept_str)

        # Adjust coef_mult_flags based on whether coefficient is ±1
        for idx_in_list, (idx, c) in enumerate(zip(kept_indices, kept_coefs)):
            c_str = f"{c:.8g}"
            if abs(c - 1.0) < 1e-8:
                term_str = f"{names[idx]}"
                coef_mult_flags[idx_in_list] = False
            elif abs(c + 1.0) < 1e-8:
                term_str = f"-({names[idx]})"
                coef_mult_flags[idx_in_list] = False
            else:
                term_str = f"({c_str})*({names[idx]})"
                coef_mult_flags[idx_in_list] = True
            terms_str.append(term_str)

        if terms_str:
            expression = " + ".join(terms_str)
        else:
            expression = "0"

        # Compute predictions from selected features
        n = x1.shape[0]
        y_pred = np.zeros(n, dtype=float)
        if intercept_use:
            y_pred += intercept_round
        for idx, c in zip(kept_indices, kept_coefs):
            y_pred += c * features[idx]

        # Estimate complexity: C = 2 * (#binary ops) + (#unary ops)
        k = len(kept_indices)
        unary_ops = 0
        binary_ops = 0
        for flag, idx in zip(coef_mult_flags, kept_indices):
            unary_ops += unary_per_feature[idx]
            binary_ops += binary_internal_per_feature[idx]
            if flag:
                binary_ops += 1  # coefficient multiplication

        if intercept_use:
            n_terms = k + 1
        else:
            n_terms = k

        if n_terms > 1:
            binary_ops += (n_terms - 1)  # additions between terms

        complexity = 2 * binary_ops + unary_ops

        details = {"complexity": int(complexity)}

        return {
            "expression": expression,
            "predictions": y_pred.tolist(),
            "details": details,
        }