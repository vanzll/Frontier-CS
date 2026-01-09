import numpy as np

class Solution:
    def __init__(self, **kwargs):
        self.round_digits = kwargs.get("round_digits", 12)
        self.use_general_if_better = kwargs.get("use_general_if_better", True)

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")

        x1 = X[:, 0]
        x2 = X[:, 1]

        # Baseline linear model (for comparison)
        A_lin = np.column_stack([x1, x2, np.ones(n)])
        coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
        preds_lin = A_lin @ coeffs_lin
        mse_lin = np.mean((y - preds_lin) ** 2)

        # Specialized McCormick-like model: a*sin(x1 + x2) + b*(x1 - x2)**2 + c*x1 + d*x2 + e
        A_spec = np.column_stack([
            np.sin(x1 + x2),
            (x1 - x2) ** 2,
            x1,
            x2,
            np.ones(n),
        ])
        coeffs_spec, _, _, _ = np.linalg.lstsq(A_spec, y, rcond=None)
        coeffs_spec = self._round_coeffs(coeffs_spec, self.round_digits)
        preds_spec = A_spec @ coeffs_spec
        mse_spec = np.mean((y - preds_spec) ** 2)

        # General model with richer dictionary (only used if clearly better)
        # Terms: [sin(x1 + x2), sin(x1), sin(x2), x1**2, x2**2, x1*x2, x1, x2, 1]
        A_gen = np.column_stack([
            np.sin(x1 + x2),
            np.sin(x1),
            np.sin(x2),
            x1 ** 2,
            x2 ** 2,
            x1 * x2,
            x1,
            x2,
            np.ones(n),
        ])
        coeffs_gen, _, _, _ = np.linalg.lstsq(A_gen, y, rcond=None)
        # Drop near-zero coefficients to reduce complexity
        coeffs_gen = self._prune_and_round_coeffs(coeffs_gen, A_gen, y, round_digits=self.round_digits)
        preds_gen = A_gen @ coeffs_gen
        mse_gen = np.mean((y - preds_gen) ** 2)

        # Choose best model: prefer specialized unless general is significantly better
        # If general MSE is at least 2% lower than specialized, choose general; else specialized
        choose_general = self.use_general_if_better and (mse_gen < 0.98 * mse_spec)

        if choose_general:
            expression = self._build_expression(
                coeffs_gen,
                [
                    "sin(x1 + x2)",
                    "sin(x1)",
                    "sin(x2)",
                    "x1**2",
                    "x2**2",
                    "x1*x2",
                    "x1",
                    "x2",
                    "1",
                ]
            )
            predictions = preds_gen
            details = {"model": "general", "mse": float(mse_gen), "mse_linear_baseline": float(mse_lin)}
        else:
            expression = self._build_expression(
                coeffs_spec,
                [
                    "sin(x1 + x2)",
                    "(x1 - x2)**2",
                    "x1",
                    "x2",
                    "1",
                ]
            )
            predictions = preds_spec
            details = {"model": "specialized_mccormick_form", "mse": float(mse_spec), "mse_linear_baseline": float(mse_lin)}

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details
        }

    def _round_coeffs(self, coeffs, digits=12):
        rounded = np.zeros_like(coeffs, dtype=float)
        for i, c in enumerate(coeffs):
            rounded[i] = self._round_sig(c, digits)
        return rounded

    def _round_sig(self, x, digits=12):
        if x == 0 or not np.isfinite(x):
            return 0.0
        exp = int(np.floor(np.log10(abs(x))))
        dec = max(digits - 1 - exp, 0)
        return float(np.round(x, decimals=dec))

    def _prune_and_round_coeffs(self, coeffs, A, y, eps=1e-12, round_digits=12):
        # Round first
        coeffs = self._round_coeffs(coeffs, round_digits)
        # Prune tiny coefficients by magnitude threshold and by contribution threshold
        # Contribution threshold: if removing a term increases MSE by negligible amount, drop it
        preds_full = A @ coeffs
        mse_full = np.mean((y - preds_full) ** 2)

        keep = np.ones_like(coeffs, dtype=bool)
        # First pass: magnitude threshold
        mag_thresh = max(1e-10, eps)
        keep &= np.abs(coeffs) > mag_thresh

        # Second pass: contribution-based pruning for very small effects
        for j in range(len(coeffs)):
            if not keep[j]:
                continue
            if np.abs(coeffs[j]) < 1e-6:
                # Evaluate MSE increase by dropping term j
                c_backup = coeffs[j]
                coeffs[j] = 0.0
                preds_drop = A @ coeffs
                mse_drop = np.mean((y - preds_drop) ** 2)
                # If MSE increase is smaller than 0.1% of current MSE or absolute 1e-9, drop
                if (mse_drop - mse_full) <= max(1e-9, 1e-3 * mse_full):
                    keep[j] = False
                    mse_full = mse_drop
                    preds_full = preds_drop
                else:
                    coeffs[j] = c_backup

        coeffs[~keep] = 0.0
        # Re-round after pruning to clean up near-zero artifacts
        coeffs = self._round_coeffs(coeffs, round_digits)
        return coeffs

    def _build_expression(self, coeffs, terms):
        term_strs = []
        for c, t in zip(coeffs, terms):
            if abs(c) < 1e-15:
                continue
            c_str = self._float_to_str(c)
            if t == "1":
                term_strs.append(f"{c_str}")
            else:
                # Prefer cleaner multiplications without redundant parentheses for simple terms
                # For safety, wrap complex terms
                if any(ch in t for ch in ["+", "-", "*", "/", "**", " "]) and not (t.startswith("(") and t.endswith(")")):
                    t_fmt = f"({t})"
                else:
                    t_fmt = t
                term_strs.append(f"{c_str}*{t_fmt}")
        if not term_strs:
            return "0"
        return " + ".join(term_strs)

    def _float_to_str(self, x):
        # Format with significant digits and remove trailing zeros/decimal if not needed
        s = f"{self._round_sig(float(x), self.round_digits):.{self.round_digits}g}"
        return s