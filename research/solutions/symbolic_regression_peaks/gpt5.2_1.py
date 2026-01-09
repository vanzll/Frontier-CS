import numpy as np

def _safe_lstsq(A: np.ndarray, b: np.ndarray):
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(coef)):
            return None
        return coef
    except Exception:
        return None

def _fmt(c: float) -> str:
    if not np.isfinite(c):
        return "0.0"
    s = format(float(c), ".17g")
    if s == "-0":
        s = "0"
    return f"({s})"

class Solution:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n = int(X.shape[0])
        if n == 0:
            return {"expression": "0.0", "predictions": [], "details": {}}

        x1 = X[:, 0].astype(float, copy=False)
        x2 = X[:, 1].astype(float, copy=False)

        # Linear baseline
        A_lin = np.column_stack([x1, x2, np.ones_like(x1)])
        coef_lin = _safe_lstsq(A_lin, y)
        if coef_lin is None:
            coef_lin = np.array([0.0, 0.0, float(np.mean(y))], dtype=float)
        a_lin, b_lin, c_lin = (float(coef_lin[0]), float(coef_lin[1]), float(coef_lin[2]))
        pred_lin = a_lin * x1 + b_lin * x2 + c_lin
        mse_lin = float(np.mean((y - pred_lin) ** 2))

        # Peaks-structure linear-in-parameters model
        # y ≈ c1*(1-x1)^2*exp(-x1^2-(x2+1)^2) + c2*(x1/5 - x1^3 - x2^5)*exp(-x1^2-x2^2) + c3*exp(-(x1+1)^2-x2^2) + c0
        t1 = (1.0 - x1) ** 2 * np.exp(-(x1 ** 2) - (x2 + 1.0) ** 2)
        t2 = (x1 / 5.0 - x1 ** 3 - x2 ** 5) * np.exp(-(x1 ** 2) - (x2 ** 2))
        t3 = np.exp(-((x1 + 1.0) ** 2) - (x2 ** 2))
        A_pk = np.column_stack([t1, t2, t3, np.ones_like(x1)])
        coef_pk = _safe_lstsq(A_pk, y)
        if coef_pk is None:
            coef_pk = np.array([3.0, -10.0, -1.0 / 3.0, 0.0], dtype=float)
        c1, c2, c3, c0 = map(float, coef_pk.tolist())
        pred_pk = c1 * t1 + c2 * t2 + c3 * t3 + c0
        mse_pk = float(np.mean((y - pred_pk) ** 2))

        use_peaks = np.isfinite(mse_pk) and (mse_pk <= mse_lin * 0.999999)

        if use_peaks:
            expression = (
                f"{_fmt(c1)}*(1-x1)**2*exp(-(x1**2)-(x2+1)**2)"
                f" + {_fmt(c2)}*(x1/5 - x1**3 - x2**5)*exp(-(x1**2)-(x2**2))"
                f" + {_fmt(c3)}*exp(-((x1+1)**2)-(x2**2))"
            )
            if abs(c0) > 0.0:
                expression += f" + {_fmt(c0)}"
            predictions = pred_pk
            details = {"mse": mse_pk, "baseline_mse": mse_lin}
        else:
            expression = f"{_fmt(a_lin)}*x1 + {_fmt(b_lin)}*x2 + {_fmt(c_lin)}"
            predictions = pred_lin
            details = {"mse": mse_lin, "baseline_mse": mse_lin}

        return {
            "expression": expression,
            "predictions": predictions.tolist(),
            "details": details,
        }