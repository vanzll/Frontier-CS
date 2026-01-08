import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _as_2d_float(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() > 2:
        x = x.view(x.shape[0], -1)
    if x.shape[-1] != input_dim:
        x = x.view(x.shape[0], -1)
    return x.to(dtype=torch.float32).contiguous()


def _as_1d_long(y: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if y.dim() > 1:
        y = y.view(-1)
    return y.to(dtype=torch.long).contiguous()


def _collect_from_loader(loader, input_dim: int, device: str = "cpu"):
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Unsupported batch format from loader.")
        x = _as_2d_float(x, input_dim)
        y = _as_1d_long(y)
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs, dim=0).to(device=device, non_blocking=False).contiguous()
    Y = torch.cat(ys, dim=0).to(device=device, non_blocking=False).contiguous()
    return X, Y


def _standardize_stats(X: torch.Tensor):
    mean = X.mean(dim=0)
    var = (X - mean).pow(2).mean(dim=0)
    std = torch.sqrt(var + 1e-6)
    inv_std = 1.0 / std
    return mean.contiguous(), inv_std.contiguous()


class _BaseStdModel(nn.Module):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone().to(dtype=torch.float32).contiguous())
        self.register_buffer("inv_std", inv_std.detach().clone().to(dtype=torch.float32).contiguous())

    def _std(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return (x - self.mean) * self.inv_std


class RidgeClassifier(_BaseStdModel):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__(mean, inv_std)
        self.register_buffer("weight", weight.detach().clone().to(dtype=torch.float32).contiguous())  # (C, D)
        self.register_buffer("bias", bias.detach().clone().to(dtype=torch.float32).contiguous())    # (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._std(x)
        return F.linear(x, self.weight, self.bias)


class LDAClassifier(_BaseStdModel):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__(mean, inv_std)
        self.register_buffer("weight", weight.detach().clone().to(dtype=torch.float32).contiguous())  # (C, D)
        self.register_buffer("bias", bias.detach().clone().to(dtype=torch.float32).contiguous())    # (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._std(x)
        return F.linear(x, self.weight, self.bias)


class MLPResClassifier(_BaseStdModel):
    def __init__(self, input_dim: int, num_classes: int, hidden1: int, mean: torch.Tensor, inv_std: torch.Tensor, dropout: float = 0.10):
        super().__init__(mean, inv_std)
        self.ln_in = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.fc3 = nn.Linear(input_dim, num_classes)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._std(x)
        x0 = self.ln_in(x)
        h = self.fc1(x0)
        h = self.ln1(h)
        h = F.gelu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fc2(h)
        h = self.ln2(h)
        h = F.gelu(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x0
        return self.fc3(h)


class EnsembleModel(nn.Module):
    def __init__(self, models, weights):
        super().__init__()
        self.models = nn.ModuleList(models)
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for w, m in zip(self.weights, self.models):
            if out is None:
                out = m(x) * w
            else:
                out = out + m(x) * w
        return out


def _eval_acc(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> float:
    model.eval()
    correct = 0
    total = y.numel()
    with torch.inference_mode():
        for i in range(0, total, batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return float(correct) / float(total) if total > 0 else 0.0


def _fit_ridge(Xtr: torch.Tensor, ytr: torch.Tensor, Xv: torch.Tensor, yv: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor, num_classes: int):
    Xs = ((Xtr - mean) * inv_std).to(dtype=torch.float64)
    N, D = Xs.shape
    ones = torch.ones((N, 1), dtype=torch.float64, device=Xs.device)
    Xb = torch.cat([Xs, ones], dim=1)  # (N, D+1)

    Y = torch.zeros((N, num_classes), dtype=torch.float64, device=Xs.device)
    Y[torch.arange(N, device=Xs.device), ytr.to(dtype=torch.long)] = 1.0

    XtX = Xb.T @ Xb  # (D+1, D+1)
    XtY = Xb.T @ Y   # (D+1, C)

    best = None
    best_acc = -1.0

    lambdas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
    eye = torch.eye(D + 1, dtype=torch.float64, device=Xs.device)

    for lam in lambdas:
        A = XtX + (lam * float(N)) * eye
        jitter = 1e-9
        for _ in range(3):
            try:
                L = torch.linalg.cholesky(A + jitter * eye)
                Wb = torch.cholesky_solve(XtY, L)  # (D+1, C)
                break
            except Exception:
                jitter *= 10.0
        else:
            continue

        W = Wb[:D, :].T.to(dtype=torch.float32).contiguous()  # (C, D)
        b = Wb[D, :].to(dtype=torch.float32).contiguous()     # (C,)
        model = RidgeClassifier(mean, inv_std, W, b)
        acc = _eval_acc(model, Xv, yv, batch_size=1024)
        if acc > best_acc:
            best_acc = acc
            best = model

    return best, best_acc


def _fit_lda(Xtr: torch.Tensor, ytr: torch.Tensor, Xv: torch.Tensor, yv: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor, num_classes: int):
    Xs = ((Xtr - mean) * inv_std).to(dtype=torch.float64)
    y = ytr.to(dtype=torch.long)
    N, D = Xs.shape

    mu = torch.zeros((num_classes, D), dtype=torch.float64, device=Xs.device)
    counts = torch.zeros((num_classes,), dtype=torch.float64, device=Xs.device)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            mu[c] = Xs.index_select(0, idx).mean(dim=0)
            counts[c] = float(idx.numel())

    mu_y = mu.index_select(0, y)
    Xc = Xs - mu_y

    denom = max(1.0, float(N - num_classes))
    S = (Xc.T @ Xc) / denom  # (D, D)

    best = None
    best_acc = -1.0

    trace = torch.trace(S) / float(D)
    eye = torch.eye(D, dtype=torch.float64, device=Xs.device)

    shrinkages = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
    for a in shrinkages:
        Sigma = (1.0 - a) * S + a * trace * eye
        jitter = 1e-7
        for _ in range(4):
            try:
                L = torch.linalg.cholesky(Sigma + jitter * eye)
                M = torch.cholesky_solve(mu.T, L)  # (D, C) = Sigma^{-1} * mu^T
                break
            except Exception:
                jitter *= 10.0
        else:
            continue

        const = -0.5 * (mu.T * M).sum(dim=0)  # (C,)
        W = M.T.to(dtype=torch.float32).contiguous()         # (C, D)
        b = const.to(dtype=torch.float32).contiguous()       # (C,)
        model = LDAClassifier(mean, inv_std, W, b)
        acc = _eval_acc(model, Xv, yv, batch_size=1024)
        if acc > best_acc:
            best_acc = acc
            best = model

    return best, best_acc


def _train_mlp(model: nn.Module, Xtr: torch.Tensor, ytr: torch.Tensor, Xv: torch.Tensor, yv: torch.Tensor, max_epochs: int = 60):
    N = ytr.numel()
    batch_size = 256 if N >= 256 else N
    steps_per_epoch = int(math.ceil(N / batch_size))
    total_steps = max(1, max_epochs * steps_per_epoch)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=25.0
    )

    best_state = None
    best_acc = -1.0
    patience = 15
    bad = 0

    model.train()
    for epoch in range(max_epochs):
        perm = torch.randperm(N, device=Xtr.device)
        model.train()
        for si in range(0, N, batch_size):
            idx = perm[si:si + batch_size]
            xb = Xtr.index_select(0, idx)
            yb = ytr.index_select(0, idx)

            mix = (epoch >= 4) and (torch.rand((), device=Xtr.device).item() < 0.35) and (idx.numel() >= 2)
            if mix:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                j = torch.randperm(xb.shape[0], device=xb.device)
                xb2 = xb.index_select(0, j)
                yb2 = yb.index_select(0, j)
                xb = xb * lam + xb2 * (1.0 - lam)
                logits = model(xb)
                loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb2)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        acc = _eval_acc(model, Xv, yv, batch_size=1024)
        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if best_acc >= 0.98:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def _logits_on_val(model: nn.Module, Xv: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    outs = []
    with torch.inference_mode():
        for i in range(0, Xv.shape[0], batch_size):
            outs.append(model(Xv[i:i + batch_size]).to(dtype=torch.float32))
    return torch.cat(outs, dim=0).contiguous()


def _pick_best_ensemble(candidates, Xv: torch.Tensor, yv: torch.Tensor):
    kept = []
    for name, model, acc in candidates:
        if model is None:
            continue
        kept.append((name, model, float(acc)))
    if not kept:
        return None

    val_logits = []
    scales = []
    for _, m, _ in kept:
        L = _logits_on_val(m, Xv)
        val_logits.append(L)
        s = float(L.std().item())
        scales.append(1.0 / (s + 1e-6))

    y = yv
    best_acc = -1.0
    best_idx = None
    best_w = None

    # Single
    for i in range(len(kept)):
        pred = val_logits[i].argmax(dim=1)
        acc = float((pred == y).float().mean().item())
        if acc > best_acc:
            best_acc = acc
            best_idx = [i]
            best_w = [1.0]

    # Pairs, step 0.1
    step = 0.1
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            for a in np.arange(0.0, 1.0 + 1e-9, step):
                b = 1.0 - float(a)
                w1 = float(a) * scales[i]
                w2 = float(b) * scales[j]
                comb = val_logits[i] * w1 + val_logits[j] * w2
                pred = comb.argmax(dim=1)
                acc = float((pred == y).float().mean().item())
                if acc > best_acc:
                    best_acc = acc
                    best_idx = [i, j]
                    best_w = [w1, w2]

    # Triples, step 0.2 to limit search
    step3 = 0.2
    if len(kept) >= 3:
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                for k in range(j + 1, len(kept)):
                    for a in np.arange(0.0, 1.0 + 1e-9, step3):
                        for b in np.arange(0.0, 1.0 - float(a) + 1e-9, step3):
                            c = 1.0 - float(a) - float(b)
                            if c < -1e-9:
                                continue
                            w1 = float(a) * scales[i]
                            w2 = float(b) * scales[j]
                            w3 = float(c) * scales[k]
                            comb = val_logits[i] * w1 + val_logits[j] * w2 + val_logits[k] * w3
                            pred = comb.argmax(dim=1)
                            acc = float((pred == y).float().mean().item())
                            if acc > best_acc:
                                best_acc = acc
                                best_idx = [i, j, k]
                                best_w = [w1, w2, w3]

    models = [kept[i][1] for i in best_idx]
    weights = best_w
    wsum = float(sum(weights))
    if wsum <= 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / wsum for w in weights]

    if len(models) == 1:
        return models[0], best_acc

    ens = EnsembleModel(models, weights)
    acc_check = _eval_acc(ens, Xv, yv, batch_size=1024)
    return ens, acc_check


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        device = str(metadata.get("device", "cpu"))
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500000))

        try:
            nthreads = min(8, os.cpu_count() or 8)
            torch.set_num_threads(nthreads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        torch.manual_seed(0)
        np.random.seed(0)

        Xtr, ytr = _collect_from_loader(train_loader, input_dim, device=device)
        if val_loader is None:
            Xv, yv = Xtr, ytr
        else:
            Xv, yv = _collect_from_loader(val_loader, input_dim, device=device)

        mean, inv_std = _standardize_stats(Xtr)

        ridge_model, ridge_acc = _fit_ridge(Xtr, ytr, Xv, yv, mean, inv_std, num_classes)
        lda_model, lda_acc = _fit_lda(Xtr, ytr, Xv, yv, mean, inv_std, num_classes)

        hidden1 = 576
        max_hidden1 = max(64, int((param_limit - (input_dim * num_classes + num_classes + 4 * input_dim + 1)) / (2 * input_dim + 3)))
        hidden1 = min(hidden1, max_hidden1)
        hidden1 = max(hidden1, 256)

        mlp_model = MLPResClassifier(input_dim, num_classes, hidden1, mean, inv_std, dropout=0.10).to(device)
        if _count_trainable_params(mlp_model) > param_limit:
            while hidden1 > 128:
                hidden1 = int(hidden1 * 0.95)
                mlp_model = MLPResClassifier(input_dim, num_classes, hidden1, mean, inv_std, dropout=0.10).to(device)
                if _count_trainable_params(mlp_model) <= param_limit:
                    break

        mlp_acc = -1.0
        if _count_trainable_params(mlp_model) <= param_limit:
            mlp_model, mlp_acc = _train_mlp(mlp_model, Xtr, ytr, Xv, yv, max_epochs=60)

        candidates = [
            ("ridge", ridge_model, ridge_acc),
            ("lda", lda_model, lda_acc),
            ("mlp", mlp_model, mlp_acc),
        ]

        best_model, best_acc = _pick_best_ensemble(candidates, Xv, yv)
        if best_model is None:
            best_model = ridge_model if ridge_model is not None else (lda_model if lda_model is not None else mlp_model)

        best_model = best_model.to(device)
        best_model.eval()

        if _count_trainable_params(best_model) > param_limit:
            safe = ridge_model if ridge_model is not None else lda_model
            if safe is not None:
                safe = safe.to(device)
                safe.eval()
                return safe
        return best_model