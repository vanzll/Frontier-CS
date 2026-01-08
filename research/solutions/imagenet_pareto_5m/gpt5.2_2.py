import os
import math
import copy
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _param_count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def _collect_loader(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device=device)
        yb = yb.to(device=device)
        if xb.dtype != torch.float32:
            xb = xb.float()
        if yb.dtype != torch.long:
            yb = yb.long()
        xs.append(xb)
        ys.append(yb)
    x = torch.cat(xs, dim=0).contiguous()
    y = torch.cat(ys, dim=0).contiguous()
    return x, y


def _stratified_holdout_indices(y: torch.Tensor, num_classes: int, holdout_per_class: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=y.device)
    g.manual_seed(seed)
    y = y.view(-1)
    n = y.numel()
    hold_mask = torch.zeros(n, dtype=torch.bool, device=y.device)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        k = min(holdout_per_class, max(0, idx.numel() - 1))
        if k <= 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        hold_idx = perm[:k]
        hold_mask[hold_idx] = True
    hold_idx = hold_mask.nonzero(as_tuple=False).view(-1)
    train_idx = (~hold_mask).nonzero(as_tuple=False).view(-1)
    return train_idx, hold_idx


class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("inv_std", inv_std.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        return (x - self.mean) * self.inv_std


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * bottleneck, bias=True)
        self.act = GEGLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, init_scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1, eps=1e-8)
        w = F.normalize(self.weight, dim=-1, eps=1e-8)
        return (x @ w.t()) * self.scale


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, mean: torch.Tensor, inv_std: torch.Tensor,
                 hidden_dim: int, bottleneck: int, num_blocks: int, dropout: float):
        super().__init__()
        self.std = Standardize(mean, inv_std)
        self.ln0 = nn.LayerNorm(input_dim)
        self.inp = nn.Linear(input_dim, hidden_dim, bias=True)
        self.drop_in = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([BottleneckResidualBlock(hidden_dim, bottleneck, dropout) for _ in range(num_blocks)])
        self.lnF = nn.LayerNorm(hidden_dim)
        self.head = CosineClassifier(hidden_dim, num_classes, init_scale=16.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.std(x)
        x = self.ln0(x)
        x = self.inp(x)
        x = F.gelu(x)
        x = self.drop_in(x)
        for b in self.blocks:
            x = b(x)
        x = self.lnF(x)
        return self.head(x)


class LDAClassifier(nn.Module):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.std = Standardize(mean, inv_std)
        self.register_buffer("W", W.clone().detach())
        self.register_buffer("b", b.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.std(x)
        return x @ self.W + self.b


class RidgeClassifier(nn.Module):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor, W_aug: torch.Tensor):
        super().__init__()
        self.std = Standardize(mean, inv_std)
        self.register_buffer("W_aug", W_aug.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.std(x)
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        x_aug = torch.cat([x, ones], dim=1)
        return x_aug @ self.W_aug


class HybridModel(nn.Module):
    def __init__(self, mlp: nn.Module, aux: nn.Module, alpha: float):
        super().__init__()
        self.mlp = mlp
        self.aux = aux
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = float(self.alpha.item())
        return a * self.mlp(x) + (1.0 - a) * self.aux(x)


def _fit_lda_shared_cov(X_std: torch.Tensor, y: torch.Tensor, num_classes: int, shrink: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    device = X_std.device
    Xd = X_std.to(dtype=torch.float64)
    y = y.to(device=device, dtype=torch.long)

    n, d = Xd.shape
    counts = torch.bincount(y, minlength=num_classes).to(dtype=torch.float64)
    priors = counts / counts.sum().clamp_min(1.0)
    log_priors = torch.log(priors.clamp_min(1e-12))

    means = torch.zeros((num_classes, d), dtype=torch.float64, device=device)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        means[c] = Xd.index_select(0, idx).mean(dim=0)

    scatter = torch.zeros((d, d), dtype=torch.float64, device=device)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() <= 1:
            continue
        Xc = Xd.index_select(0, idx) - means[c].unsqueeze(0)
        scatter.add_(Xc.t().matmul(Xc))

    denom = max(1.0, float(n - num_classes))
    cov = scatter / denom

    ridge_scale = float(cov.diag().mean().clamp_min(1e-12).item())
    ridge = shrink * ridge_scale
    eye = torch.eye(d, dtype=torch.float64, device=device)

    inv_cov = None
    for k in range(8):
        try:
            inv_cov = torch.linalg.inv(cov + (ridge * (10.0 ** k)) * eye)
            break
        except RuntimeError:
            inv_cov = None
    if inv_cov is None:
        inv_cov = torch.linalg.pinv(cov + (ridge * 1e3) * eye)

    W = inv_cov.matmul(means.t())  # (d, c)
    quad = torch.sum((means.matmul(inv_cov)) * means, dim=1)  # (c,)
    b = (-0.5 * quad + log_priors).to(dtype=torch.float32)

    W = W.to(dtype=torch.float32)
    return W, b


def _fit_ridge_closed_form(X_std: torch.Tensor, y: torch.Tensor, num_classes: int, l2: float = 1e-2) -> torch.Tensor:
    device = X_std.device
    Xd = X_std.to(dtype=torch.float64)
    y = y.to(device=device, dtype=torch.long)
    n, d = Xd.shape

    ones = torch.ones((n, 1), dtype=torch.float64, device=device)
    X_aug = torch.cat([Xd, ones], dim=1)  # (n, d+1)
    dp1 = d + 1

    Y = torch.zeros((n, num_classes), dtype=torch.float64, device=device)
    Y.scatter_(1, y.view(-1, 1), 1.0)

    A = X_aug.t().matmul(X_aug)
    ridge = float(l2) * float(A.diag().mean().clamp_min(1e-12).item())
    A = A + ridge * torch.eye(dp1, dtype=torch.float64, device=device)
    B = X_aug.t().matmul(Y)

    W_aug = torch.linalg.solve(A, B).to(dtype=torch.float32)  # (d+1, c)
    return W_aug


def _choose_mlp_dims(input_dim: int, num_classes: int, param_limit: int) -> Tuple[int, int, int, float]:
    best = None
    dropout = 0.05
    for num_blocks in [3, 2, 4]:
        for hidden_dim in range(2304, 768 - 1, -32):
            bottleneck = min(256, max(64, hidden_dim // 6))
            # estimate params exactly by constructing minimal modules
            dummy_mean = torch.zeros(input_dim)
            dummy_inv = torch.ones(input_dim)
            m = MLPNet(input_dim, num_classes, dummy_mean, dummy_inv, hidden_dim, bottleneck, num_blocks, dropout)
            pc = _param_count_trainable(m)
            if pc <= param_limit:
                if best is None or pc > best[0]:
                    best = (pc, hidden_dim, bottleneck, num_blocks, dropout)
            del m
    if best is None:
        return 512, 64, 2, 0.05
    _, hidden_dim, bottleneck, num_blocks, dropout = best
    return hidden_dim, bottleneck, num_blocks, dropout


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 5_000_000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass
        torch.manual_seed(0)

        X_tr_raw, y_tr_raw = _collect_loader(train_loader, device)
        X_va_raw, y_va_raw = _collect_loader(val_loader, device)

        X_all = torch.cat([X_tr_raw, X_va_raw], dim=0).contiguous()
        y_all = torch.cat([y_tr_raw, y_va_raw], dim=0).contiguous()

        if X_all.shape[1] != input_dim:
            input_dim = int(X_all.shape[1])

        mean = X_all.mean(dim=0)
        var = (X_all - mean).pow(2).mean(dim=0)
        std = var.sqrt().clamp_min(1e-6)
        inv_std = (1.0 / std).contiguous()

        # internal stratified holdout for model selection
        train_idx, hold_idx = _stratified_holdout_indices(y_all, num_classes, holdout_per_class=2, seed=1)
        if hold_idx.numel() < num_classes:  # fallback to 1 per class
            train_idx, hold_idx = _stratified_holdout_indices(y_all, num_classes, holdout_per_class=1, seed=1)

        X_train = X_all.index_select(0, train_idx).contiguous()
        y_train = y_all.index_select(0, train_idx).contiguous()
        X_hold = X_all.index_select(0, hold_idx).contiguous()
        y_hold = y_all.index_select(0, hold_idx).contiguous()

        X_train_std = (X_train - mean) * inv_std
        X_hold_std = (X_hold - mean) * inv_std
        X_all_std = (X_all - mean) * inv_std

        # Fit LDA and Ridge on internal training set
        W_lda, b_lda = _fit_lda_shared_cov(X_train_std, y_train, num_classes, shrink=0.05)
        lda_model_train = LDAClassifier(mean, inv_std, W_lda, b_lda).to(device)
        with torch.no_grad():
            acc_lda = _accuracy_from_logits(lda_model_train(X_hold), y_hold)

        W_ridge = _fit_ridge_closed_form(X_train_std, y_train, num_classes, l2=1e-2)
        ridge_model_train = RidgeClassifier(mean, inv_std, W_ridge).to(device)
        with torch.no_grad():
            acc_ridge = _accuracy_from_logits(ridge_model_train(X_hold), y_hold)

        # Train MLP on internal training set, early stopping on holdout
        hidden_dim, bottleneck, num_blocks, dropout = _choose_mlp_dims(input_dim, num_classes, param_limit)
        mlp = MLPNet(input_dim, num_classes, mean, inv_std, hidden_dim, bottleneck, num_blocks, dropout).to(device)

        if _param_count_trainable(mlp) > param_limit:
            # very conservative fallback
            hidden_dim = 1024
            bottleneck = 128
            num_blocks = 2
            dropout = 0.05
            mlp = MLPNet(input_dim, num_classes, mean, inv_std, hidden_dim, bottleneck, num_blocks, dropout).to(device)

        batch_size = 128
        n_train = X_train.shape[0]
        steps_per_epoch = max(1, math.ceil(n_train / batch_size))
        max_epochs = 200
        min_epochs = 25
        patience = 25

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=2e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            total_steps=max_epochs * steps_per_epoch,
            pct_start=0.15,
            div_factor=20.0,
            final_div_factor=50.0,
            anneal_strategy="cos",
        )

        mixup_alpha = 0.2
        mixup_prob = 0.5
        ema_decay = 0.995

        ema_shadow: Dict[str, torch.Tensor] = {}
        for name, p in mlp.named_parameters():
            if p.requires_grad:
                ema_shadow[name] = p.detach().clone()

        best_acc_mlp = -1.0
        best_ema: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0
        epochs_no_improve = 0

        g = torch.Generator(device=device)
        g.manual_seed(2)

        for epoch in range(max_epochs):
            mlp.train()
            perm = torch.randperm(n_train, generator=g, device=device)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_train.index_select(0, idx)
                yb = y_train.index_select(0, idx)

                if mixup_alpha > 0.0 and torch.rand((), generator=g, device=device).item() < mixup_prob and xb.shape[0] >= 2:
                    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(device=device)
                    lam = float(lam.item())
                    rp = torch.randperm(xb.shape[0], generator=g, device=device)
                    xb2 = xb.index_select(0, rp)
                    yb2 = yb.index_select(0, rp)
                    xmix = xb * lam + xb2 * (1.0 - lam)

                    optimizer.zero_grad(set_to_none=True)
                    logits = mlp(xmix)
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb2)
                    loss.backward()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    logits = mlp(xb)
                    loss = criterion(logits, yb)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for name, p in mlp.named_parameters():
                        if p.requires_grad:
                            ema_shadow[name].mul_(ema_decay).add_(p.detach(), alpha=(1.0 - ema_decay))

            mlp.eval()
            with torch.no_grad():
                logits_h = mlp(X_hold)
                acc = _accuracy_from_logits(logits_h, y_hold)

            if acc > best_acc_mlp + 1e-6:
                best_acc_mlp = acc
                best_epoch = epoch + 1
                best_ema = {k: v.clone() for k, v in ema_shadow.items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) >= min_epochs and epochs_no_improve >= patience:
                break

        if best_ema is not None:
            with torch.no_grad():
                name_to_param = dict(mlp.named_parameters())
                for k, v in best_ema.items():
                    if k in name_to_param:
                        name_to_param[k].copy_(v)

        mlp.eval()
        with torch.no_grad():
            acc_mlp = _accuracy_from_logits(mlp(X_hold), y_hold)

        # Hybrid tuning: MLP + LDA (trained on internal training set)
        alpha_grid = [0.0, 0.15, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        best_alpha = 0.7
        best_acc_hybrid = -1.0
        with torch.no_grad():
            mlp_logits = mlp(X_hold)
            lda_logits = lda_model_train(X_hold)
            for a in alpha_grid:
                comb = a * mlp_logits + (1.0 - a) * lda_logits
                aa = _accuracy_from_logits(comb, y_hold)
                if aa > best_acc_hybrid + 1e-9:
                    best_acc_hybrid = aa
                    best_alpha = a

        # Select best approach
        candidates = [
            ("lda", acc_lda),
            ("ridge", acc_ridge),
            ("mlp", acc_mlp),
            ("hybrid", best_acc_hybrid),
        ]
        candidates.sort(key=lambda t: t[1], reverse=True)
        choice = candidates[0][0]

        # Refit auxiliary models on all available data for final return
        W_lda_all, b_lda_all = _fit_lda_shared_cov(X_all_std, y_all, num_classes, shrink=0.05)
        lda_all = LDAClassifier(mean, inv_std, W_lda_all, b_lda_all).to(device)
        W_ridge_all = _fit_ridge_closed_form(X_all_std, y_all, num_classes, l2=1e-2)
        ridge_all = RidgeClassifier(mean, inv_std, W_ridge_all).to(device)

        # Optional fine-tune MLP on all data
        def finetune_mlp(model: nn.Module, X: torch.Tensor, y: torch.Tensor, lr: float = 6e-4, epochs: int = 12):
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=2e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * max(1, math.ceil(X.shape[0] / batch_size))))
            ema2_decay = 0.99
            ema2: Dict[str, torch.Tensor] = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    ema2[n] = p.detach().clone()
            gg = torch.Generator(device=device)
            gg.manual_seed(3)
            nX = X.shape[0]
            for _ in range(epochs):
                perm2 = torch.randperm(nX, generator=gg, device=device)
                for start in range(0, nX, batch_size):
                    idx = perm2[start:start + batch_size]
                    xb = X.index_select(0, idx)
                    yb = y.index_select(0, idx)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sch.step()
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                ema2[n].mul_(ema2_decay).add_(p.detach(), alpha=(1.0 - ema2_decay))
            with torch.no_grad():
                ntp = dict(model.named_parameters())
                for n, v in ema2.items():
                    if n in ntp:
                        ntp[n].copy_(v)
            model.eval()

        if choice in ("mlp", "hybrid"):
            finetune_mlp(mlp, X_all, y_all, lr=6e-4, epochs=12)

        if choice == "lda":
            final_model: nn.Module = lda_all
        elif choice == "ridge":
            final_model = ridge_all
        elif choice == "mlp":
            final_model = mlp
        else:
            final_model = HybridModel(mlp, lda_all, best_alpha)

        final_model = final_model.to(device)
        final_model.eval()

        # Hard safety check
        if _param_count_trainable(final_model) > param_limit:
            # fall back to LDA (0 trainable params)
            final_model = lda_all.to(device)
            final_model.eval()

        return final_model