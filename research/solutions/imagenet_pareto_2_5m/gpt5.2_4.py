import os
import math
import copy
import time
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _collect_loader(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.as_tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.as_tensor(yb)
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=False)
        xs.append(xb)
        ys.append(yb)
    x = torch.cat(xs, dim=0) if xs else torch.empty(0, device=device, dtype=torch.float32)
    y = torch.cat(ys, dim=0) if ys else torch.empty(0, device=device, dtype=torch.long)
    return x, y


def _standardize_fit(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    var = (x - mean).pow(2).mean(dim=0)
    std = torch.sqrt(var + eps)
    std = torch.clamp(std, min=eps)
    return mean, std


def _standardize_apply(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _class_means(x: torch.Tensor, y: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n, d = x.shape
    means = x.new_zeros((num_classes, d))
    counts = x.new_zeros((num_classes,))
    means.index_add_(0, y, x)
    ones = x.new_ones((n,))
    counts.index_add_(0, y, ones)
    denom = counts.clamp(min=1.0).unsqueeze(1)
    means = means / denom
    return means, counts


def _try_cholesky(cov: torch.Tensor, reg: float, max_tries: int = 8) -> torch.Tensor:
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cur = reg
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(cov + cur * eye)
        except Exception:
            cur *= 10.0
    return torch.linalg.cholesky(cov + cur * eye)


def _fit_lda(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    reg: float = 1e-3,
    shrink: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x64 = x.to(dtype=torch.float64)
    means, counts = _class_means(x64, y, num_classes)
    centered = x64 - means[y]
    n = x64.shape[0]
    cov = (centered.t() @ centered) / max(1, n)
    if shrink > 0.0:
        tr = cov.diag().mean()
        cov = (1.0 - shrink) * cov + shrink * tr * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    L = _try_cholesky(cov, reg)
    W = torch.cholesky_solve(means.t(), L).t()
    quad = (means * W).sum(dim=1)
    priors = counts / counts.sum().clamp(min=1.0)
    b = -0.5 * quad + torch.log(priors.clamp(min=1e-12))
    return W.to(dtype=torch.float32), b.to(dtype=torch.float32)


@torch.no_grad()
def _acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


class _MLPHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, h1: int = 1792, h2: int = 896, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.LayerNorm(h1),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.LayerNorm(h2),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FinalModel(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        num_classes: int,
        lda_W: Optional[torch.Tensor] = None,
        lda_b: Optional[torch.Tensor] = None,
        mlp_head: Optional[nn.Module] = None,
        beta_lda: float = 0.0,
        use_mlp: bool = True,
        use_lda: bool = True,
    ):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone().to(dtype=torch.float32))
        inv_std = (1.0 / std.detach().clone().to(dtype=torch.float32)).clamp(max=1e6)
        self.register_buffer("inv_std", inv_std)

        self.use_mlp = bool(use_mlp and (mlp_head is not None))
        self.use_lda = bool(use_lda and (lda_W is not None) and (lda_b is not None))

        if self.use_mlp:
            self.mlp = mlp_head
        else:
            self.mlp = None

        if self.use_lda:
            self.register_buffer("lda_W", lda_W.detach().clone().to(dtype=torch.float32))
            self.register_buffer("lda_b", lda_b.detach().clone().to(dtype=torch.float32))
        else:
            self.register_buffer("lda_W", torch.empty((num_classes, mean.numel()), dtype=torch.float32))
            self.register_buffer("lda_b", torch.empty((num_classes,), dtype=torch.float32))

        self.register_buffer("beta_lda", torch.tensor(float(beta_lda), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = (x - self.mean) * self.inv_std
        logits = None
        if self.use_mlp:
            logits = self.mlp(x)
        if self.use_lda:
            lda_logits = x @ self.lda_W.t() + self.lda_b
            if logits is None:
                logits = lda_logits
            else:
                logits = logits + self.beta_lda * lda_logits
        return logits


def _train_mlp(
    head: nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    max_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    noise_std: float = 0.03,
    patience: int = 25,
) -> Tuple[nn.Module, float]:
    device = xtr.device
    head = head.to(device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    n = xtr.shape[0]
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = max_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=lr * 0.05)

    best_state = None
    best_acc = -1.0
    best_epoch = -1

    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        ns = noise_std * (0.7 ** (epoch / max(1.0, max_epochs / 10.0)))
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = xtr[idx]
            if ns > 0:
                xb = xb + torch.randn_like(xb) * ns
            yb = ytr[idx]
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=2.0)
            opt.step()
            sched.step()

        head.eval()
        with torch.inference_mode():
            logits_va = head(xva)
            acc = _acc_from_logits(logits_va, yva)

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, float(best_acc)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2_500_000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass
        torch.manual_seed(0)

        xtr_raw, ytr = _collect_loader(train_loader, device=device)
        xva_raw, yva = _collect_loader(val_loader, device=device)

        mean, std = _standardize_fit(xtr_raw)
        xtr = _standardize_apply(xtr_raw, mean, std)
        xva = _standardize_apply(xva_raw, mean, std)

        reg_candidates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        best_reg = reg_candidates[0]
        best_lda_acc = -1.0
        best_lda_W = None
        best_lda_b = None

        for reg in reg_candidates:
            try:
                W, b = _fit_lda(xtr, ytr, num_classes=num_classes, reg=reg, shrink=0.05)
            except Exception:
                continue
            with torch.inference_mode():
                logits_va = xva @ W.t() + b
                acc = _acc_from_logits(logits_va, yva)
            if acc > best_lda_acc + 1e-6 or (abs(acc - best_lda_acc) <= 1e-6 and reg < best_reg):
                best_lda_acc = acc
                best_reg = reg
                best_lda_W, best_lda_b = W, b

        if best_lda_W is None:
            best_lda_W = torch.zeros((num_classes, input_dim), device=device, dtype=torch.float32)
            best_lda_b = torch.zeros((num_classes,), device=device, dtype=torch.float32)
            best_lda_acc = 0.0
            best_reg = 1e-3

        xall = torch.cat([xtr, xva], dim=0)
        yall = torch.cat([ytr, yva], dim=0)
        try:
            lda_W_final, lda_b_final = _fit_lda(xall, yall, num_classes=num_classes, reg=best_reg, shrink=0.05)
        except Exception:
            lda_W_final, lda_b_final = best_lda_W, best_lda_b

        if best_lda_acc >= 0.97:
            model = _FinalModel(
                mean=mean,
                std=std,
                num_classes=num_classes,
                lda_W=lda_W_final,
                lda_b=lda_b_final,
                mlp_head=None,
                beta_lda=0.0,
                use_mlp=False,
                use_lda=True,
            ).to(device)
            if _count_trainable_params(model) > param_limit:
                for p in model.parameters():
                    p.requires_grad_(False)
            model.eval()
            return model

        mlp_head = _MLPHead(input_dim=input_dim, num_classes=num_classes, h1=1792, h2=896, dropout=0.05).to(device)
        if _count_trainable_params(mlp_head) > param_limit:
            mlp_head = _MLPHead(input_dim=input_dim, num_classes=num_classes, h1=1536, h2=768, dropout=0.05).to(device)

        mlp_head, mlp_val_acc = _train_mlp(
            mlp_head,
            xtr=xtr,
            ytr=ytr,
            xva=xva,
            yva=yva,
            max_epochs=220,
            batch_size=256,
            lr=2e-3,
            weight_decay=1e-4,
            label_smoothing=0.05,
            noise_std=0.03,
            patience=30,
        )

        with torch.inference_mode():
            lda_logits_va = xva @ best_lda_W.t() + best_lda_b
            mlp_logits_va = mlp_head(xva)

        acc_mlp = _acc_from_logits(mlp_logits_va, yva)
        acc_lda = _acc_from_logits(lda_logits_va, yva)

        betas = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.4, 2.0, 3.0, 4.0]
        best_beta = 0.0
        best_mix_acc = -1.0
        for beta in betas:
            logits = mlp_logits_va + float(beta) * lda_logits_va
            acc = _acc_from_logits(logits, yva)
            if acc > best_mix_acc + 1e-6:
                best_mix_acc = acc
                best_beta = float(beta)

        choice = "mix"
        best_val = best_mix_acc
        if acc_lda >= best_val + 1e-6:
            choice = "lda"
            best_val = acc_lda
        if acc_mlp >= best_val + 1e-6:
            choice = "mlp"
            best_val = acc_mlp

        if choice == "lda":
            model = _FinalModel(
                mean=mean,
                std=std,
                num_classes=num_classes,
                lda_W=lda_W_final,
                lda_b=lda_b_final,
                mlp_head=None,
                beta_lda=0.0,
                use_mlp=False,
                use_lda=True,
            ).to(device)
        elif choice == "mlp":
            model = _FinalModel(
                mean=mean,
                std=std,
                num_classes=num_classes,
                lda_W=None,
                lda_b=None,
                mlp_head=mlp_head,
                beta_lda=0.0,
                use_mlp=True,
                use_lda=False,
            ).to(device)
        else:
            model = _FinalModel(
                mean=mean,
                std=std,
                num_classes=num_classes,
                lda_W=lda_W_final,
                lda_b=lda_b_final,
                mlp_head=mlp_head,
                beta_lda=best_beta,
                use_mlp=True,
                use_lda=True,
            ).to(device)

        if _count_trainable_params(model) > param_limit:
            model = _FinalModel(
                mean=mean,
                std=std,
                num_classes=num_classes,
                lda_W=lda_W_final,
                lda_b=lda_b_final,
                mlp_head=None,
                beta_lda=0.0,
                use_mlp=False,
                use_lda=True,
            ).to(device)

        model.eval()
        return model