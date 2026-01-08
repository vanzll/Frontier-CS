import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _collect_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.tensor(yb)
        xs.append(xb.detach().cpu())
        ys.append(yb.detach().cpu())
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0).long()
    return x, y


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.inference_mode()
def _eval_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> float:
    model.eval()
    n = x.shape[0]
    correct = 0
    total = 0
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


def _fit_lda(x: torch.Tensor, y: torch.Tensor, num_classes: int, shrinkage: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    x64 = x.to(torch.float64)
    y = y.long()
    n, d = x64.shape
    c = num_classes

    counts = torch.bincount(y, minlength=c).to(torch.float64)
    counts_clamped = counts.clamp_min(1.0)

    means = torch.zeros((c, d), dtype=torch.float64)
    means.index_add_(0, y, x64)
    means = means / counts_clamped[:, None]

    centered = x64 - means[y]
    denom = max(1, int(n - c))
    cov = (centered.transpose(0, 1) @ centered) / float(denom)

    tr = torch.trace(cov)
    eye = torch.eye(d, dtype=torch.float64)
    cov = (1.0 - shrinkage) * cov + shrinkage * eye * (tr / float(d))
    cov = cov + eye * 1e-6

    # weights: d x c
    weights = torch.linalg.solve(cov, means.transpose(0, 1))
    # bias: c
    bias = -0.5 * (means * weights.transpose(0, 1)).sum(dim=1)

    return weights.to(torch.float32), bias.to(torch.float32)


class InputNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        mean = mean.detach().cpu().to(torch.float32)
        std = std.detach().cpu().to(torch.float32)
        std = std.clamp_min(1e-6)
        self.register_buffer("mean", mean)
        self.register_buffer("inv_std", 1.0 / std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.ln1(h)
        h = self.drop(h)

        h2 = self.fc2(h)
        h2 = F.gelu(h2)
        h = h + h2
        h = self.ln2(h)
        h = self.drop(h)

        return self.fc3(h)


class FixedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FullModel(nn.Module):
    def __init__(
        self,
        norm: InputNorm,
        lda: FixedLinear,
        mlp: Optional[MLP],
        alpha: float,
        mode: str,
    ):
        super().__init__()
        self.norm = norm
        self.lda = lda
        self.mlp = mlp
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.mode = mode  # "lda", "mlp", "ens"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.norm(x)
        if self.mode == "lda":
            return self.lda(x)
        if self.mode == "mlp":
            return self.mlp(x)
        a = float(self.alpha.item())
        return a * self.mlp(x) + (1.0 - a) * self.lda(x)


def _compute_max_hidden(input_dim: int, num_classes: int, param_limit: int, safety_margin: int = 4096) -> int:
    # For MLP with 2 LayerNorms:
    # params = h^2 + h*(d + C + 6) + C
    d = int(input_dim)
    c = int(num_classes)
    limit = int(max(1, param_limit - safety_margin))
    b = d + c + 6
    # Solve h^2 + b*h + c - limit <= 0
    disc = b * b + 4 * (limit - c)
    if disc <= 0:
        return 64
    h = int(math.floor((-b + math.sqrt(disc)) / 2.0))
    return max(64, h)


def _train_mlp(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: Optional[torch.Tensor],
    y_val: Optional[torch.Tensor],
    max_epochs: int = 80,
    batch_size: int = 256,
    base_lr: float = 3e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.02,
    mixup_alpha: float = 0.15,
    patience: int = 12,
    min_epochs: int = 10,
) -> Tuple[nn.Module, int, float]:
    device = torch.device("cpu")
    model.to(device)
    x_train = x_train.to(device, dtype=torch.float32)
    y_train = y_train.to(device, dtype=torch.long)
    if x_val is not None:
        x_val = x_val.to(device, dtype=torch.float32)
        y_val = y_val.to(device, dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    best_state = None
    best_acc = -1.0
    best_epoch = 0
    bad = 0

    n = x_train.shape[0]

    for epoch in range(max_epochs):
        # Cosine schedule with short warmup
        warmup = 5
        if epoch < warmup:
            lr = base_lr * float(epoch + 1) / float(warmup)
        else:
            t = float(epoch - warmup) / float(max(1, max_epochs - warmup))
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)

            if mixup_alpha > 0.0 and xb.shape[0] >= 2:
                lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
                perm2 = torch.randperm(xb.shape[0], device=device)
                xb2 = xb.index_select(0, perm2)
                yb2 = yb.index_select(0, perm2)
                xmix = lam * xb + (1.0 - lam) * xb2
                logits = model(xmix)
                loss = lam * F.cross_entropy(logits, yb, label_smoothing=label_smoothing) + (1.0 - lam) * F.cross_entropy(
                    logits, yb2, label_smoothing=label_smoothing
                )
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if x_val is not None:
            val_acc = _eval_acc(model, x_val, y_val, batch_size=2048)
        else:
            val_acc = _eval_acc(model, x_train, y_train, batch_size=2048)

        if val_acc > best_acc + 1e-5:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch + 1 >= min_epochs and bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, best_epoch, float(best_acc)


@torch.inference_mode()
def _best_alpha_for_ensemble(mlp_model: nn.Module, lda_model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, float]:
    mlp_model.eval()
    lda_model.eval()
    logits_m = mlp_model(x_val)
    logits_l = lda_model(x_val)

    best_a = 1.0
    best_acc = -1.0
    # Slightly finer search near high-mlp weights
    grid = [i / 20.0 for i in range(0, 21)] + [0.85, 0.9, 0.95, 0.975]
    grid = sorted(set(max(0.0, min(1.0, a)) for a in grid))
    for a in grid:
        logits = a * logits_m + (1.0 - a) * logits_l
        acc = _accuracy_from_logits(logits, y_val)
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_a = a
    return float(best_a), float(best_acc)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2_500_000))

        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

        x_tr_raw, y_tr = _collect_from_loader(train_loader)
        if val_loader is not None:
            x_va_raw, y_va = _collect_from_loader(val_loader)
        else:
            x_va_raw, y_va = None, None

        x_tr_raw = x_tr_raw.to(torch.float32)
        if x_va_raw is not None:
            x_va_raw = x_va_raw.to(torch.float32)

        # Fit normalization on training only
        mean = x_tr_raw.mean(dim=0)
        std = x_tr_raw.std(dim=0, unbiased=False).clamp_min(1e-6)
        norm = InputNorm(mean, std)

        x_tr = norm(x_tr_raw)
        if x_va_raw is not None:
            x_va = norm(x_va_raw)
        else:
            x_va = None

        # LDA on train
        lda_w, lda_b = _fit_lda(x_tr, y_tr, num_classes=num_classes, shrinkage=0.15)
        lda = FixedLinear(lda_w.transpose(0, 1).contiguous(), lda_b.contiguous())
        lda_wrap = nn.Sequential(nn.Identity(), lda)  # for interface consistency

        # Train MLP on train, validate on val
        hidden = _compute_max_hidden(input_dim, num_classes, param_limit, safety_margin=8192)
        # Ensure model capacity isn't too small
        hidden = max(hidden, 512)

        mlp = MLP(input_dim=input_dim, hidden_dim=hidden, num_classes=num_classes, dropout=0.06)
        mlp_params = _count_trainable_params(mlp)
        if mlp_params > param_limit:
            # Back off if needed
            hidden = _compute_max_hidden(input_dim, num_classes, param_limit, safety_margin=16384)
            hidden = max(hidden, 256)
            mlp = MLP(input_dim=input_dim, hidden_dim=hidden, num_classes=num_classes, dropout=0.06)

        mlp_trained, best_epoch, _ = _train_mlp(
            mlp,
            x_tr,
            y_tr,
            x_va if x_va is not None else None,
            y_va if y_va is not None else None,
            max_epochs=90,
            batch_size=256,
            base_lr=3e-3,
            weight_decay=1.2e-4,
            label_smoothing=0.02,
            mixup_alpha=0.18,
            patience=14,
            min_epochs=12,
        )

        # Decide ensemble weight using validation if available; otherwise default
        if x_va is not None:
            val_acc_lda = _eval_acc(lda, x_va, y_va, batch_size=2048)
            val_acc_mlp = _eval_acc(mlp_trained, x_va, y_va, batch_size=2048)
            alpha, val_acc_ens = _best_alpha_for_ensemble(mlp_trained, lda, x_va, y_va)

            # Prefer best among {lda, mlp, ens}
            best_mode = "ens"
            best_val = val_acc_ens
            if val_acc_mlp >= best_val + 1e-6:
                best_mode = "mlp"
                best_val = val_acc_mlp
                alpha = 1.0
            if val_acc_lda >= best_val + 1e-6:
                best_mode = "lda"
                best_val = val_acc_lda
                alpha = 0.0
        else:
            best_mode = "ens"
            alpha = 0.92

        # Refit using train+val if available
        if x_va is not None:
            x_all = torch.cat([x_tr, x_va], dim=0)
            y_all = torch.cat([y_tr, y_va], dim=0)
        else:
            x_all = x_tr
            y_all = y_tr

        # Refit LDA on all
        lda_w2, lda_b2 = _fit_lda(x_all, y_all, num_classes=num_classes, shrinkage=0.12)
        lda2 = FixedLinear(lda_w2.transpose(0, 1).contiguous(), lda_b2.contiguous())

        # Retrain MLP on all for best_epoch epochs (or fixed if no val)
        if best_mode != "lda":
            epochs_final = int(best_epoch) if x_va is not None else 70
            epochs_final = max(15, min(120, epochs_final))
            mlp2 = MLP(input_dim=input_dim, hidden_dim=hidden, num_classes=num_classes, dropout=0.04)
            mlp2, _, _ = _train_mlp(
                mlp2,
                x_all,
                y_all,
                None,
                None,
                max_epochs=epochs_final,
                batch_size=256,
                base_lr=2.8e-3,
                weight_decay=1.2e-4,
                label_smoothing=0.015,
                mixup_alpha=0.12,
                patience=epochs_final + 1,
                min_epochs=epochs_final,
            )
        else:
            mlp2 = None

        final = FullModel(norm=norm, lda=lda2, mlp=mlp2, alpha=alpha, mode=best_mode)

        # Hard param limit check (trainable only)
        trainable = _count_trainable_params(final)
        if trainable > param_limit:
            # Fallback to LDA-only (should always be under limit)
            final = FullModel(norm=norm, lda=lda2, mlp=None, alpha=0.0, mode="lda")

        final.eval()
        return final