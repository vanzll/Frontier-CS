import math
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        return batch[0], None
    return batch, None


def _infer_dims(train_loader, metadata: Optional[dict]):
    if metadata is not None and "input_dim" in metadata and "num_classes" in metadata:
        return int(metadata["input_dim"]), int(metadata["num_classes"])
    for batch in train_loader:
        x, y = _unpack_batch(batch)
        x = torch.as_tensor(x)
        input_dim = int(x.shape[-1])
        if y is not None:
            y = torch.as_tensor(y)
            num_classes = int(y.max().item()) + 1
        else:
            num_classes = 128
        return input_dim, num_classes
    return 384, 128


def _param_count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_xy(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        x, y = _unpack_batch(batch)
        if y is None:
            continue
        x = torch.as_tensor(x, device=device)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        xs.append(x.detach().to(torch.float32))
        ys.append(y.detach())
    if not xs:
        return torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _compute_standardization(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def _compute_lda_head(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    shrinkage: float = 0.12,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x expected standardized
    device = x.device
    n, d = x.shape
    mu = torch.zeros((num_classes, d), device=device, dtype=torch.float64)
    counts = torch.zeros((num_classes,), device=device, dtype=torch.float64)

    y64 = y.to(torch.long)
    x64 = x.to(torch.float64)

    for k in range(num_classes):
        mask = (y64 == k)
        ck = mask.sum().item()
        if ck > 0:
            mu[k] = x64[mask].mean(dim=0)
            counts[k] = ck
        else:
            counts[k] = 1.0

    mu_y = mu[y64]  # (n, d)
    xc = x64 - mu_y
    # within-class covariance
    cov = (xc.T @ xc) / max(1.0, float(n))
    cov = (1.0 - shrinkage) * cov + shrinkage * torch.eye(d, device=device, dtype=torch.float64)
    cov = cov + eps * torch.eye(d, device=device, dtype=torch.float64)

    # solve for inv(cov) @ mu_k via cholesky
    L = torch.linalg.cholesky(cov)
    # inv_cov_mu: (d, K)
    inv_cov_mu = torch.cholesky_solve(mu.T, L)
    W = inv_cov_mu.T.contiguous()  # (K, d)
    # b_k = -0.5 * mu_k^T inv_cov mu_k
    quad = (mu * W).sum(dim=1)  # (K,)
    b = (-0.5 * quad).to(torch.float64)

    return W.to(torch.float32), b.to(torch.float32)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # residual scaling for stability
        self.res_scale = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = r + self.res_scale * x
        x = self.act(x)
        return x


class _HybridMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        base_dim: int,
        bottleneck_dim: int,
        num_blocks: int,
        dropout: float,
        mean: torch.Tensor,
        std: torch.Tensor,
        lda_W: Optional[torch.Tensor],
        lda_b: Optional[torch.Tensor],
    ):
        super().__init__()
        self.register_buffer("x_mean", mean.detach().clone().to(torch.float32))
        self.register_buffer("x_std", std.detach().clone().to(torch.float32))

        self.proj = nn.Linear(input_dim, base_dim, bias=True)
        self.ln0 = nn.LayerNorm(base_dim)
        self.act = nn.GELU()
        self.drop0 = nn.Dropout(dropout)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(_ResidualMLPBlock(base_dim, bottleneck_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        self.ln_final = nn.LayerNorm(base_dim)
        self.head = nn.Linear(base_dim, num_classes, bias=True)

        if lda_W is not None and lda_b is not None:
            self.register_buffer("lda_W", lda_W.detach().clone().to(torch.float32))
            self.register_buffer("lda_b", lda_b.detach().clone().to(torch.float32))
            self.lda_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.lda_W = None
            self.lda_b = None
            self.lda_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        x = x.to(torch.float32)
        xz = (x - self.x_mean) / self.x_std

        h = self.proj(xz)
        h = self.ln0(h)
        h = self.act(h)
        h = self.drop0(h)
        h = self.blocks(h)
        h = self.ln_final(h)
        logits = self.head(h)

        if self.lda_W is not None and self.lda_b is not None and self.lda_scale is not None:
            lda_logits = xz @ self.lda_W.t() + self.lda_b
            logits = logits + self.lda_scale * lda_logits

        return logits


@dataclass
class _EMA:
    decay: float
    shadow: Dict[str, torch.Tensor] = None

    def __post_init__(self):
        if self.shadow is None:
            self.shadow = {}

    @torch.no_grad()
    def init_from(self, model: nn.Module):
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            backup[name] = p.detach().clone()
            if name in self.shadow:
                p.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in backup:
                p.copy_(backup[name])


def _accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _unpack_batch(batch)
            if y is None:
                continue
            x = torch.as_tensor(x, device=device)
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    if total == 0:
        return 0.0
    return correct / total


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and ("bias" not in name) and ("ln" not in name) and ("norm" not in name):
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.995),
        eps=1e-8,
    )


def _choose_dims(input_dim: int, num_classes: int, param_limit: int, num_blocks: int = 2) -> Tuple[int, int]:
    # Search base_dim (multiple of 32) for max capacity under limit with margin.
    best = (512, 256)
    margin = 20000

    def est_params(base: int, bottleneck: int) -> int:
        # linear params include bias
        proj = (input_dim + 1) * base
        block = ((base + 1) * bottleneck) + ((bottleneck + 1) * base)
        head = (base + 1) * num_classes
        # layernorm params: 2*dim
        ln = 2 * base  # ln0
        ln += num_blocks * (2 * bottleneck + 2 * base)  # ln1, ln2 per block
        ln += 2 * base  # ln_final
        # plus lda_scale scalar maybe
        extra = 1
        return proj + num_blocks * block + head + ln + extra

    for base in range(384, 2049, 32):
        bottleneck = max(128, (base // 2 // 32) * 32)
        params = est_params(base, bottleneck)
        if params <= (param_limit - margin):
            best = (base, bottleneck)
    return best


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        _set_seed(0)

        input_dim, num_classes = _infer_dims(train_loader, metadata)
        device_str = (metadata or {}).get("device", "cpu")
        device = torch.device(device_str)

        param_limit = int((metadata or {}).get("param_limit", 2_500_000))

        # Collect data for standardization and LDA head
        x_tr, y_tr = _collect_xy(train_loader, device=device)
        x_va, y_va = (_collect_xy(val_loader, device=device) if val_loader is not None else (torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long)))

        if x_tr.numel() == 0:
            # Fallback minimal model
            model = nn.Sequential(nn.Linear(input_dim, num_classes)).to(device)
            return model

        x_all = x_tr if x_va.numel() == 0 else torch.cat([x_tr, x_va], dim=0)
        mean, std = _compute_standardization(x_all)

        # Standardize train for LDA
        x_tr_z = (x_tr - mean) / std
        lda_W, lda_b = _compute_lda_head(x_tr_z, y_tr, num_classes=num_classes, shrinkage=0.12)

        base_dim, bottleneck_dim = _choose_dims(input_dim, num_classes, param_limit, num_blocks=2)

        model = _HybridMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            base_dim=base_dim,
            bottleneck_dim=bottleneck_dim,
            num_blocks=2,
            dropout=0.10,
            mean=mean,
            std=std,
            lda_W=lda_W,
            lda_b=lda_b,
        ).to(device)

        if _param_count_trainable(model) > param_limit:
            # As a safe fallback, shrink further
            base_dim, bottleneck_dim = 640, 320
            model = _HybridMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                base_dim=base_dim,
                bottleneck_dim=bottleneck_dim,
                num_blocks=1,
                dropout=0.10,
                mean=mean,
                std=std,
                lda_W=lda_W,
                lda_b=lda_b,
            ).to(device)

        # Training setup
        lr = 2.2e-3
        wd = 1.2e-2
        optimizer = _make_optimizer(model, lr=lr, weight_decay=wd)

        epochs = 140
        warmup_epochs = 6
        min_lr_scale = 0.06

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            t = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
            return min_lr_scale + 0.5 * (1.0 - min_lr_scale) * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        label_smoothing = 0.06
        mixup_alpha = 0.10
        mixup_prob = 0.35

        ema = _EMA(decay=0.995)
        ema.init_from(model)

        best_acc = -1.0
        best_state = None
        patience = 18
        bad_epochs = 0

        start_time = time.time()
        max_seconds = 3300  # keep margin

        for epoch in range(epochs):
            if time.time() - start_time > max_seconds:
                break

            model.train()
            for batch in train_loader:
                x, y = _unpack_batch(batch)
                if y is None:
                    continue
                x = torch.as_tensor(x, device=device)
                y = torch.as_tensor(y, device=device, dtype=torch.long)
                if x.ndim > 2:
                    x = x.view(x.shape[0], -1)
                x = x.to(torch.float32)

                do_mix = (mixup_alpha > 0) and (random.random() < mixup_prob) and (x.shape[0] >= 2)
                if do_mix:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    perm = torch.randperm(x.shape[0], device=device)
                    x2 = x[perm]
                    y2 = y[perm]
                    x_mix = lam * x + (1.0 - lam) * x2
                    y1 = F.one_hot(y, num_classes=num_classes).to(torch.float32)
                    y2oh = F.one_hot(y2, num_classes=num_classes).to(torch.float32)
                    y_mix = lam * y1 + (1.0 - lam) * y2oh
                    logits = model(x_mix)
                    loss = _soft_cross_entropy(logits, y_mix)
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(model)

            scheduler.step()

            # Validate using EMA weights when possible
            if val_loader is not None:
                backup = ema.copy_to(model)
                acc = _accuracy(model, val_loader, device=device)
                ema.restore(model, backup)
            else:
                acc = 0.0

            if acc > best_acc + 1e-4:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # Finalize with EMA if it improves val
        if val_loader is not None:
            acc_raw = _accuracy(model, val_loader, device=device)
            backup = ema.copy_to(model)
            acc_ema = _accuracy(model, val_loader, device=device)
            if acc_ema + 1e-4 < acc_raw:
                ema.restore(model, backup)
            else:
                # keep EMA weights already copied
                pass

        model.eval()
        return model