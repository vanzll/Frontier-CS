import os
import math
import copy
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _compute_mean_std(train_loader) -> Tuple[torch.Tensor, torch.Tensor]:
    n = 0
    s = None
    ss = None
    for xb, _ in train_loader:
        if not torch.is_tensor(xb):
            xb = torch.tensor(xb)
        xb = xb.detach().cpu()
        if xb.dim() > 2:
            xb = xb.view(xb.size(0), -1)
        xb = xb.to(torch.float64)
        if s is None:
            s = xb.sum(dim=0)
            ss = (xb * xb).sum(dim=0)
        else:
            s += xb.sum(dim=0)
            ss += (xb * xb).sum(dim=0)
        n += xb.size(0)

    if n == 0:
        raise ValueError("Empty train_loader")

    mean = s / n
    var = ss / n - mean * mean
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)
    return mean.to(torch.float32), std.to(torch.float32)


class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        mean: torch.Tensor,
        std: torch.Tensor,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1).clone())
        self.register_buffer("std", std.view(1, -1).clone())

        dims = [input_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList([_Block(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)])
        last_dim = dims[-1] if len(dims) > 1 else input_dim
        self.head = nn.Linear(last_dim, num_classes)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(torch.float32)
        return (x - self.mean) / (self.std + 1e-6)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.head(x)


def _param_count_for_hidden(input_dim: int, num_classes: int, hidden_dims: List[int]) -> int:
    total = 0
    prev = input_dim
    for d in hidden_dims:
        total += prev * d + d  # Linear
        total += 2 * d         # LayerNorm affine
        prev = d
    total += prev * num_classes + num_classes  # Head
    return total


def _choose_hidden_dims(input_dim: int, num_classes: int, param_limit: int) -> List[int]:
    # Prefer deeper models if capacity fits; maximize parameter usage under limit.
    h1_list = [288, 272, 264, 256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 112, 96]
    h2_list = [256, 240, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 112, 96]
    h3_list = [224, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 112, 96, 80, 64]

    best = None
    best_params = -1
    best_depth = -1

    # depth 3
    for h1 in h1_list:
        for h2 in h2_list:
            if h2 > h1:
                continue
            for h3 in h3_list:
                if h3 > h2:
                    continue
                dims = [h1, h2, h3]
                p = _param_count_for_hidden(input_dim, num_classes, dims)
                if p <= param_limit and (p > best_params or (p == best_params and 3 > best_depth)):
                    best = dims
                    best_params = p
                    best_depth = 3

    # depth 2
    for h1 in h1_list:
        for h2 in h2_list:
            if h2 > h1:
                continue
            dims = [h1, h2]
            p = _param_count_for_hidden(input_dim, num_classes, dims)
            if p <= param_limit and (p > best_params or (p == best_params and 2 > best_depth)):
                best = dims
                best_params = p
                best_depth = 2

    # depth 1
    for h1 in h1_list:
        dims = [h1]
        p = _param_count_for_hidden(input_dim, num_classes, dims)
        if p <= param_limit and (p > best_params or (p == best_params and 1 > best_depth)):
            best = dims
            best_params = p
            best_depth = 1

    # linear fallback
    if best is None:
        best = []
    return best


@torch.no_grad()
def _accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.tensor(yb)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


def _fit_lda_on_features(model: _MLP, loader, num_classes: int, device: torch.device, ridge: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats = []
    labels = []
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.tensor(yb)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()
        f = model.extract_features(xb)
        feats.append(f.detach().cpu())
        labels.append(yb.detach().cpu())
    X = torch.cat(feats, dim=0).to(torch.float32)  # N x d
    y = torch.cat(labels, dim=0).to(torch.long)    # N

    N, d = X.shape
    counts = torch.bincount(y, minlength=num_classes).to(torch.float32)
    counts_safe = counts.clone()
    counts_safe[counts_safe == 0] = 1.0
    mu = torch.zeros(num_classes, d, dtype=torch.float32)
    mu.index_add_(0, y, X)
    mu = mu / counts_safe.unsqueeze(1)

    Xc = X - mu[y]
    denom = max(1, N - num_classes)
    cov = (Xc.t().matmul(Xc)) / float(denom)

    tr = torch.trace(cov).item()
    scale = (tr / max(1, d)) if tr > 0 else 1.0
    cov = cov + (ridge * scale + 1e-6) * torch.eye(d, dtype=torch.float32)

    invcov = torch.linalg.inv(cov)
    W = (invcov.matmul(mu.t())).t().contiguous()  # C x d
    quad = torch.einsum("cd,dd,cd->c", mu, invcov, mu)  # C
    b = -0.5 * quad
    pri = counts / max(1.0, counts.sum().item())
    b = b + torch.log(torch.clamp(pri, min=1e-12))
    return W.to(device), b.to(device)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 200000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass
        torch.manual_seed(0)

        mean, std = _compute_mean_std(train_loader)
        hidden_dims = _choose_hidden_dims(input_dim, num_classes, param_limit)

        dropout = 0.10
        model = _MLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            mean=mean,
            std=std,
            dropout=dropout,
        ).to(device)

        if _count_trainable_params(model) > param_limit:
            # As a last resort, shrink until it fits
            while hidden_dims and _count_trainable_params(model) > param_limit:
                hidden_dims = hidden_dims[:-1]  # drop last hidden layer
                model = _MLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dims=hidden_dims,
                    mean=mean,
                    std=std,
                    dropout=dropout,
                ).to(device)
            if _count_trainable_params(model) > param_limit:
                model = _MLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dims=[],
                    mean=mean,
                    std=std,
                    dropout=0.0,
                ).to(device)

        train_steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 0
        if train_steps_per_epoch <= 0:
            train_steps_per_epoch = 16
        max_epochs = 140
        total_steps = max_epochs * train_steps_per_epoch
        warmup_steps = max(10, int(0.08 * total_steps))
        base_lr = 3e-3
        weight_decay = 6e-3
        noise_std = 0.01

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.06)

        best_state = copy.deepcopy(model.state_dict())
        best_val = -1.0
        best_epoch = -1
        patience = 18
        global_step = 0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                if not torch.is_tensor(xb):
                    xb = torch.tensor(xb)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std

                # custom warmup + cosine
                global_step += 1
                if global_step <= warmup_steps:
                    lr = base_lr * (global_step / float(warmup_steps))
                else:
                    t = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_acc = _accuracy(model, val_loader, device)
            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= patience:
                break

        model.load_state_dict(best_state)

        # LDA refinement on learned feature space (only change head weights/bias)
        with torch.no_grad():
            old_w = model.head.weight.detach().clone()
            old_b = model.head.bias.detach().clone()

        try:
            W, b = _fit_lda_on_features(model, train_loader, num_classes, device=device, ridge=1e-2)
            with torch.no_grad():
                model.head.weight.copy_(W)
                model.head.bias.copy_(b)

            lda_val = _accuracy(model, val_loader, device)
            if lda_val + 1e-6 < best_val:
                with torch.no_grad():
                    model.head.weight.copy_(old_w)
                    model.head.bias.copy_(old_b)
        except Exception:
            with torch.no_grad():
                model.head.weight.copy_(old_w)
                model.head.bias.copy_(old_b)

        return model.to(device)