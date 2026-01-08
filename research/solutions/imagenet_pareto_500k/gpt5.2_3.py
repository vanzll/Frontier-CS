import math
import os
import time
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _collect_loader(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        xs.append(x.to(device=device, dtype=torch.float32))
        ys.append(y.to(device=device, dtype=torch.long))
    if not xs:
        raise ValueError("Empty loader")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class _ResidualBottleneck(nn.Module):
    def __init__(self, width: int, bottleneck: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, bottleneck, bias=True)
        self.fc2 = nn.Linear(bottleneck, width, bias=True)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class _MLPResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int,
        n_blocks: int,
        bottleneck: int,
        dropout: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("x_mean", mean.clone().detach())
        self.register_buffer("x_std", std.clone().detach())
        self.in_ln = nn.LayerNorm(input_dim)
        self.fc_in = nn.Linear(input_dim, width, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.drop_in = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_ResidualBottleneck(width, bottleneck, dropout) for _ in range(n_blocks)])
        self.out_ln = nn.LayerNorm(width)
        self.fc_out = nn.Linear(width, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = (x - self.x_mean) / self.x_std
        x = self.in_ln(x)
        x = self.fc_in(x)
        x = self.act(x)
        x = self.drop_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_ln(x)
        x = self.fc_out(x)
        return x


def _find_max_width_for_limit(
    input_dim: int,
    num_classes: int,
    param_limit: int,
    n_blocks: int,
    dropout: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    width_low: int = 64,
    width_high: int = 1024,
) -> Tuple[int, int]:
    def make(width: int) -> _MLPResNet:
        bottleneck = max(16, int(math.ceil(width / 4)))
        return _MLPResNet(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            n_blocks=n_blocks,
            bottleneck=bottleneck,
            dropout=dropout,
            mean=mean,
            std=std,
        )

    lo, hi = width_low, width_high
    best_w = lo
    best_b = max(16, int(math.ceil(lo / 4)))
    while lo <= hi:
        mid = (lo + hi) // 2
        model = make(mid)
        pc = _count_trainable_params(model)
        if pc <= param_limit:
            best_w = mid
            best_b = max(16, int(math.ceil(mid / 4)))
            lo = mid + 1
        else:
            hi = mid - 1
    return best_w, best_b


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> float:
    model.eval()
    n = x.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / max(1, n)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500_000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        try:
            num_threads = min(8, os.cpu_count() or 8)
            torch.set_num_threads(num_threads)
        except Exception:
            pass

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_train, y_train = _collect_loader(train_loader, device=device)
        X_val, y_val = _collect_loader(val_loader, device=device)

        if X_train.ndim != 2 or X_train.shape[1] != input_dim:
            X_train = X_train.view(X_train.shape[0], -1)
        if X_val.ndim != 2 or X_val.shape[1] != input_dim:
            X_val = X_val.view(X_val.shape[0], -1)

        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0, unbiased=False)
        std = std.clamp_min(1e-3)

        dropout = 0.10
        n_blocks = 4
        width, bottleneck = _find_max_width_for_limit(
            input_dim=input_dim,
            num_classes=num_classes,
            param_limit=param_limit,
            n_blocks=n_blocks,
            dropout=dropout,
            mean=mean,
            std=std,
            width_low=128,
            width_high=768,
        )

        model = _MLPResNet(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            n_blocks=n_blocks,
            bottleneck=bottleneck,
            dropout=dropout,
            mean=mean,
            std=std,
        ).to(device)

        pc = _count_trainable_params(model)
        if pc > param_limit:
            while pc > param_limit and width > 64:
                width -= 1
                bottleneck = max(16, int(math.ceil(width / 4)))
                model = _MLPResNet(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    width=width,
                    n_blocks=n_blocks,
                    bottleneck=bottleneck,
                    dropout=dropout,
                    mean=mean,
                    std=std,
                ).to(device)
                pc = _count_trainable_params(model)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2, betas=(0.9, 0.99))

        batch_size = 128
        n_train = X_train.shape[0]
        steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
        max_epochs = 80
        total_steps = steps_per_epoch * max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=6e-3,
            total_steps=total_steps,
            pct_start=0.12,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=30.0,
        )

        mixup_alpha = 0.20
        mixup_prob = 0.35
        noise_std = 0.01

        best_state = None
        best_val = -1.0
        patience = 12
        bad_epochs = 0

        start_time = time.time()
        time_budget_s = 300.0

        model.train()
        for epoch in range(max_epochs):
            if time.time() - start_time > time_budget_s:
                break

            perm = torch.randperm(n_train, device=device)
            for si in range(0, n_train, batch_size):
                idx = perm[si : si + batch_size]
                xb = X_train[idx]
                yb = y_train[idx]

                if noise_std > 0:
                    xb = xb + noise_std * torch.randn_like(xb)

                if mixup_alpha > 0 and np.random.rand() < mixup_prob and xb.shape[0] >= 2:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    rand_idx = torch.randperm(xb.shape[0], device=device)
                    xb2 = xb[rand_idx]
                    yb2 = yb[rand_idx]
                    xb = lam * xb + (1.0 - lam) * xb2
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

            val_acc = _accuracy(model, X_val, y_val, batch_size=512)
            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        model.eval()
        return model