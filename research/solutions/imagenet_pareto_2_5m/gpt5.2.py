import os
import math
import copy
import time
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputStandardize(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("inv_std", torch.ones(dim, dtype=torch.float32))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().to(dtype=torch.float32).view(-1)
        std = std.detach().to(dtype=torch.float32).view(-1)
        inv_std = (std.clamp_min(self.eps)).reciprocal()
        self.mean.copy_(mean)
        self.inv_std.copy_(inv_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class ResidualLinearBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05, layer_scale_init: float = 1e-2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((dim,), float(layer_scale_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = y * self.layer_scale
        return x + y


class ResMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        self.in_norm = InputStandardize(self.input_dim)

        self.skip = nn.Linear(self.input_dim, self.num_classes, bias=True)

        self.proj = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.blocks = nn.ModuleList([ResidualLinearBlock(self.hidden_dim, dropout=dropout) for _ in range(self.num_blocks)])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes, bias=True)

        self.logit_scale_skip = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.logit_scale_head = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initialize_skip_from_centroids(self, class_means_normed: torch.Tensor):
        with torch.no_grad():
            w = class_means_normed.to(dtype=torch.float32).contiguous()
            if w.shape != self.skip.weight.shape:
                return
            self.skip.weight.copy_(w)
            b = -0.5 * (w * w).sum(dim=1)
            self.skip.bias.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float32)
        x = self.in_norm(x)

        logits_skip = self.skip(x) * self.logit_scale_skip

        h = self.proj(x)
        h = F.gelu(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        logits_head = self.head(h) * self.logit_scale_head

        return logits_head + logits_skip


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.long)
        if xb.dim() > 2:
            xb = xb.view(xb.size(0), -1)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return float(correct) / float(max(1, total))


@torch.no_grad()
def _compute_stats_and_centroids(train_loader, input_dim: int, num_classes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sum_x = torch.zeros(input_dim, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(input_dim, dtype=torch.float64, device=device)
    n = 0

    class_sum = torch.zeros(num_classes, input_dim, dtype=torch.float64, device=device)
    class_cnt = torch.zeros(num_classes, dtype=torch.float64, device=device)

    for xb, yb in train_loader:
        if not torch.is_tensor(xb):
            xb = torch.tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.tensor(yb)
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.long)
        if xb.dim() > 2:
            xb = xb.view(xb.size(0), -1)
        xb64 = xb.to(dtype=torch.float64)
        sum_x += xb64.sum(dim=0)
        sum_x2 += (xb64 * xb64).sum(dim=0)
        n += xb64.size(0)

        for c in range(num_classes):
            mask = (yb == c)
            if mask.any():
                sel = xb64[mask]
                class_sum[c] += sel.sum(dim=0)
                class_cnt[c] += sel.size(0)

    mean = sum_x / max(1, n)
    var = (sum_x2 / max(1, n)) - mean * mean
    var = var.clamp_min(1e-8)
    std = var.sqrt()

    class_means = class_sum / class_cnt.clamp_min(1.0).unsqueeze(1)
    class_means_normed = (class_means - mean.unsqueeze(0)) / std.unsqueeze(0)

    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32), class_means_normed.to(dtype=torch.float32)


def _build_model_under_limit(
    input_dim: int,
    num_classes: int,
    param_limit: int,
    dropout: float = 0.05,
    blocks_candidates: Optional[List[int]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if blocks_candidates is None:
        blocks_candidates = [5, 6, 7]

    best = None
    best_info = None

    for B in blocks_candidates:
        lo, hi = 64, 2048
        best_H = None
        best_cnt = None
        while lo <= hi:
            mid = (lo + hi) // 2
            m = ResMLPClassifier(input_dim, num_classes, mid, B, dropout=dropout)
            cnt = _count_trainable_params(m)
            if cnt <= param_limit:
                best_H = mid
                best_cnt = cnt
                lo = mid + 1
            else:
                hi = mid - 1

        if best_H is None:
            continue

        m = ResMLPClassifier(input_dim, num_classes, best_H, B, dropout=dropout)
        cnt = _count_trainable_params(m)
        if cnt <= param_limit:
            if (best is None) or (cnt > best_info["param_count"]):
                best = m
                best_info = {"hidden_dim": best_H, "num_blocks": B, "param_count": cnt}

    if best is None:
        best = ResMLPClassifier(input_dim, num_classes, hidden_dim=128, num_blocks=2, dropout=dropout)
        best_info = {"hidden_dim": 128, "num_blocks": 2, "param_count": _count_trainable_params(best)}

    return best, best_info


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        metadata = metadata or {}
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

        model, _info = _build_model_under_limit(
            input_dim=input_dim,
            num_classes=num_classes,
            param_limit=param_limit,
            dropout=0.05,
            blocks_candidates=[5, 6, 7],
        )
        model.to(device)

        mean, std, class_means_normed = _compute_stats_and_centroids(train_loader, input_dim, num_classes, device=device)
        model.in_norm.set_stats(mean, std)
        model.initialize_skip_from_centroids(class_means_normed)

        if _count_trainable_params(model) > param_limit:
            for p in model.parameters():
                p.requires_grad_(False)
            return model.to(torch.device("cpu"))

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        max_epochs = 140
        patience = 25

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02, betas=(0.9, 0.99))

        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-3,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=200.0,
            anneal_strategy="cos",
        )

        best_state = None
        best_val = -1.0
        best_epoch = -1

        no_improve = 0
        t0 = time.time()

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                if not torch.is_tensor(xb):
                    xb = torch.tensor(xb)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb)
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)
                if xb.dim() > 2:
                    xb = xb.view(xb.size(0), -1)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            val_acc = _accuracy(model, val_loader, device)
            if val_acc > best_val + 1e-5:
                best_val = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

            if (time.time() - t0) > 3300:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        model.to(torch.device("cpu"))
        return model