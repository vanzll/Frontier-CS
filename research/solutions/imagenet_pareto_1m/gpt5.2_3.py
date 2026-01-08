import os
import math
import copy
import random
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn


def _seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _gather_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected loader to yield (inputs, targets).")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


class _InputNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("std", std.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, h1: int, h2: int, mean: torch.Tensor, std: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.norm = _InputNorm(mean, std)
        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


def _estimate_norm_stats(x_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x_train.float()
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    std = torch.sqrt(var + 1e-6)
    std = torch.clamp(std, min=1e-3)
    return mean, std


def _net_param_count(input_dim: int, num_classes: int, h1: int, h2: int) -> int:
    # Trainable params for:
    # fc1: input_dim*h1 + h1
    # bn1: 2*h1
    # fc2: h1*h2 + h2
    # bn2: 2*h2
    # fc3: h2*num_classes + num_classes
    return (input_dim * h1 + h1) + (2 * h1) + (h1 * h2 + h2) + (2 * h2) + (h2 * num_classes + num_classes)


def _choose_dims(input_dim: int, num_classes: int, limit: int) -> Tuple[int, int]:
    # Prefer larger dims but keep within limit with a small safety margin.
    margin = 256
    best = None
    best_score = -1

    # Candidates geared for this problem, but robust to minor changes.
    h1_candidates = [512, 640, 768, 896, 1024, 1152, 1280]
    h2_candidates = [256, 320, 384, 448, 512, 576, 640, 704, 768]

    for h1 in h1_candidates:
        for h2 in h2_candidates:
            if h2 > h1:
                continue
            p = _net_param_count(input_dim, num_classes, h1, h2)
            if p <= limit - margin:
                score = p  # maximize parameter usage
                if score > best_score:
                    best_score = score
                    best = (h1, h2)

    if best is not None:
        return best

    # Fallback: minimal viable dims.
    h1, h2 = 256, 256
    while _net_param_count(input_dim, num_classes, h1, h2) > limit and (h1 > 64 and h2 > 64):
        h1 = max(64, h1 - 64)
        h2 = min(h2, h1)
        h2 = max(64, h2 - 64)
    h2 = min(h2, h1)
    return h1, h2


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512, device: str = "cpu") -> float:
    model.eval()
    n = x.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size].to(device=device, dtype=torch.float32)
        yb = y[i:i + batch_size].to(device=device, dtype=torch.long)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / max(1, n)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        _seed_all(0)

        if metadata is None:
            metadata = {}
        device = metadata.get("device", "cpu")
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 1_000_000))

        try:
            n_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
        except Exception:
            n_threads = 8
        try:
            torch.set_num_threads(max(1, min(8, n_threads)))
        except Exception:
            pass

        x_train, y_train = _gather_from_loader(train_loader)
        if val_loader is not None:
            x_val, y_val = _gather_from_loader(val_loader)
        else:
            # simple split if no val loader
            n = x_train.shape[0]
            idx = torch.randperm(n)
            split = max(1, int(0.2 * n))
            val_idx = idx[:split]
            tr_idx = idx[split:]
            x_val, y_val = x_train[val_idx], y_train[val_idx]
            x_train, y_train = x_train[tr_idx], y_train[tr_idx]

        if x_train.ndim != 2 or x_train.shape[1] != input_dim:
            input_dim = int(x_train.shape[1])

        mean, std = _estimate_norm_stats(x_train)

        h1, h2 = _choose_dims(input_dim, num_classes, param_limit)
        model = _MLPNet(input_dim, num_classes, h1, h2, mean, std, dropout=0.10).to(device)

        # Hard constraint check
        if _count_trainable_params(model) > param_limit:
            # Reduce dims defensively
            h1, h2 = 512, 256
            model = _MLPNet(input_dim, num_classes, h1, h2, mean, std, dropout=0.10).to(device)
            if _count_trainable_params(model) > param_limit:
                # Last resort: small model
                h1, h2 = 256, 128
                model = _MLPNet(input_dim, num_classes, h1, h2, mean, std, dropout=0.10).to(device)

        # Build training loader from tensors for speed and control
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        val_ds = torch.utils.data.TensorDataset(x_val, y_val)

        batch_size = 256
        if x_train.shape[0] <= 512:
            batch_size = 128
        if x_train.shape[0] <= 256:
            batch_size = 64

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        val_x = x_val
        val_y = y_val

        # Optimizer and schedule
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or (".bn" in name) or ("bn" in name):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-2}, {"params": no_decay, "weight_decay": 0.0}],
            lr=3e-3,
            betas=(0.9, 0.99),
            eps=1e-8,
        )

        total_epochs = 180
        steps_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = max(10, int(0.05 * total_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_acc = -1.0
        best_state = None
        patience = 30
        bad = 0
        min_epochs = 40

        use_swa = True
        swa_start = int(0.65 * total_epochs)
        swa_model = torch.optim.swa_utils.AveragedModel(model) if use_swa else None
        swa_n = 0

        global_step = 0
        noise_std = 0.02

        for epoch in range(total_epochs):
            model.train()
            for xb, yb in train_dl:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)

                if noise_std > 0:
                    xb = xb + noise_std * torch.randn_like(xb)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_n += 1

            # validation
            va = _accuracy(model, val_x, val_y, batch_size=512, device=device)

            improved = va > best_acc + 1e-5
            if improved:
                best_acc = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if epoch >= min_epochs and bad >= patience:
                break

        # Candidate models: best checkpoint, and SWA (if available)
        best_model = model
        if best_state is not None:
            best_model.load_state_dict(best_state, strict=True)

        if use_swa and swa_n > 0:
            swa_candidate = copy.deepcopy(best_model)
            swa_candidate.load_state_dict(swa_model.module.state_dict(), strict=False)
            try:
                torch.optim.swa_utils.update_bn(train_dl, swa_candidate, device=device)
            except Exception:
                pass
            acc_best = _accuracy(best_model, val_x, val_y, batch_size=512, device=device)
            acc_swa = _accuracy(swa_candidate, val_x, val_y, batch_size=512, device=device)
            if acc_swa >= acc_best - 1e-6:
                best_model = swa_candidate

        # Final safety: ensure within param limit
        if _count_trainable_params(best_model) > param_limit:
            # Fallback to a guaranteed-small model
            h1, h2 = 512, 256
            best_model = _MLPNet(input_dim, num_classes, h1, h2, mean, std, dropout=0.10).to(device)
            # quick fit
            optimizer = torch.optim.AdamW(best_model.parameters(), lr=2e-3, weight_decay=1e-2)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            best_model.train()
            for _ in range(30):
                for xb, yb in train_dl:
                    xb = xb.to(device=device, dtype=torch.float32)
                    yb = yb.to(device=device, dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(best_model(xb), yb)
                    loss.backward()
                    optimizer.step()

        best_model.eval()
        return best_model.cpu()