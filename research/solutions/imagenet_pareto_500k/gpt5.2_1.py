import os
import math
import copy
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loader_to_tensors(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected DataLoader to yield (inputs, targets).")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        xs.append(x.detach().to(device=device, dtype=torch.float32, non_blocking=False))
        ys.append(y.detach().to(device=device, dtype=torch.long, non_blocking=False))
    if not xs:
        raise ValueError("Empty loader.")
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return X, y


@torch.no_grad()
def _accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> float:
    was_training = model.training
    model.eval()
    n = X.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == y[i:i + batch_size]).sum().item()
    if was_training:
        model.train()
    return correct / max(1, n)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, init_scale: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2.0, dim=1, eps=1e-6)
        w = F.normalize(self.weight, p=2.0, dim=1, eps=1e-6)
        return (x @ w.t()) * self.scale.clamp(1.0, 200.0)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(self.norm(x))
        y = F.gelu(y, approximate="tanh")
        y = self.dropout(y)
        return x + y


class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        h1: int,
        h2: int,
        blocks: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
        aug_noise_std: float = 0.0,
        cosine_head: bool = True,
    ):
        super().__init__()
        self.register_buffer("mu", mu.view(1, -1).contiguous())
        self.register_buffer("sigma", sigma.view(1, -1).contiguous())
        self.aug_noise_std = float(aug_noise_std)

        self.in_norm = nn.LayerNorm(input_dim)
        self.in_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

        self.fc1 = nn.Linear(input_dim, h1)
        self.norm1 = nn.LayerNorm(h1)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([ResidualBlock(h2, dropout=dropout) for _ in range(blocks)])
        self.out_norm = nn.LayerNorm(h2)

        if cosine_head:
            self.head = CosineClassifier(h2, num_classes, init_scale=12.0)
        else:
            self.head = nn.Linear(h2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = (x - self.mu) / self.sigma
        x = self.in_norm(x)
        if self.training and self.aug_noise_std > 0:
            x = x + self.aug_noise_std * torch.randn_like(x)
        x = self.in_dropout(x)

        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.drop1(x)

        x = self.norm1(x)
        x = self.fc2(x)
        x = F.gelu(x, approximate="tanh")
        x = self.drop2(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.out_norm(x)
        x = self.head(x)
        return x


def _compute_norm_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X.mean(dim=0)
    sigma = X.std(dim=0, unbiased=False)
    sigma = sigma.clamp_min(1e-4)
    return mu, sigma


def _build_model_under_limit(
    input_dim: int,
    num_classes: int,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    param_limit: int,
    blocks: int,
    h2: int,
    dropout: float,
    input_dropout: float,
    aug_noise_std: float,
    cosine_head: bool,
) -> MLPNet:
    in_dim = int(input_dim)
    nc = int(num_classes)
    B = int(blocks)
    h2 = int(h2)

    def estimate_h1_max() -> int:
        head_extra = 1 if cosine_head else nc
        C = (2 * in_dim) + (B * (h2 * h2 + 3 * h2)) + (3 * h2) + (h2 * nc) + head_extra
        denom = in_dim + 3 + h2
        if denom <= 0:
            return 0
        return (param_limit - C) // denom

    h1 = int(estimate_h1_max())
    h1 = max(64, h1)
    h1 = (h1 // 8) * 8
    h1 = max(64, h1)

    model = MLPNet(
        input_dim=in_dim,
        num_classes=nc,
        h1=h1,
        h2=h2,
        blocks=B,
        mu=mu,
        sigma=sigma,
        dropout=dropout,
        input_dropout=input_dropout,
        aug_noise_std=aug_noise_std,
        cosine_head=cosine_head,
    )
    pc = _count_trainable_params(model)

    if pc > param_limit:
        for _ in range(200):
            h1 = max(64, h1 - 8)
            model = MLPNet(
                input_dim=in_dim,
                num_classes=nc,
                h1=h1,
                h2=h2,
                blocks=B,
                mu=mu,
                sigma=sigma,
                dropout=dropout,
                input_dropout=input_dropout,
                aug_noise_std=aug_noise_std,
                cosine_head=cosine_head,
            )
            pc = _count_trainable_params(model)
            if pc <= param_limit:
                break

    if pc > param_limit:
        raise RuntimeError("Unable to build a model within the parameter limit.")

    return model


def _train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor],
    y_val: Optional[torch.Tensor],
    max_epochs: int = 200,
    batch_size: int = 256,
    lr_max: float = 3e-3,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    patience: int = 30,
) -> Tuple[Dict[str, torch.Tensor], float]:
    n_train = X_train.shape[0]
    if n_train <= 0:
        raise ValueError("No training samples.")
    batch_size = int(min(batch_size, n_train))

    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr_max), weight_decay=float(weight_decay), betas=(0.9, 0.95))

    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = max(1, steps_per_epoch * max_epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(lr_max),
        total_steps=total_steps,
        pct_start=0.12,
        anneal_strategy="cos",
        div_factor=12.0,
        final_div_factor=120.0,
    )

    best_acc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=X_train.device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            scheduler.step()

        if X_val is not None and y_val is not None:
            acc = _accuracy(model, X_val, y_val, batch_size=1024)
            if acc > best_acc + 1e-4:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch >= max(10, patience) and bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_acc = _accuracy(model, X_val, y_val, batch_size=1024) if (X_val is not None and y_val is not None) else -1.0

    return best_state, float(best_acc)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        try:
            cpu_cnt = os.cpu_count() or 8
            torch.set_num_threads(min(8, max(1, cpu_cnt)))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500000))

        X_train, y_train = _loader_to_tensors(train_loader, device=device)
        X_val, y_val = (None, None)
        if val_loader is not None:
            X_val, y_val = _loader_to_tensors(val_loader, device=device)

        X_stats = X_train if X_val is None else torch.cat([X_train, X_val], dim=0)
        mu, sigma = _compute_norm_stats(X_stats)

        candidates = [
            {"blocks": 3, "h2": 208, "dropout": 0.10, "input_dropout": 0.03, "aug_noise_std": 0.06},
            {"blocks": 2, "h2": 240, "dropout": 0.10, "input_dropout": 0.03, "aug_noise_std": 0.06},
        ]

        best_state = None
        best_val_acc = -1.0
        best_model = None

        for cfg in candidates:
            model = _build_model_under_limit(
                input_dim=input_dim,
                num_classes=num_classes,
                mu=mu,
                sigma=sigma,
                param_limit=param_limit,
                blocks=int(cfg["blocks"]),
                h2=int(cfg["h2"]),
                dropout=float(cfg["dropout"]),
                input_dropout=float(cfg["input_dropout"]),
                aug_noise_std=float(cfg["aug_noise_std"]),
                cosine_head=True,
            ).to(device)

            if _count_trainable_params(model) > param_limit:
                continue

            state, val_acc = _train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                max_epochs=220,
                batch_size=256,
                lr_max=3.2e-3 if cfg["blocks"] <= 2 else 2.8e-3,
                weight_decay=1.2e-2,
                label_smoothing=0.10,
                grad_clip=1.0,
                patience=35,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = state
                best_model = model

        if best_model is None:
            best_model = _build_model_under_limit(
                input_dim=input_dim,
                num_classes=num_classes,
                mu=mu,
                sigma=sigma,
                param_limit=param_limit,
                blocks=2,
                h2=240,
                dropout=0.10,
                input_dropout=0.03,
                aug_noise_std=0.06,
                cosine_head=True,
            ).to(device)
            best_state = {k: v.detach().clone() for k, v in best_model.state_dict().items()}

        best_model.load_state_dict(best_state)

        if X_val is not None and y_val is not None and X_val.numel() > 0:
            X_all = torch.cat([X_train, X_val], dim=0)
            y_all = torch.cat([y_train, y_val], dim=0)
            ft_epochs = 18
            ft_bs = int(min(256, X_all.shape[0]))
            criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
            optimizer = torch.optim.AdamW(best_model.parameters(), lr=9e-4, weight_decay=1.0e-2, betas=(0.9, 0.95))
            steps_per_epoch = (X_all.shape[0] + ft_bs - 1) // ft_bs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, ft_epochs * steps_per_epoch))
            best_model.train()
            for _ in range(ft_epochs):
                perm = torch.randperm(X_all.shape[0], device=device)
                for i in range(0, X_all.shape[0], ft_bs):
                    idx = perm[i:i + ft_bs]
                    xb = X_all.index_select(0, idx)
                    yb = y_all.index_select(0, idx)
                    optimizer.zero_grad(set_to_none=True)
                    logits = best_model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

        if _count_trainable_params(best_model) > param_limit:
            for p in best_model.parameters():
                p.requires_grad_(False)

        best_model.eval()
        return best_model