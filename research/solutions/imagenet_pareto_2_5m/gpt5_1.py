import math
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class InputNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, loader, device="cpu"):
        total = 0
        sum_ = None
        sum_sq = None
        for x, _ in loader:
            x = x.to(device, non_blocking=True).float()
            if sum_ is None:
                sum_ = x.sum(dim=0, dtype=torch.float64)
                sum_sq = (x * x).sum(dim=0, dtype=torch.float64)
            else:
                sum_ += x.sum(dim=0, dtype=torch.float64)
                sum_sq += (x * x).sum(dim=0, dtype=torch.float64)
            total += x.size(0)
        if total > 0:
            mean = (sum_ / total).to(self.mean.dtype)
            var = (sum_sq / total - (mean.double() ** 2)).clamp(min=0).to(self.mean.dtype)
            std = torch.sqrt(var + self.eps)
            std[std < self.eps] = 1.0
            self.mean.copy_(mean)
            self.std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp_min(self.eps)


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, h1: int, h2: int, use_bn: bool = True, drop: float = 0.2):
        super().__init__()
        self.norm = InputNormalizer(input_dim)
        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2) if use_bn else nn.Identity()
        self.fc_out = nn.Linear(h2, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        # Initialize
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="gelu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="gelu")
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_out.bias)
        nn.init.normal_(self.fc_out.weight, std=0.02)

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
        x = self.fc_out(x)
        return x


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    if target.dtype == torch.long:
        n_classes = logits.size(-1)
        nll = F.nll_loss(log_probs, target, reduction='none')
        if smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * nll + smoothing * smooth_loss
        else:
            loss = nll
        return loss.mean()
    else:
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = -(target * log_probs).sum(dim=-1)
        return loss.mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().float()
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.copy_(self.backup[name])
        self.backup = {}


def choose_hidden_dims(input_dim: int, num_classes: int, param_limit: int, use_bn: bool = True) -> Tuple[int, int]:
    best = None
    best_params = -1

    # search over w1, compute optimal w2 analytically then fine-tune
    max_w1 = 2048
    min_w1 = 512
    step = 16

    for w1 in range(max_w1, min_w1 - 1, -step):
        denom = w1 + num_classes + (3 if use_bn else 1)
        const = input_dim * w1 + w1 + num_classes + (2 * w1 if use_bn else 0)
        w2_est = (param_limit - const) // denom
        if w2_est <= 64:
            continue
        # round to step
        w2 = max(64, int(w2_est // step) * step)

        # adjust down until within budget
        for shrink in range(0, 64):
            candidate_w2 = w2 - shrink * step
            if candidate_w2 < 64:
                break
            params = (input_dim * w1 + w1 * candidate_w2 + candidate_w2 * num_classes +
                      w1 + candidate_w2 + num_classes +
                      (2 * (w1 + candidate_w2) if use_bn else 0))
            if params <= param_limit:
                if params > best_params:
                    best = (w1, candidate_w2)
                    best_params = params
                break

    # Fallback if nothing found (shouldn't happen)
    if best is None:
        # try symmetric setting
        d = int((math.sqrt((input_dim + num_classes + (4 if use_bn else 2)) ** 2 +
                           4 * (param_limit - num_classes)) - (input_dim + num_classes + (4 if use_bn else 2))) / 2)
        d = max(256, (d // step) * step)
        w1 = w2 = d
        # ensure under limit
        while True:
            params = (input_dim * w1 + w1 * w2 + w2 * num_classes +
                      w1 + w2 + num_classes +
                      (2 * (w1 + w2) if use_bn else 0))
            if params <= param_limit or w1 <= 128:
                break
            w1 -= step
            w2 -= step
        best = (w1, w2)
    return best


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and ("bn" not in name.lower()):
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='mean')
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)
    acc = (correct / total) if total > 0 else 0.0
    avg_loss = (loss_sum / total) if total > 0 else 0.0
    return acc, avg_loss


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        set_seed(42)

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2500000))
        device = str(metadata.get("device", "cpu"))

        use_bn = True

        h1, h2 = choose_hidden_dims(input_dim, num_classes, param_limit, use_bn=use_bn)

        model = MLPNet(input_dim, num_classes, h1=h1, h2=h2, use_bn=use_bn, drop=0.25)
        model.to(device)

        # Ensure parameter constraint
        assert count_trainable_params(model) <= param_limit

        # Fit input normalization on training data
        model.norm.fit(train_loader, device=device)

        # Training setup
        epochs = 180
        lr_max = 3e-3
        weight_decay = 0.02
        smoothing = 0.08
        grad_clip = 1.0

        optimizer = create_optimizer(model, lr=lr_max, weight_decay=weight_decay)

        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr_max,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0,
        )

        ema = ModelEMA(model, decay=0.995)

        best_acc = -1.0
        best_ema_shadow = None
        best_state_dict = None
        patience = 30
        no_improve_epochs = 0

        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).long()

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = soft_cross_entropy(logits, y, smoothing=smoothing)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                ema.update(model)

            # Validation with EMA weights
            ema.apply_to(model)
            if val_loader is not None:
                val_acc, _ = evaluate(model, val_loader, device)
            else:
                val_acc = 0.0
            ema.restore(model)

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                # Save EMA shadow as best
                best_ema_shadow = {k: v.clone() for k, v in ema.shadow.items()}
                best_state_dict = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                break

        # Load best weights (EMA)
        if best_ema_shadow is not None:
            for name, p in model.named_parameters():
                if p.requires_grad and name in best_ema_shadow:
                    p.data.copy_(best_ema_shadow[name].to(p.device))
        elif best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        model.eval()
        return model