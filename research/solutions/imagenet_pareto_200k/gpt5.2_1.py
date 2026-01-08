import math
import copy
import os
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _InputStandardizer(nn.Module):
    def __init__(self, input_dim: int, mean: torch.Tensor, inv_std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, input_dim))
        self.register_buffer("inv_std", inv_std.reshape(1, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class _MLPRes(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.std = _InputStandardizer(input_dim, mean=mean, inv_std=inv_std)
        self.norm0 = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, num_classes, bias=True)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.std(x)
        x = self.norm0(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm1(x)

        res = x
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + res

        x = self.fc3(x)
        return x


def _collect_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
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
    x_all = torch.cat(xs, dim=0).contiguous()
    y_all = torch.cat(ys, dim=0).contiguous()
    if x_all.dtype != torch.float32:
        x_all = x_all.float()
    if y_all.dtype != torch.long:
        y_all = y_all.long()
    return x_all, y_all


def _compute_mean_std(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    inv_std = torch.rsqrt(var + eps)
    return mean, inv_std


def _accuracy_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _param_count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        device = metadata.get("device", "cpu")
        if device != "cpu":
            device = "cpu"
        device = torch.device(device)

        train_x, train_y = _collect_from_loader(train_loader)
        val_x, val_y = _collect_from_loader(val_loader)

        input_dim = int(metadata.get("input_dim", train_x.shape[1]))
        num_classes = int(metadata.get("num_classes", int(train_y.max().item()) + 1))
        param_limit = int(metadata.get("param_limit", 200_000))

        if train_x.shape[1] != input_dim:
            input_dim = int(train_x.shape[1])

        mean, inv_std = _compute_mean_std(train_x)

        # Choose max hidden_dim H for 2-hidden-layer MLP: D*H + H + H*H + H + H*C + C <= L
        D, C, L = input_dim, num_classes, param_limit
        best_h = 64
        for h in range(512, 31, -1):
            params = (D * h + h) + (h * h + h) + (h * C + C)
            if params <= L:
                best_h = h
                break

        model = _MLPRes(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=best_h,
            mean=mean,
            inv_std=inv_std,
            dropout=0.10,
        ).to(device)

        if _param_count_trainable(model) > param_limit:
            # Fallback to a safe configuration
            fallback_h = 256
            while fallback_h > 32:
                model = _MLPRes(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dim=fallback_h,
                    mean=mean,
                    inv_std=inv_std,
                    dropout=0.10,
                ).to(device)
                if _param_count_trainable(model) <= param_limit:
                    break
                fallback_h -= 8

        batch_size = getattr(train_loader, "batch_size", None)
        if batch_size is None or batch_size <= 0:
            batch_size = 128
        batch_size = int(batch_size)

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        lr = 3e-3
        weight_decay = 8e-5
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training configuration
        max_epochs = 180
        patience = 20
        mixup_alpha = 0.20
        ema_decay = 0.995
        clip_norm = 1.0

        # EMA storage (CPU tensors for safety)
        ema = [p.detach().clone() for p in model.parameters() if p.requires_grad]

        def _ema_update():
            i = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                ep = ema[i]
                ep.mul_(ema_decay).add_(p.detach(), alpha=(1.0 - ema_decay))
                i += 1

        def _eval_acc(use_model: nn.Module) -> float:
            use_model.eval()
            with torch.inference_mode():
                logits = use_model(val_x)
                return _accuracy_logits(logits, val_y)

        best_state = None
        best_acc = -1.0
        bad_epochs = 0

        total_steps = max(1, (train_x.shape[0] + batch_size - 1) // batch_size) * max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.12,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=50.0,
        )

        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)

        n = train_x.shape[0]
        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n, generator=gen, device=device)

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = train_x[idx]
                yb = train_y[idx]

                if mixup_alpha > 0.0 and xb.shape[0] >= 2:
                    lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
                    lam = max(0.0, min(1.0, lam))
                    perm2 = torch.randperm(xb.shape[0], generator=gen, device=device)
                    xb2 = xb[perm2]
                    yb2 = yb[perm2]
                    xmix = xb.mul(lam).add(xb2, alpha=(1.0 - lam))

                    logits = model(xmix)
                    loss = lam * F.cross_entropy(logits, yb) + (1.0 - lam) * F.cross_entropy(logits, yb2)
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_norm is not None and clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()
                _ema_update()
                scheduler.step()

            # Validate with EMA weights applied temporarily
            model_eval = model
            # Swap to EMA for eval
            backup = []
            i = 0
            for p in model_eval.parameters():
                if not p.requires_grad:
                    continue
                backup.append(p.detach().clone())
                p.data.copy_(ema[i].to(p.device))
                i += 1

            val_acc = _eval_acc(model_eval)

            # Restore
            i = 0
            for p in model_eval.parameters():
                if not p.requires_grad:
                    continue
                p.data.copy_(backup[i])
                i += 1

            if val_acc > best_acc + 1e-6:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Finalize: load EMA weights for returned model
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(ema[i].to(p.device))
            i += 1

        model.eval()
        return model.to(device)