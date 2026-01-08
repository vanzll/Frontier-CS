import os
import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int,
        blocks: int = 3,
        dropout: float = 0.1,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        res_scale: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.blocks = int(blocks)
        self.res_scale = float(res_scale)

        if mean is None:
            mean = torch.zeros(self.input_dim, dtype=torch.float32)
        if std is None:
            std = torch.ones(self.input_dim, dtype=torch.float32)

        mean = mean.to(dtype=torch.float32).view(1, -1).contiguous()
        inv_std = (1.0 / std.to(dtype=torch.float32).clamp_min(1e-6)).view(1, -1).contiguous()
        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("inv_std", inv_std, persistent=True)

        self.proj = nn.Linear(self.input_dim, self.width, bias=True)
        self.ln0 = nn.LayerNorm(self.width)

        self.linears = nn.ModuleList([nn.Linear(self.width, self.width, bias=True) for _ in range(self.blocks)])
        self.lns = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(self.blocks)])
        self.drop = nn.Dropout(p=float(dropout))

        self.ln_final = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)
        x = (x - self.mean) * self.inv_std

        x = self.proj(x)
        x = F.gelu(x)
        x = self.ln0(x)

        for lin, ln in zip(self.linears, self.lns):
            y = lin(x)
            y = F.gelu(y)
            y = ln(y)
            y = self.drop(y)
            x = x + y * self.res_scale

        x = self.ln_final(x)
        x = self.head(x)
        return x


def _collect_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Unsupported batch format.")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    X = torch.cat(xs, dim=0).contiguous()
    y = torch.cat(ys, dim=0).contiguous()
    if X.dtype != torch.float32:
        X = X.to(dtype=torch.float32)
    if y.dtype != torch.long:
        y = y.to(dtype=torch.long)
    return X, y


def _estimate_params(input_dim: int, num_classes: int, width: int, blocks: int) -> int:
    # params = blocks*w^2 + (in + out + 3*blocks + 5)*w + out
    in_dim = int(input_dim)
    out_dim = int(num_classes)
    w = int(width)
    b = int(blocks)
    return b * w * w + (in_dim + out_dim + 3 * b + 5) * w + out_dim


def _choose_width(input_dim: int, num_classes: int, param_limit: int, blocks: int = 3) -> int:
    in_dim = int(input_dim)
    out_dim = int(num_classes)
    limit = int(param_limit)
    b = int(blocks)

    if b <= 0:
        # single linear in->w->out not supported here; just return something safe
        return 256

    # Solve b*w^2 + (in+out+3b+5)*w + out - limit <= 0 for w
    lin = in_dim + out_dim + 3 * b + 5
    disc = lin * lin + 4 * b * (limit - out_dim)
    if disc <= 0:
        w0 = 64
    else:
        w0 = int(((-lin + int(math.isqrt(disc))) // (2 * b)))
        w0 = max(w0, 64)

    def round_down(x: int, m: int = 8) -> int:
        return (x // m) * m

    w = round_down(w0, 8)
    if w < 64:
        w = 64

    # Adjust down if needed
    while w >= 64 and _estimate_params(in_dim, out_dim, w, b) > limit:
        w -= 8

    if w < 64:
        w = 64
        while _estimate_params(in_dim, out_dim, w, b) > limit and w > 8:
            w -= 8
        w = max(w, 8)
    return w


@torch.inference_mode()
def _accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    n = int(X.shape[0])
    correct = 0
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
    return correct / max(1, n)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 1_000_000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass

        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)

        X_train, y_train = _collect_from_loader(train_loader)
        X_val, y_val = _collect_from_loader(val_loader)

        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)

        blocks = 3
        width = _choose_width(input_dim, num_classes, param_limit, blocks=blocks)

        model = _ResidualMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            blocks=blocks,
            dropout=0.1,
            mean=mean,
            std=std,
            res_scale=0.5,
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # Emergency width reduction
            while width > 32:
                width -= 8
                model = _ResidualMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    width=width,
                    blocks=blocks,
                    dropout=0.1,
                    mean=mean,
                    std=std,
                    res_scale=0.5,
                ).to(device)
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if param_count <= param_limit:
                    break

        X_train = X_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device, dtype=torch.long)
        X_val = X_val.to(device=device, dtype=torch.float32)
        y_val = y_val.to(device=device, dtype=torch.long)

        epochs = 200
        batch_size = 256
        steps_per_epoch = (X_train.shape[0] + batch_size - 1) // batch_size
        total_steps = int(epochs * steps_per_epoch)
        warmup_steps = max(10, int(0.05 * total_steps))

        lr_max = 3e-3
        weight_decay = 2e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay, betas=(0.9, 0.99))

        label_smoothing = 0.05
        mixup_alpha = 0.2
        mixup_prob = 0.6
        grad_clip = 1.0

        params = [p for p in model.parameters() if p.requires_grad]
        ema_decay = 0.995
        ema = [p.detach().clone() for p in params]
        backup = [torch.empty_like(t) for t in ema]

        best_val = -1.0
        best_ema = None
        best_epoch = -1
        patience = 35
        min_epochs = 40
        bad = 0

        step_idx = 0
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(X_train.shape[0], device=device)

            for j in range(0, X_train.shape[0], batch_size):
                idx = perm[j : j + batch_size]
                xb = X_train.index_select(0, idx)
                yb = y_train.index_select(0, idx)

                do_mix = (mixup_alpha > 0.0) and (torch.rand((), device=device).item() < mixup_prob) and (xb.shape[0] > 1)
                if do_mix:
                    lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample(()).to(device=device))
                    perm2 = torch.randperm(xb.shape[0], device=device)
                    xb2 = xb.index_select(0, perm2)
                    yb2 = yb.index_select(0, perm2)
                    xb = xb.mul(lam).add(xb2, alpha=(1.0 - lam))

                if step_idx < warmup_steps:
                    lr = lr_max * float(step_idx + 1) / float(warmup_steps)
                else:
                    t = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr = lr_max * 0.5 * (1.0 + math.cos(math.pi * t))
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)

                if do_mix:
                    loss1 = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
                    loss2 = F.cross_entropy(logits, yb2, label_smoothing=label_smoothing)
                    loss = lam * loss1 + (1.0 - lam) * loss2
                else:
                    loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                with torch.no_grad():
                    for p, e in zip(params, ema):
                        e.mul_(ema_decay).add_(p.data, alpha=(1.0 - ema_decay))

                step_idx += 1

            # Evaluate EMA model on val
            with torch.no_grad():
                for p, b in zip(params, backup):
                    b.copy_(p.data)
                for p, e in zip(params, ema):
                    p.data.copy_(e)

            val_acc = _accuracy(model, X_val, y_val, batch_size=1024)

            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.data.copy_(b)

            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_epoch = epoch
                best_ema = [t.detach().clone() for t in ema]
                bad = 0
            else:
                bad += 1

            if epoch + 1 >= min_epochs and bad >= patience:
                break

        if best_ema is not None:
            with torch.no_grad():
                for p, e in zip(params, best_ema):
                    p.data.copy_(e)

        # Optional light fine-tune on train+val combined (no val usage)
        X_all = torch.cat([X_train, X_val], dim=0)
        y_all = torch.cat([y_train, y_val], dim=0)
        finetune_epochs = 25
        ft_batch = 256
        ft_steps_per_epoch = (X_all.shape[0] + ft_batch - 1) // ft_batch
        ft_total_steps = finetune_epochs * ft_steps_per_epoch
        ft_lr_max = 7e-4
        ft_warmup = max(5, int(0.1 * ft_total_steps))
        ft_step = 0

        optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr_max, weight_decay=1e-4, betas=(0.9, 0.99))
        model.train()
        for _ in range(finetune_epochs):
            perm = torch.randperm(X_all.shape[0], device=device)
            for j in range(0, X_all.shape[0], ft_batch):
                idx = perm[j : j + ft_batch]
                xb = X_all.index_select(0, idx)
                yb = y_all.index_select(0, idx)

                if ft_step < ft_warmup:
                    lr = ft_lr_max * float(ft_step + 1) / float(ft_warmup)
                else:
                    t = float(ft_step - ft_warmup) / float(max(1, ft_total_steps - ft_warmup))
                    lr = ft_lr_max * 0.5 * (1.0 + math.cos(math.pi * t))
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.02)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ft_step += 1

        model = model.to(torch.device("cpu"))
        model.eval()

        # Safety: ensure parameter constraint
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # Return a safe tiny linear model as fallback under limit
            fallback = nn.Linear(input_dim, num_classes, bias=True)
            fallback.eval()
            return fallback

        return model