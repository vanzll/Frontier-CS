import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, n_res_layers: int = 4, dropout: float = 0.05):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.n_res_layers = int(n_res_layers)

        self.register_buffer("x_mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(self.input_dim, dtype=torch.float32))

        self.in_norm = nn.LayerNorm(self.input_dim)
        self.proj = nn.Linear(self.input_dim, self.width)
        self.proj_norm = nn.LayerNorm(self.width)

        self.layers = nn.ModuleList([nn.Linear(self.width, self.width) for _ in range(self.n_res_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(self.n_res_layers)])

        self.final_norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, self.num_classes)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().to(dtype=torch.float32).view(-1)
        std = std.detach().to(dtype=torch.float32).view(-1)
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            return
        self.x_mean.copy_(mean)
        self.x_std.copy_(std.clamp_min(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        x = x.to(dtype=torch.float32)
        x = (x - self.x_mean) / self.x_std
        x = self.in_norm(x)

        x = self.proj(x)
        x = self.proj_norm(x)
        x = self.act(x)
        x = self.drop(x)

        for lin, nrm in zip(self.layers, self.norms):
            y = lin(x)
            y = nrm(y)
            y = self.act(y)
            y = self.drop(y)
            x = x + y

        x = self.final_norm(x)
        return self.head(x)


def _collect_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected batches of (inputs, targets)")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        xs.append(x.detach().cpu().to(dtype=torch.float32))
        ys.append(y.detach().cpu().to(dtype=torch.long))
    x = torch.cat(xs, dim=0) if xs else torch.empty((0, 0), dtype=torch.float32)
    y = torch.cat(ys, dim=0) if ys else torch.empty((0,), dtype=torch.long)
    return x, y


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int = 1024) -> float:
    model.eval()
    correct = 0
    total = int(y.numel())
    if total == 0:
        return 0.0
    with torch.inference_mode():
        for i in range(0, total, batch_size):
            xb = x[i : i + batch_size].to(device, dtype=torch.float32)
            yb = y[i : i + batch_size].to(device, dtype=torch.long)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
    return correct / total


def _param_count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _pick_width(input_dim: int, num_classes: int, param_limit: int, n_res_layers: int, round_to: int = 8) -> int:
    input_dim = int(input_dim)
    num_classes = int(num_classes)
    L = int(n_res_layers)

    # Architecture params:
    # - LayerNorm(input_dim): 2*input_dim
    # - Linear(input_dim->w): input_dim*w + w
    # - (L * (Linear(w->w): w*w + w))
    # - (LayerNorm(w) after proj + after each res + final): (L + 2) * 2*w
    # - head Linear(w->C): w*C + C
    # Total = L*w^2 + (input_dim + num_classes + 3*L + 5)*w + (num_classes + 2*input_dim)
    coeff_w = input_dim + num_classes + 3 * L + 5
    const = num_classes + 2 * input_dim

    def f(w: int) -> int:
        return L * w * w + coeff_w * w + const

    lo, hi = 64, 8192
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f(mid) <= param_limit:
            lo = mid
        else:
            hi = mid - 1

    w = max(64, lo)
    if round_to and round_to > 1:
        w = (w // round_to) * round_to
        w = max(64, w)
        while w >= 64 and f(w) > param_limit:
            w -= round_to
        w = max(64, w)
    else:
        while w >= 64 and f(w) > param_limit:
            w -= 1
        w = max(64, w)

    return int(w)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 5_000_000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        except Exception:
            pass

        train_x, train_y = _collect_from_loader(train_loader)
        val_x, val_y = _collect_from_loader(val_loader)

        all_x = train_x
        all_y = train_y
        if val_x.numel() > 0:
            all_x = torch.cat([train_x, val_x], dim=0)
            all_y = torch.cat([train_y, val_y], dim=0)

        mean = all_x.mean(dim=0) if all_x.numel() > 0 else torch.zeros(input_dim, dtype=torch.float32)
        var = (all_x * all_x).mean(dim=0) - mean * mean if all_x.numel() > 0 else torch.ones(input_dim, dtype=torch.float32)
        std = (var.clamp_min(1e-6)).sqrt()

        n_res_layers = 4
        width = _pick_width(input_dim, num_classes, param_limit, n_res_layers, round_to=8)
        model = ResidualMLP(input_dim=input_dim, num_classes=num_classes, width=width, n_res_layers=n_res_layers, dropout=0.05)
        model.set_input_stats(mean, std)
        model.to(device)

        # Safety: ensure under limit (in case formula mismatch)
        while _param_count_trainable(model) > param_limit and model.width > 64:
            width = max(64, model.width - 8)
            model = ResidualMLP(input_dim=input_dim, num_classes=num_classes, width=width, n_res_layers=n_res_layers, dropout=0.05)
            model.set_input_stats(mean, std)
            model.to(device)

        # Training hyperparams
        batch_size = 256
        n_train = int(train_x.shape[0])
        if n_train > 0:
            batch_size = min(batch_size, n_train)
        max_epochs = 160
        min_epochs = 25
        patience = 25

        base_lr = 2.0e-3
        min_lr = 2.0e-4
        weight_decay = 2.0e-2
        label_smoothing = 0.08
        grad_clip = 1.0

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=weight_decay)

        steps_per_epoch = max(1, math.ceil(max(1, n_train) / batch_size))
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = max(10, int(0.06 * total_steps))

        def set_lr(step: int):
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
            else:
                t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * float(t)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # EMA via AveragedModel with exponential averaging
        decay = 0.99

        def ema_avg(averaged_param, param, num_averaged):
            return averaged_param * decay + param * (1.0 - decay)

        try:
            from torch.optim.swa_utils import AveragedModel
            ema_model = AveragedModel(model, avg_fn=ema_avg)
            ema_model.to(device)
            use_ema = True
        except Exception:
            ema_model = None
            use_ema = False

        best_acc = -1.0
        best_state = None
        bad_epochs = 0
        global_step = 0

        if n_train == 0:
            model.eval()
            return model

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                xb = train_x[idx].to(device, dtype=torch.float32)
                yb = train_y[idx].to(device, dtype=torch.long)

                set_lr(global_step)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if use_ema:
                    ema_model.update_parameters(model)
                global_step += 1

            if val_y.numel() > 0:
                eval_model = ema_model if use_ema else model
                val_acc = _accuracy(eval_model, val_x, val_y, device=device, batch_size=1024)
                if val_acc > best_acc + 1e-4:
                    best_acc = val_acc
                    best_state = copy.deepcopy(eval_model.module.state_dict() if use_ema else eval_model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if epoch + 1 >= min_epochs and bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # Light fine-tuning on train+val combined
        n_all = int(all_x.shape[0])
        if n_all > n_train:
            ft_epochs = 12
            ft_bs = min(256, n_all)
            ft_base_lr = 6.0e-4
            ft_min_lr = 1.5e-4
            ft_steps_per_epoch = max(1, math.ceil(n_all / ft_bs))
            ft_total_steps = ft_epochs * ft_steps_per_epoch
            ft_warmup = max(10, int(0.08 * ft_total_steps))

            optimizer = torch.optim.AdamW(model.parameters(), lr=ft_base_lr, betas=(0.9, 0.95), weight_decay=1.5e-2)

            def ft_set_lr(step: int):
                if step < ft_warmup:
                    lr = ft_base_lr * (step + 1) / ft_warmup
                else:
                    t = (step - ft_warmup) / max(1, (ft_total_steps - ft_warmup))
                    lr = ft_min_lr + 0.5 * (ft_base_lr - ft_min_lr) * (1.0 + math.cos(math.pi * float(t)))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            ft_step = 0
            for _ in range(ft_epochs):
                model.train()
                perm = torch.randperm(n_all)
                for start in range(0, n_all, ft_bs):
                    idx = perm[start : start + ft_bs]
                    xb = all_x[idx].to(device, dtype=torch.float32)
                    yb = all_y[idx].to(device, dtype=torch.long)
                    ft_set_lr(ft_step)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    ft_step += 1

        model.eval()
        return model