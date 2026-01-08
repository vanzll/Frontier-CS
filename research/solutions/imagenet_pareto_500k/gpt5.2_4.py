import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualFF(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class _CosineClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int, init_scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = torch.exp(self.log_scale).clamp(1.0, 100.0)
        return scale * (x @ w.t())


class _MLPResNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, mean: torch.Tensor, std: torch.Tensor,
                 dim: int = 384, hidden: int = 288, blocks: int = 2, dropout: float = 0.10):
        super().__init__()
        self.register_buffer("x_mean", mean.reshape(1, -1).float(), persistent=True)
        self.register_buffer("x_std", std.reshape(1, -1).float(), persistent=True)

        if dim != input_dim:
            self.in_proj = nn.Linear(input_dim, dim, bias=True)
        else:
            self.in_proj = nn.Identity()

        self.stem_ln = nn.LayerNorm(dim)
        self.blocks = nn.Sequential(*[_ResidualFF(dim, hidden, dropout) for _ in range(blocks)])
        self.out_ln = nn.LayerNorm(dim)
        self.head = _CosineClassifier(dim, num_classes, init_scale=30.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()
        x = (x - self.x_mean) / (self.x_std + 1e-6)
        x = self.in_proj(x)
        x = self.stem_ln(x)
        x = self.blocks(x)
        x = self.out_ln(x)
        x = self.head(x)
        return x


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {}
        self.device = None
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
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict_to(self, model: nn.Module, state: dict):
        for name, p in model.named_parameters():
            if p.requires_grad and name in state:
                p.copy_(state[name])


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _collect_loader(loader):
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected batches as (inputs, targets).")
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    if x.dim() > 2:
        x = x.view(x.shape[0], -1)
    return x.float(), y.long()


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> float:
    model.eval()
    n = x.shape[0]
    correct = 0
    total = 0
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return float(correct) / float(max(1, total))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _cosine_warmup(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float):
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, t))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def _make_param_groups(model: nn.Module, weight_decay: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if n.endswith(".bias") or "ln" in n or "layernorm" in n or "norm" in n or "log_scale" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500000))
        device = metadata.get("device", "cpu")
        if device != "cpu":
            device = "cpu"

        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        x_train, y_train = _collect_loader(train_loader)
        x_val, y_val = _collect_loader(val_loader)

        if x_train.shape[1] != input_dim:
            input_dim = int(x_train.shape[1])

        mean = x_train.mean(dim=0)
        std = x_train.std(dim=0).clamp_min(1e-3)

        model = _MLPResNet(
            input_dim=input_dim,
            num_classes=num_classes,
            mean=mean,
            std=std,
            dim=384 if input_dim == 384 else input_dim,
            hidden=288,
            blocks=2,
            dropout=0.10,
        ).to(device)

        if _count_trainable_params(model) > param_limit:
            model = _MLPResNet(
                input_dim=input_dim,
                num_classes=num_classes,
                mean=mean,
                std=std,
                dim=384 if input_dim == 384 else input_dim,
                hidden=256,
                blocks=2,
                dropout=0.10,
            ).to(device)

        if _count_trainable_params(model) > param_limit:
            model = _MLPResNet(
                input_dim=input_dim,
                num_classes=num_classes,
                mean=mean,
                std=std,
                dim=384 if input_dim == 384 else input_dim,
                hidden=256,
                blocks=1,
                dropout=0.10,
            ).to(device)

        if _count_trainable_params(model) > param_limit:
            dim = 256 if input_dim >= 256 else input_dim
            model = _MLPResNet(
                input_dim=input_dim,
                num_classes=num_classes,
                mean=mean,
                std=std,
                dim=dim,
                hidden=192,
                blocks=1,
                dropout=0.10,
            ).to(device)

        if _count_trainable_params(model) > param_limit:
            raise RuntimeError("Could not fit model within parameter limit.")

        batch_size = 256
        max_epochs = 180
        label_smoothing = 0.05
        mixup_alpha = 0.20
        grad_clip = 1.0

        base_lr = 3.5e-3
        min_lr = 2.0e-4
        weight_decay = 2.0e-2

        groups = _make_param_groups(model, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.99))

        steps_per_epoch = int(math.ceil(x_train.shape[0] / batch_size))
        total_steps = max(1, steps_per_epoch * max_epochs)
        warmup_steps = int(0.08 * total_steps)

        ema = _EMA(model, decay=0.995)
        best_state = None
        best_val = -1.0

        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        model.train()
        global_step = 0
        n_train = x_train.shape[0]
        for epoch in range(max_epochs):
            idx = torch.randperm(n_train)
            for bi in range(0, n_train, batch_size):
                batch_idx = idx[bi:bi + batch_size]
                xb = x_train[batch_idx].to(device, non_blocking=False)
                yb = y_train[batch_idx].to(device, non_blocking=False)

                lr = _cosine_warmup(global_step, total_steps, base_lr, warmup_steps, min_lr)
                _set_lr(optimizer, lr)

                do_mixup = (mixup_alpha > 0.0) and (xb.shape[0] >= 2)
                if do_mixup:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(xb.shape[0], device=xb.device)
                    xb2 = xb[perm]
                    yb2 = yb[perm]
                    xb = xb.mul(lam).add_(xb2, alpha=(1.0 - lam))
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                if do_mixup:
                    loss = lam * F.cross_entropy(logits, yb, label_smoothing=label_smoothing) + (1.0 - lam) * F.cross_entropy(
                        logits, yb2, label_smoothing=label_smoothing
                    )
                else:
                    loss = loss_fn(logits, yb)

                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                ema.update(model)
                global_step += 1

            if (epoch + 1) % 1 == 0:
                cur_state = ema.state_dict()
                ema.load_state_dict_to(model, cur_state)
                val_acc = _accuracy(model, x_val, y_val, batch_size=512)
                if val_acc > best_val + 1e-6:
                    best_val = val_acc
                    best_state = deepcopy(cur_state)

        if best_state is not None:
            ema.load_state_dict_to(model, best_state)

        x_all = torch.cat([x_train, x_val], dim=0)
        y_all = torch.cat([y_train, y_val], dim=0)

        ft_epochs = 30
        ft_batch_size = 256
        ft_base_lr = 1.2e-3
        ft_min_lr = 1.0e-4
        ft_wd = 1.5e-2
        ft_mixup_alpha = 0.10
        ft_label_smoothing = 0.03

        groups = _make_param_groups(model, weight_decay=ft_wd)
        optimizer = torch.optim.AdamW(groups, lr=ft_base_lr, betas=(0.9, 0.99))

        steps_per_epoch = int(math.ceil(x_all.shape[0] / ft_batch_size))
        total_steps = max(1, steps_per_epoch * ft_epochs)
        warmup_steps = int(0.10 * total_steps)

        ema2 = _EMA(model, decay=0.997)

        model.train()
        global_step = 0
        n_all = x_all.shape[0]
        for epoch in range(ft_epochs):
            idx = torch.randperm(n_all)
            for bi in range(0, n_all, ft_batch_size):
                batch_idx = idx[bi:bi + ft_batch_size]
                xb = x_all[batch_idx].to(device, non_blocking=False)
                yb = y_all[batch_idx].to(device, non_blocking=False)

                lr = _cosine_warmup(global_step, total_steps, ft_base_lr, warmup_steps, ft_min_lr)
                _set_lr(optimizer, lr)

                do_mixup = (ft_mixup_alpha > 0.0) and (xb.shape[0] >= 2)
                if do_mixup:
                    lam = float(np.random.beta(ft_mixup_alpha, ft_mixup_alpha))
                    perm = torch.randperm(xb.shape[0], device=xb.device)
                    xb2 = xb[perm]
                    yb2 = yb[perm]
                    xb = xb.mul(lam).add_(xb2, alpha=(1.0 - lam))

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                if do_mixup:
                    loss = lam * F.cross_entropy(logits, yb, label_smoothing=ft_label_smoothing) + (1.0 - lam) * F.cross_entropy(
                        logits, yb2, label_smoothing=ft_label_smoothing
                    )
                else:
                    loss = F.cross_entropy(logits, yb, label_smoothing=ft_label_smoothing)

                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                ema2.update(model)
                global_step += 1

        ema2.load_state_dict_to(model, ema2.state_dict())
        model.eval()
        return model