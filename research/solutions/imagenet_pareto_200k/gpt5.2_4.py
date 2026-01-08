import math
import time
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn


def _extract_xy(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        x = batch.get("inputs", batch.get("x", batch.get("data", None)))
        y = batch.get("targets", batch.get("y", batch.get("label", batch.get("labels", None))))
        if x is None or y is None:
            raise ValueError("Unsupported batch dict keys")
        return x, y
    raise ValueError("Unsupported batch type")


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def _accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = _extract_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return float(correct) / float(total) if total > 0 else 0.0


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: v.detach().clone() for k, v in sd.items()}


def _compute_mean_std(train_loader, input_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    s1 = torch.zeros(input_dim, dtype=torch.float64)
    s2 = torch.zeros(input_dim, dtype=torch.float64)
    n = 0
    for batch in train_loader:
        x, _ = _extract_xy(batch)
        x = x.detach()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float64, device=torch.device("cpu"))
        if x.size(1) != input_dim:
            x = x[:, :input_dim]
        s1 += x.sum(dim=0)
        s2 += (x * x).sum(dim=0)
        n += x.size(0)
    if n == 0:
        mean = torch.zeros(input_dim, dtype=torch.float32)
        std = torch.ones(input_dim, dtype=torch.float32)
        return mean.to(device), std.to(device)
    mean = (s1 / n)
    var = (s2 / n) - mean * mean
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)
    mean = mean.to(dtype=torch.float32, device=device)
    std = std.to(dtype=torch.float32, device=device)
    return mean, std


def _max_width_for_budget(input_dim: int, num_classes: int, depth: int, param_limit: int) -> int:
    # Architecture:
    # buffers: mean/std (not params)
    # in_ln: LayerNorm(input_dim) => 2*input_dim params
    # depth hidden layers each with LayerNorm(width) => depth*(2*width)
    # linears: input_dim->w, (depth-1) times w->w, w->num_classes (all with bias)
    # rezero alphas: (depth-1) params
    best_w = 8
    for w in range(1024, 7, -1):
        linears = (input_dim * w + w) + (depth - 1) * (w * w + w) + (w * num_classes + num_classes)
        norms = 2 * input_dim + depth * (2 * w)
        alphas = max(0, depth - 1)
        total = linears + norms + alphas
        if total <= param_limit:
            best_w = w
            break
    return best_w


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        one_minus = 1.0 - d
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(n, None)
            if s is None:
                self.shadow[n] = p.detach().clone()
            else:
                s.mul_(d).add_(p.detach(), alpha=one_minus)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])


class _ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int,
        depth: int,
        dropout: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.depth = int(depth)

        self.register_buffer("mean", mean.detach().clone())
        invstd = 1.0 / torch.clamp(std.detach().clone(), min=1e-6)
        self.register_buffer("invstd", invstd)

        self.in_ln = nn.LayerNorm(self.input_dim)

        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.fcs.append(nn.Linear(self.input_dim, self.width, bias=True))
        self.lns.append(nn.LayerNorm(self.width))
        for _ in range(self.depth - 1):
            self.fcs.append(nn.Linear(self.width, self.width, bias=True))
            self.lns.append(nn.LayerNorm(self.width))

        self.out = nn.Linear(self.width, self.num_classes, bias=True)

        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(p=float(dropout))

        if self.depth > 1:
            self.alphas = nn.Parameter(torch.full((self.depth - 1,), 0.1, dtype=torch.float32))
        else:
            self.alphas = nn.Parameter(torch.zeros(0, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float32)
        x = (x - self.mean) * self.invstd
        x = self.in_ln(x)

        h = self.fcs[0](x)
        h = self.lns[0](h)
        h = self.act(h)
        h = self.drop(h)

        for i in range(1, self.depth):
            z = self.fcs[i](h)
            z = self.lns[i](z)
            z = self.act(z)
            z = self.drop(z)
            h = h + self.alphas[i - 1] * z

        return self.out(h)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 200000))

        torch.manual_seed(0)
        try:
            torch.set_num_threads(min(8, int(torch.get_num_threads())))
        except Exception:
            pass

        mean, std = _compute_mean_std(train_loader, input_dim=input_dim, device=device)

        depth = 4
        width = _max_width_for_budget(input_dim, num_classes, depth, param_limit)
        width = max(32, int(width))
        dropout = 0.10

        model = _ResidualMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            depth=depth,
            dropout=dropout,
            mean=mean,
            std=std,
        ).to(device)

        if _count_trainable_params(model) > param_limit:
            # Fallback: reduce width until it fits.
            w = width
            while w > 16:
                w -= 1
                model = _ResidualMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    width=w,
                    depth=depth,
                    dropout=dropout,
                    mean=mean,
                    std=std,
                ).to(device)
                if _count_trainable_params(model) <= param_limit:
                    break

        max_epochs = 260
        warmup_epochs = 8
        patience = 40
        min_epochs = 25

        try:
            first_batch = next(iter(train_loader))
            bx, _ = _extract_xy(first_batch)
            bs = int(bx.size(0))
        except Exception:
            bs = 64

        base_lr = 2.2e-3 * math.sqrt(max(1.0, bs / 64.0))
        min_lr = base_lr * 0.05

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.03, betas=(0.9, 0.98))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

        ema = _EMA(model, decay=0.995)

        best_acc = -1.0
        best_state = None
        bad_epochs = 0

        t_start = time.time()
        time_budget = 3450.0

        has_val = val_loader is not None

        for epoch in range(max_epochs):
            if (time.time() - t_start) > time_budget:
                break

            if epoch < warmup_epochs:
                lr = base_lr * float(epoch + 1) / float(warmup_epochs)
            else:
                prog = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            model.train()
            for batch in train_loader:
                x, y = _extract_xy(batch)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(model)

            if has_val:
                # Evaluate with EMA weights
                backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                ema.copy_to(model)
                val_acc = _accuracy(model, val_loader, device=device)
                for n, p in model.named_parameters():
                    if p.requires_grad and n in backup:
                        p.data.copy_(backup[n])

                improved = val_acc > best_acc + 1e-5
                if improved:
                    backup2 = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                    ema.copy_to(model)
                    best_state = _clone_state_dict(model)
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in backup2:
                            p.data.copy_(backup2[n])

                    best_acc = val_acc
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if epoch + 1 >= min_epochs and bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        else:
            ema.copy_to(model)

        model.eval()
        return model