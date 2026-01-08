import os
import math
import copy
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loader_to_tensors(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.tensor(yb)
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)
        xs.append(xb)
        ys.append(yb)
    x = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
    y = torch.cat(ys, dim=0) if len(ys) > 1 else ys[0]
    return x, y


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 2048) -> float:
    model.eval()
    n = x.shape[0]
    correct = 0
    for i in range(0, n, bs):
        xb = x[i:i + bs]
        yb = y[i:i + bs]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / max(1, n)


class _FeatureNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32), persistent=True)
        self.register_buffer("std", torch.ones(dim, dtype=torch.float32), persistent=True)

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().to(dtype=torch.float32)
        std = std.detach().to(dtype=torch.float32)
        std = torch.clamp(std, min=self.eps)
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _ResidualBottleneck(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float = 0.10, res_scale: float = 1.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        self.res_scale = float(res_scale)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y * self.res_scale


class _Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        w1: int,
        w2: int,
        bottleneck: int,
        blocks: int,
        dropout: float = 0.10
    ):
        super().__init__()
        self.norm = _FeatureNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, w1)
        self.ln1 = nn.LayerNorm(w1)
        self.fc2 = nn.Linear(w1, w2)
        self.ln2 = nn.LayerNorm(w2)
        self.blocks = nn.ModuleList([_ResidualBottleneck(w2, bottleneck, dropout=dropout, res_scale=1.0) for _ in range(blocks)])
        self.ln_out = nn.LayerNorm(w2)
        self.head = nn.Linear(w2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)
        x = self.head(x)
        return x


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def store(self) -> List[torch.Tensor]:
        return [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def copy_to_model(self):
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s.data)

    @torch.no_grad()
    def restore(self, backup: List[torch.Tensor]):
        for p, b in zip(self.params, backup):
            p.data.copy_(b.data)

    @torch.no_grad()
    def shadow_clone(self) -> List[torch.Tensor]:
        return [t.detach().clone() for t in self.shadow]

    @torch.no_grad()
    def load_shadow(self, shadow: List[torch.Tensor]):
        for sdst, ssrc in zip(self.shadow, shadow):
            sdst.data.copy_(ssrc.data)


def _make_param_groups(model: nn.Module, weight_decay: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or ".ln" in name or "LayerNorm" in name or ".norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _round_to(x: int, m: int) -> int:
    return int((x + m - 1) // m * m)


def _build_model_under_limit(input_dim: int, num_classes: int, param_limit: int) -> _Net:
    w2 = int(round(input_dim * 5 / 3))
    w2 = max(256, min(768, _round_to(w2, 8)))
    w1 = int(round(w2 * 2.8))
    w1 = max(w2, min(2048, _round_to(w1, 8)))
    bottleneck = max(64, min(256, _round_to(int(round(w2 * 0.3375)), 8)))
    blocks = 2
    dropout = 0.10

    for _ in range(200):
        model = _Net(input_dim, num_classes, w1=w1, w2=w2, bottleneck=bottleneck, blocks=blocks, dropout=dropout)
        p = _count_trainable_params(model)
        if p <= param_limit:
            return model

        if bottleneck > 64:
            bottleneck = max(64, bottleneck - 16)
            continue
        if w1 > w2:
            w1 = max(w2, w1 - 64)
            continue
        if w2 > 256:
            w2 = max(256, w2 - 32)
            w1 = max(w2, _round_to(int(round(w2 * 2.8)), 8))
            bottleneck = max(64, min(256, _round_to(int(round(w2 * 0.33)), 8)))
            continue
        if blocks > 0:
            blocks -= 1
            continue

        model = _Net(input_dim, num_classes, w1=max(256, w1), w2=max(256, w2), bottleneck=64, blocks=0, dropout=dropout)
        if _count_trainable_params(model) <= param_limit:
            return model
        break

    model = _Net(input_dim, num_classes, w1=512, w2=256, bottleneck=64, blocks=0, dropout=0.10)
    return model


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        try:
            cpu_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or "0")
        except Exception:
            cpu_threads = 0
        if device.type == "cpu":
            nthreads = min(8, os.cpu_count() or 8)
            if cpu_threads > 0:
                nthreads = min(nthreads, cpu_threads)
            try:
                torch.set_num_threads(nthreads)
            except Exception:
                pass

        sample_x, sample_y = next(iter(train_loader))
        inferred_input_dim = int(sample_x.shape[-1])
        input_dim = int(metadata.get("input_dim", inferred_input_dim))
        num_classes = int(metadata.get("num_classes", int(sample_y.max().item()) + 1))
        param_limit = int(metadata.get("param_limit", 2_500_000))

        train_x, train_y = _loader_to_tensors(train_loader, device=device)
        val_x, val_y = _loader_to_tensors(val_loader, device=device)

        if train_x.shape[-1] != input_dim:
            input_dim = int(train_x.shape[-1])

        model = _build_model_under_limit(input_dim, num_classes, param_limit).to(device)

        param_count = _count_trainable_params(model)
        if param_count > param_limit:
            while param_count > param_limit:
                for mod in model.modules():
                    if isinstance(mod, nn.Linear):
                        with torch.no_grad():
                            mod.weight.zero_()
                            if mod.bias is not None:
                                mod.bias.zero_()
                param_count = _count_trainable_params(model)
                break

        with torch.no_grad():
            mean = train_x.mean(dim=0)
            std = train_x.std(dim=0, unbiased=False)
            std = torch.clamp(std, min=1e-6)
            model.norm.set_stats(mean, std)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            _make_param_groups(model, weight_decay=1e-2),
            lr=4e-3,
            betas=(0.9, 0.99),
            eps=1e-8
        )

        n = train_x.shape[0]
        batch_size = 256 if n >= 256 else max(32, int(2 ** math.floor(math.log2(max(32, n)))))
        max_epochs = 260
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=3e-4)

        ema = _EMA(model, decay=0.995)

        best_acc = -1.0
        best_shadow = ema.shadow_clone()
        best_epoch = 0
        patience = 40

        gclip = 1.0

        for epoch in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = train_x.index_select(0, idx)
                yb = train_y.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if gclip is not None and gclip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), gclip)
                optimizer.step()
                ema.update()

            scheduler.step()

            if epoch <= 5 or epoch % 1 == 0:
                backup = ema.store()
                ema.copy_to_model()
                acc = _accuracy(model, val_x, val_y, bs=2048)
                ema.restore(backup)

                if acc > best_acc + 1e-4:
                    best_acc = acc
                    best_shadow = ema.shadow_clone()
                    best_epoch = epoch
                elif epoch - best_epoch >= patience:
                    break

        ema.load_shadow(best_shadow)
        ema.copy_to_model()
        model.eval()

        return model