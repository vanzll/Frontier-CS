import os
import math
import copy
import time
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_cpu_threads():
    try:
        n = os.cpu_count() or 8
        n = max(1, min(8, int(n)))
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass


def _gather_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Unsupported batch format")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        x = x.detach().cpu()
        y = y.detach().cpu()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.long).view(-1)
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs, dim=0) if xs else torch.empty(0, dtype=torch.float32)
    Y = torch.cat(ys, dim=0) if ys else torch.empty(0, dtype=torch.long)
    return X, Y


def _compute_mean_invstd(X: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    if X.numel() == 0:
        return torch.zeros(1, dtype=torch.float32), torch.ones(1, dtype=torch.float32)
    mean = X.mean(dim=0)
    var = X.var(dim=0, unbiased=False)
    invstd = torch.rsqrt(var + eps)
    return mean.to(torch.float32), invstd.to(torch.float32)


def _count_trainable_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


class _ResBlock(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float = 0.0, layerscale_init: float = 1e-3):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, expansion, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion, d_model, bias=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.full((d_model,), float(layerscale_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y * self.gamma


class _ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_blocks: int,
        expansion: int,
        dropout: float = 0.0,
        layerscale_init: float = 1e-3,
        mean: Optional[torch.Tensor] = None,
        invstd: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.d_model = int(input_dim)

        if mean is None:
            mean = torch.zeros(self.d_model, dtype=torch.float32)
        if invstd is None:
            invstd = torch.ones(self.d_model, dtype=torch.float32)
        self.register_buffer("x_mean", mean.view(1, -1).contiguous(), persistent=True)
        self.register_buffer("x_invstd", invstd.view(1, -1).contiguous(), persistent=True)

        self.ln_in = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [_ResBlock(self.d_model, expansion, dropout=dropout, layerscale_init=layerscale_init) for _ in range(n_blocks)]
        )
        self.ln_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.to(dtype=torch.float32)
        x = (x - self.x_mean) * self.x_invstd
        x = self.ln_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)
        return self.head(x)


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is None:
            return
        for name, p in model.named_parameters():
            if p.requires_grad and name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = None


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        _set_cpu_threads()
        torch.manual_seed(0)

        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500000))
        device = metadata.get("device", "cpu")
        if device != "cpu":
            device = "cpu"

        X_train, y_train = _gather_from_loader(train_loader)
        X_val, y_val = _gather_from_loader(val_loader)

        if X_train.numel() == 0:
            model = nn.Linear(input_dim, num_classes)
            return model.to(device)

        mean, invstd = _compute_mean_invstd(X_train, eps=1e-5)

        dropout = 0.05
        layerscale_init = 1e-3

        def build_model(n_blocks: int, expansion: int) -> _ResMLP:
            return _ResMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                n_blocks=n_blocks,
                expansion=expansion,
                dropout=dropout,
                layerscale_init=layerscale_init,
                mean=mean,
                invstd=invstd,
            )

        chosen = None
        for n_blocks in (8, 6, 5, 4, 3, 2, 1):
            exp_max = 1024
            exp_min = 16
            lo, hi = exp_min, exp_max
            best_exp = None
            while lo <= hi:
                mid = (lo + hi) // 2
                m = build_model(n_blocks, mid)
                pc = _count_trainable_params(m)
                if pc <= param_limit:
                    best_exp = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_exp is not None:
                m = build_model(n_blocks, best_exp)
                pc = _count_trainable_params(m)
                if pc <= param_limit:
                    chosen = (n_blocks, best_exp)
                    break

        if chosen is None:
            m = build_model(1, 16)
            pc = _count_trainable_params(m)
            while pc > param_limit:
                if isinstance(m.head, nn.Linear):
                    m.head = nn.Linear(m.d_model, max(2, m.num_classes // 2), bias=True)
                pc = _count_trainable_params(m)
            model = m.to(device)
            return model

        n_blocks, expansion = chosen
        model = build_model(n_blocks, expansion).to(device)

        if _count_trainable_params(model) > param_limit:
            while expansion > 16 and _count_trainable_params(model) > param_limit:
                expansion -= 1
                model = build_model(n_blocks, expansion).to(device)

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        N = X_train.size(0)
        batch_size = 256
        if N < batch_size:
            batch_size = max(32, int(2 ** math.floor(math.log2(max(32, N)))))
            batch_size = min(batch_size, N)

        base_lr = 3e-3
        lr = base_lr * math.sqrt(batch_size / 256.0)

        weight_decay = 0.02
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8)
        max_epochs = 90
        steps_per_epoch = max(1, (N + batch_size - 1) // batch_size)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max_epochs * steps_per_epoch,
            pct_start=0.12,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=12.0,
            final_div_factor=60.0,
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        ema = _EMA(model, decay=0.995)

        best_acc = -1.0
        best_state = None
        patience = 18
        no_improve = 0

        def eval_val() -> float:
            model.eval()
            ema.apply_to(model)
            with torch.inference_mode():
                logits = model(X_val)
                preds = logits.argmax(dim=1)
                acc = (preds == y_val).float().mean().item()
            ema.restore(model)
            return float(acc)

        start_time = time.time()
        time_budget = 3300.0

        for epoch in range(max_epochs):
            if time.time() - start_time > time_budget:
                break

            model.train()
            perm = torch.randperm(N, device=device)
            total_loss = 0.0
            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_train.index_select(0, idx)
                yb = y_train.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update(model)

                total_loss += float(loss.detach().item()) * int(yb.numel())

            val_acc = eval_val()
            if val_acc > best_acc + 1e-5:
                best_acc = val_acc
                ema.apply_to(model)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and epoch >= 25:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        model.to("cpu")
        return model