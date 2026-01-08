import os
import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {}
        self._names = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._names.append(name)
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def apply_and_restore(self, model: nn.Module):
        ema = self

        class _Ctx:
            def __init__(self):
                self.backup = {}

            def __enter__(self_nonlocal):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in ema.shadow:
                            self_nonlocal.backup[name] = p.detach().clone()
                            p.copy_(ema.shadow[name])
                return model

            def __exit__(self_nonlocal, exc_type, exc, tb):
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in self_nonlocal.backup:
                            p.copy_(self_nonlocal.backup[name])
                return False

        return _Ctx()


class _CentroidResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_blocks: int,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        centroids: torch.Tensor,
        dropout: float = 0.08,
        centroid_tau: float = 10.0,
        init_alpha: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        self.register_buffer("mean", mean.view(1, -1).contiguous(), persistent=True)
        self.register_buffer("inv_std", inv_std.view(1, -1).contiguous(), persistent=True)

        centroids = centroids.contiguous()
        self.register_buffer("centroids", centroids, persistent=True)
        self.centroid_tau = float(centroid_tau)

        self.stem = nn.Linear(self.input_dim, self.hidden_dim)
        self.stem_ln = nn.LayerNorm(self.hidden_dim)

        self.blocks_ln = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_blocks)])
        self.blocks_fc = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_blocks)])

        self.final_ln = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

        self.drop = nn.Dropout(float(dropout))

        init_alpha = max(1e-6, float(init_alpha))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = self._standardize(x)

        h = self.stem(x_std)
        h = self.stem_ln(h)
        h = F.gelu(h)

        for ln, fc in zip(self.blocks_ln, self.blocks_fc):
            r = h
            y = ln(h)
            y = fc(y)
            y = F.gelu(y)
            y = self.drop(y)
            h = r + y

        h = self.final_ln(h)
        h = F.gelu(h)
        logits_mlp = self.head(h)

        x_n = F.normalize(x_std, dim=-1, eps=1e-6)
        c_n = self.centroids
        logits_c = (x_n @ c_n.t()) * self.centroid_tau

        alpha = torch.exp(self.log_alpha).clamp(0.0, 10.0)
        return logits_mlp + alpha * logits_c


def _compute_train_stats_and_centroids(
    train_loader,
    input_dim: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    sum_x = torch.zeros(input_dim, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(input_dim, dtype=torch.float64, device=device)
    n_total = 0

    cent_sum = torch.zeros(num_classes, input_dim, dtype=torch.float64, device=device)
    cent_cnt = torch.zeros(num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
            yb = yb.to(device=device, dtype=torch.long, non_blocking=False)
            bs = xb.shape[0]
            n_total += bs
            x64 = xb.to(torch.float64)
            sum_x += x64.sum(dim=0)
            sum_x2 += (x64 * x64).sum(dim=0)

            for c in range(num_classes):
                mask = (yb == c)
                if mask.any():
                    x_c = x64[mask]
                    cent_sum[c] += x_c.sum(dim=0)
                    cent_cnt[c] += x_c.shape[0]

    mean = (sum_x / max(1, n_total)).to(torch.float32)
    var = (sum_x2 / max(1, n_total)) - (sum_x / max(1, n_total)) ** 2
    var = torch.clamp(var, min=1e-8).to(torch.float32)
    std = torch.sqrt(var)
    inv_std = (1.0 / torch.clamp(std, min=1e-4)).to(torch.float32)

    centroids = cent_sum / cent_cnt.clamp_min(1).unsqueeze(1).to(torch.float64)
    centroids = centroids.to(torch.float32)
    centroids = (centroids - mean.view(1, -1)) * inv_std.view(1, -1)
    centroids = F.normalize(centroids, dim=-1, eps=1e-6)

    std_mean_scalar = float(std.mean().item())
    return mean.cpu(), inv_std.cpu(), centroids.cpu(), std_mean_scalar


@torch.no_grad()
def _accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=False)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        md = metadata or {}
        input_dim = int(md.get("input_dim", 384))
        num_classes = int(md.get("num_classes", 128))
        param_limit = int(md.get("param_limit", 5_000_000))
        device_str = md.get("device", "cpu")
        device = torch.device(device_str)

        try:
            torch.set_num_threads(min(8, max(1, (os.cpu_count() or 8))))
        except Exception:
            pass

        mean, inv_std, centroids, std_mean_scalar = _compute_train_stats_and_centroids(
            train_loader=train_loader,
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
        )

        hidden_dim = 1200
        num_blocks = 3
        dropout = 0.08
        centroid_tau = 10.0
        init_alpha = 1.0

        model = _CentroidResMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            mean=mean,
            inv_std=inv_std,
            centroids=centroids,
            dropout=dropout,
            centroid_tau=centroid_tau,
            init_alpha=init_alpha,
        ).to(device)

        if _count_trainable_params(model) > param_limit:
            candidates = [
                (3, 1184),
                (2, 1440),
                (4, 1024),
                (3, 1120),
                (2, 1280),
                (1, 2048),
                (1, 1536),
                (1, 1024),
                (0, 0),
            ]
            for nb, hd in candidates:
                if nb <= 0:
                    break
                m = _CentroidResMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dim=int(hd),
                    num_blocks=int(nb),
                    mean=mean,
                    inv_std=inv_std,
                    centroids=centroids,
                    dropout=dropout,
                    centroid_tau=centroid_tau,
                    init_alpha=init_alpha,
                ).to(device)
                if _count_trainable_params(m) <= param_limit:
                    model = m
                    break

        if _count_trainable_params(model) > param_limit:
            class _CentroidOnly(nn.Module):
                def __init__(self, mean_, inv_std_, centroids_, tau_):
                    super().__init__()
                    self.register_buffer("mean", mean_.view(1, -1), persistent=True)
                    self.register_buffer("inv_std", inv_std_.view(1, -1), persistent=True)
                    self.register_buffer("centroids", centroids_, persistent=True)
                    self.tau = float(tau_)

                def forward(self, x):
                    x = (x - self.mean) * self.inv_std
                    x = F.normalize(x, dim=-1, eps=1e-6)
                    return (x @ self.centroids.t()) * self.tau

            model = _CentroidOnly(mean, inv_std, centroids, centroid_tau).to(device)
            model.eval()
            return model.cpu()

        lr = 3e-3
        weight_decay = 0.06
        label_smoothing = 0.08
        max_epochs = 300
        min_epochs = 40
        patience = 45
        warmup_epochs = 6
        grad_clip = 1.0
        ema_decay = 0.995

        steps_per_epoch = max(1, len(train_loader))
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        min_lr = 2e-5

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            t = min(1.0, max(0.0, t))
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            floor = min_lr / lr
            return floor + (1.0 - floor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        ema = _EMA(model, decay=ema_decay)

        noise_std = 0.02 * max(1e-6, std_mean_scalar)
        mixup_p = 0.15
        mixup_alpha = 0.2
        beta_dist = torch.distributions.Beta(mixup_alpha, mixup_alpha)

        best_acc = -1.0
        best_state = None
        no_improve = 0
        global_step = 0

        start_time = time.time()
        time_budget = 3300.0

        for epoch in range(max_epochs):
            if (time.time() - start_time) > time_budget:
                break

            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
                yb = yb.to(device=device, dtype=torch.long, non_blocking=False)

                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std

                do_mix = (torch.rand((), device=device).item() < mixup_p) and (xb.shape[0] > 1)
                if do_mix:
                    lam = float(beta_dist.sample().to(device).item())
                    idx = torch.randperm(xb.shape[0], device=device)
                    xb_mix = lam * xb + (1.0 - lam) * xb[idx]
                    y_a = yb
                    y_b = yb[idx]
                    logits = model(xb_mix)
                    loss = lam * ce(logits, y_a) + (1.0 - lam) * ce(logits, y_b)
                else:
                    logits = model(xb)
                    loss = ce(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                ema.update(model)
                scheduler.step()

                global_step += 1

            with ema.apply_and_restore(model):
                val_acc = _accuracy(model, val_loader, device=device)

            if val_acc > best_acc + 1e-6:
                best_acc = val_acc
                with ema.apply_and_restore(model):
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch + 1 >= min_epochs and no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        else:
            with ema.apply_and_restore(model):
                pass

        model.eval()
        return model.cpu()