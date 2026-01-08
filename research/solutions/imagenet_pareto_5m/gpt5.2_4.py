import os
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def _set_num_threads(n: int = 8) -> None:
    try:
        n = int(n)
        if n > 0:
            torch.set_num_threads(n)
            torch.set_num_interop_threads(max(1, min(4, n)))
    except Exception:
        pass


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _gather_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.as_tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.as_tensor(yb)
        xb = xb.detach().cpu()
        yb = yb.detach().cpu()
        if xb.dim() > 2:
            xb = xb.view(xb.size(0), -1)
        xs.append(xb)
        ys.append(yb.view(-1))
    if len(xs) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    X = torch.cat(xs, dim=0).contiguous()
    y = torch.cat(ys, dim=0).contiguous().long()
    return X, y


def _round_to_multiple(x: float, m: int) -> int:
    return int(max(m, int(round(x / m) * m)))


def _stratified_split_indices(y: torch.Tensor, num_classes: int, val_per_class: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    train_idx = []
    val_idx = []
    for c in range(num_classes):
        idx = torch.nonzero(y == c, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        v = min(val_per_class, perm.numel() - 1) if perm.numel() > 1 else 0
        if v > 0:
            val_idx.append(perm[:v])
            train_idx.append(perm[v:])
        else:
            train_idx.append(perm)
    train_idx = torch.cat(train_idx, dim=0) if len(train_idx) else torch.empty(0, dtype=torch.long)
    val_idx = torch.cat(val_idx, dim=0) if len(val_idx) else torch.empty(0, dtype=torch.long)
    return train_idx, val_idx


@torch.no_grad()
def _compute_mean_std(X: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    Xf = X.float()
    mean = Xf.mean(dim=0)
    var = Xf.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps)
    return mean, std


@torch.no_grad()
def _fit_lda(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    shrinkage: float = 0.15,
    jitter_scale: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = X.float()
    y = y.long()
    n, d = X.shape
    if n < num_classes + 2:
        W = torch.zeros((num_classes, d), dtype=torch.float32)
        b = torch.zeros((num_classes,), dtype=torch.float32)
        return W, b

    counts = torch.bincount(y, minlength=num_classes).float()
    counts_clamped = counts.clamp_min(1.0)

    sum_x = torch.zeros((num_classes, d), dtype=torch.float32)
    sum_x.index_add_(0, y, X)
    mu = sum_x / counts_clamped.unsqueeze(1)

    centered = X - mu[y]

    denom = float(max(1, n - num_classes))
    cov = (centered.T @ centered) / denom

    tr = torch.trace(cov).clamp_min(1e-12)
    avg_var = tr / float(d)
    cov = (1.0 - float(shrinkage)) * cov + float(shrinkage) * (avg_var * torch.eye(d, dtype=cov.dtype))

    jitter = float(jitter_scale) * float(avg_var)
    cov = cov + jitter * torch.eye(d, dtype=cov.dtype)

    mu64 = mu.to(torch.float64)
    cov64 = cov.to(torch.float64)

    try:
        chol = torch.linalg.cholesky(cov64)
        Wt = torch.cholesky_solve(mu64.T, chol)  # d x K
        W = Wt.T.to(torch.float32)  # K x d
        q = 0.5 * (mu64 * (Wt.T)).sum(dim=1).to(torch.float32)
    except Exception:
        diag = cov64.diag().clamp_min(1e-8).to(torch.float32)
        W = (mu / diag.unsqueeze(0)).to(torch.float32)
        q = 0.5 * (mu * W).sum(dim=1).to(torch.float32)

    pri = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-8)
    b = (-q + pri.log().to(torch.float32))
    return W, b


class _HybridResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        bottleneck_dim: int,
        n_blocks: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        lda_W: torch.Tensor,
        lda_b: torch.Tensor,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.register_buffer("mean", mean.float().view(1, -1))
        self.register_buffer("inv_std", (1.0 / std.float().clamp_min(1e-6)).view(1, -1))

        self.register_buffer("lda_W", lda_W.float())  # K x D
        self.register_buffer("lda_b", lda_b.float())  # K

        self.lda_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.res_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "fc1": nn.Linear(hidden_dim, bottleneck_dim),
                        "fc2": nn.Linear(bottleneck_dim, hidden_dim),
                        "ln": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self.dropout_p = float(dropout)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.head.weight.mul_(0.05)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return (x - self.mean) * self.inv_std

    def lda_logits_from_norm(self, xn: torch.Tensor) -> torch.Tensor:
        return xn @ self.lda_W.t() + self.lda_b

    def residual_logits_from_norm(self, xn: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(xn)
        h = self.ln_in(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        for blk in self.blocks:
            y = blk["fc1"](h)
            y = self.act(y)
            y = F.dropout(y, p=self.dropout_p, training=self.training)
            y = blk["fc2"](y)
            y = blk["ln"](y)
            y = self.act(y)
            y = F.dropout(y, p=self.dropout_p, training=self.training)
            h = h + y

        h = self.ln_out(h)
        h = self.act(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        return self.head(h)

    def forward_from_normalized(self, xn: torch.Tensor) -> torch.Tensor:
        lda = self.lda_logits_from_norm(xn)
        res = self.residual_logits_from_norm(xn)
        return self.lda_scale * lda + self.res_scale * res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.normalize(x)
        return self.forward_from_normalized(xn)


class _FixedLDAModel(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, lda_W: torch.Tensor, lda_b: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float().view(1, -1))
        self.register_buffer("inv_std", (1.0 / std.float().clamp_min(1e-6)).view(1, -1))
        self.register_buffer("W", lda_W.float())
        self.register_buffer("b", lda_b.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        xn = (x - self.mean) * self.inv_std
        return xn @ self.W.t() + self.b


@torch.no_grad()
def _eval_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).long()
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return float(correct) / float(max(1, total))


def _make_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and ("weight" in name) and ("ln" not in name.lower()) and ("norm" not in name.lower()):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> nn.Module:
        _set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 5_000_000))
        device_str = str(metadata.get("device", "cpu"))
        device = torch.device(device_str)

        torch.manual_seed(0)

        X_tr, y_tr = _gather_from_loader(train_loader)
        X_va, y_va = _gather_from_loader(val_loader) if val_loader is not None else (torch.empty(0), torch.empty(0, dtype=torch.long))
        if X_va.numel() > 0:
            X_all = torch.cat([X_tr, X_va], dim=0)
            y_all = torch.cat([y_tr, y_va], dim=0)
        else:
            X_all = X_tr
            y_all = y_tr

        X_all = X_all.view(X_all.size(0), -1).contiguous()
        if X_all.size(1) != input_dim:
            input_dim = int(X_all.size(1))

        mean, std = _compute_mean_std(X_all)

        Xn_all = (X_all.float() - mean) / std.clamp_min(1e-6)
        lda_W, lda_b = _fit_lda(Xn_all, y_all, num_classes=num_classes, shrinkage=0.18, jitter_scale=2e-4)

        lda_model = _FixedLDAModel(mean=mean, std=std, lda_W=lda_W, lda_b=lda_b).to(device)

        val_per_class = 2
        train_idx, inner_val_idx = _stratified_split_indices(y_all, num_classes=num_classes, val_per_class=val_per_class, seed=1)
        if inner_val_idx.numel() < num_classes:
            train_idx, inner_val_idx = _stratified_split_indices(y_all, num_classes=num_classes, val_per_class=1, seed=1)

        def estimate_params(hidden_dim: int, bottleneck_dim: int, n_blocks: int) -> int:
            h = int(hidden_dim)
            b = int(bottleneck_dim)
            nb = int(n_blocks)
            p = 0
            p += (input_dim + 1) * h  # fc_in
            p += 2 * h  # ln_in
            p += nb * ((h + 1) * b + (b + 1) * h + 2 * h)  # blocks
            p += 2 * h  # ln_out
            p += (h + 1) * num_classes  # head
            p += 2  # lda_scale, res_scale
            return int(p)

        best_cfg = None
        best_p = -1
        ratios = [0.12, 0.15, 0.18, 0.22, 0.26]
        for n_blocks in (4, 3, 2):
            for h in range(2560, 1023, -64):
                for r in ratios:
                    b = _round_to_multiple(h * r, 64)
                    b = max(192, min(b, h))
                    p = estimate_params(h, b, n_blocks)
                    if p <= param_limit and p > best_p:
                        best_p = p
                        best_cfg = (h, b, n_blocks)
                for b in (192, 256, 320, 384, 448, 512):
                    b = int(min(max(192, b), h))
                    b = _round_to_multiple(b, 64)
                    p = estimate_params(h, b, n_blocks)
                    if p <= param_limit and p > best_p:
                        best_p = p
                        best_cfg = (h, b, n_blocks)

        if best_cfg is None:
            best_cfg = (1024, 192, 2)

        hidden_dim, bottleneck_dim, n_blocks = best_cfg

        model = _HybridResidualMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            n_blocks=n_blocks,
            mean=mean,
            std=std,
            lda_W=lda_W,
            lda_b=lda_b,
            dropout=0.10,
        ).to(device)

        if _count_trainable_params(model) > param_limit:
            model = lda_model
            model.eval()
            return model

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_inner_val = X_all[inner_val_idx] if inner_val_idx.numel() > 0 else X_va
        y_inner_val = y_all[inner_val_idx] if inner_val_idx.numel() > 0 else y_va

        train_ds = TensorDataset(X_train, y_train)
        inner_val_ds = TensorDataset(X_inner_val, y_inner_val) if X_inner_val.numel() > 0 else None

        batch_size = 256
        train_dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, num_workers=0, drop_last=False)
        inner_val_dl = DataLoader(inner_val_ds, batch_size=512, shuffle=False, num_workers=0, drop_last=False) if inner_val_ds is not None else None

        criterion = nn.CrossEntropyLoss(label_smoothing=0.06)

        base_lr = 3e-3
        weight_decay = 1.5e-4
        optimizer = torch.optim.AdamW(_make_param_groups(model, weight_decay=weight_decay), lr=base_lr, betas=(0.9, 0.95))

        max_epochs = 220
        patience = 30
        noise_std = 0.04

        steps_per_epoch = max(1, len(train_dl))
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = min(200, max(50, total_steps // 20))
        global_step = 0

        best_state = None
        best_val = -1.0
        no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                xn = model.normalize(xb)
                if noise_std > 0:
                    xn = xn + torch.randn_like(xn) * noise_std

                if global_step < warmup_steps:
                    lr_mult = float(global_step + 1) / float(max(1, warmup_steps))
                else:
                    t = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr_mult = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * lr_mult

                logits = model.forward_from_normalized(xn)
                loss = criterion(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1

            if inner_val_dl is not None:
                val_acc = _eval_accuracy(model, inner_val_dl, device)
            else:
                val_acc = -1.0

            if val_acc > best_val + 1e-4:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and epoch >= 40:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        full_ds = TensorDataset(X_all, y_all)
        full_dl = DataLoader(full_ds, batch_size=min(batch_size, len(full_ds)), shuffle=True, num_workers=0, drop_last=False)

        ft_lr = 6e-4
        ft_epochs = 6
        optimizer = torch.optim.AdamW(_make_param_groups(model, weight_decay=weight_decay * 0.7), lr=ft_lr, betas=(0.9, 0.95))
        total_steps = ft_epochs * max(1, len(full_dl))
        warmup_steps = min(30, max(10, total_steps // 10))
        global_step = 0
        for _ in range(ft_epochs):
            model.train()
            for xb, yb in full_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                xn = model.normalize(xb)
                xn = xn + torch.randn_like(xn) * (noise_std * 0.5)

                if global_step < warmup_steps:
                    lr_mult = float(global_step + 1) / float(max(1, warmup_steps))
                else:
                    t = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr_mult = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
                for pg in optimizer.param_groups:
                    pg["lr"] = ft_lr * lr_mult

                logits = model.forward_from_normalized(xn)
                loss = criterion(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1

        if inner_val_dl is not None:
            hybrid_acc = _eval_accuracy(model, inner_val_dl, device)
            lda_acc = _eval_accuracy(lda_model, inner_val_dl, device)
            chosen = model if hybrid_acc >= lda_acc else lda_model
        else:
            chosen = model

        chosen.eval()
        return chosen.to(device)