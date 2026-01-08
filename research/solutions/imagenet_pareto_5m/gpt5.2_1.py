import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _rebuild_loader(loader, batch_size: int, shuffle: bool, seed: int = 0):
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return loader
    collate_fn = getattr(loader, "collate_fn", None)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
        generator=g if shuffle else None,
    )


@torch.no_grad()
def _collect_xy(loader, input_dim: int, device: str = "cpu"):
    xs = []
    ys = []
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.as_tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.as_tensor(yb)
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.long)
        if xb.dim() == 1:
            xb = xb.view(1, -1)
        xb = xb.view(xb.size(0), -1)
        if xb.size(1) != input_dim:
            xb = xb[:, :input_dim]
        xs.append(xb)
        ys.append(yb.view(-1))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return X, y


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


@torch.no_grad()
def _compute_lda_from_train(X: torch.Tensor, y: torch.Tensor, num_classes: int, shrinkage: float = 0.2):
    X = X.float()
    N, D = X.shape
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False).clamp_min(1e-6)
    inv_std = 1.0 / std
    Xs = (X - mean) * inv_std

    counts = torch.bincount(y, minlength=num_classes).to(Xs.device)
    priors = (counts.float() + 1e-6) / float(N + 1e-6 * num_classes)
    log_prior = torch.log(priors)

    onehot = F.one_hot(y, num_classes=num_classes).to(dtype=Xs.dtype)
    sums = onehot.t().matmul(Xs)  # (K, D)
    denom = counts.clamp_min(1).unsqueeze(1).to(dtype=Xs.dtype)
    mus = sums / denom

    try:
        Xsd = Xs.double()
        musd = mus.double()
        centered = Xsd - musd[y]
        cov = centered.t().matmul(centered) / float(max(1, N - num_classes))
        tr = cov.diag().mean()
        cov = (1.0 - shrinkage) * cov + shrinkage * tr * torch.eye(D, dtype=cov.dtype, device=cov.device)

        jitter = 1e-6
        inv_cov = None
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(cov + jitter * torch.eye(D, dtype=cov.dtype, device=cov.device))
                inv_cov = torch.cholesky_inverse(L)
                break
            except Exception:
                jitter *= 10.0
        if inv_cov is None:
            raise RuntimeError("Cholesky failed")

        W = (inv_cov.matmul(musd.t())).t().contiguous()  # (K, D)
        tmp = musd.matmul(inv_cov)  # (K, D)
        quad = (tmp * musd).sum(dim=1)  # (K,)
        b = (-0.5 * quad + log_prior.double()).contiguous()

        Wf = W.float()
        bf = b.float()
    except Exception:
        Wf = mus.float().contiguous()
        bf = (-0.5 * (Wf * Wf).sum(dim=1) + log_prior.float()).contiguous()

    with torch.no_grad():
        logits = Xs.matmul(Wf.t()) + bf
        s = logits.std().clamp_min(1e-6)
        lda_scale = (1.0 / s).item()

    return mean.float(), inv_std.float(), Wf, bf, float(lda_scale)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, num_blocks: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln": nn.LayerNorm(hidden_dim),
                        "fc1": nn.Linear(hidden_dim, hidden_dim),
                        "fc2": nn.Linear(hidden_dim, hidden_dim),
                        "drop": nn.Dropout(dropout),
                    }
                )
            )
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        x = F.gelu(x)
        for blk in self.blocks:
            r = x
            z = blk["ln"](x)
            z = blk["fc1"](z)
            z = F.gelu(z)
            z = blk["drop"](z)
            z = blk["fc2"](z)
            z = blk["drop"](z)
            x = F.gelu(r + z)
        x = self.out_ln(x)
        return self.head(x)


class EnsembleNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        lda_W: torch.Tensor,
        lda_b: torch.Tensor,
        lda_scale: float,
    ):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1).contiguous())
        self.register_buffer("inv_std", inv_std.view(1, -1).contiguous())
        self.register_buffer("lda_W", lda_W.contiguous())  # (K, D)
        self.register_buffer("lda_b", lda_b.contiguous())  # (K,)
        self.register_buffer("lda_scale", torch.tensor(float(lda_scale), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.mlp = ResidualMLP(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(dtype=torch.float32)
        if x.dim() == 1:
            x = x.view(1, -1)
        x = x.view(x.size(0), -1)
        x = (x - self.mean) * self.inv_std
        lda_logits = (x.matmul(self.lda_W.t()) + self.lda_b) * self.lda_scale
        mlp_logits = self.mlp(x)
        return mlp_logits + self.beta * lda_logits


@torch.no_grad()
def _accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            xb = torch.as_tensor(xb)
        if not torch.is_tensor(yb):
            yb = torch.as_tensor(yb)
        xb = xb.to(dtype=torch.float32)
        yb = yb.to(dtype=torch.long)
        if xb.dim() == 1:
            xb = xb.view(1, -1)
        xb = xb.view(xb.size(0), -1)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb.view(-1)).sum().item()
        total += yb.numel()
    return float(correct) / float(max(1, total))


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        device = metadata.get("device", "cpu")
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 5_000_000))

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        torch.manual_seed(0)
        np.random.seed(0)

        X_train, y_train = _collect_xy(train_loader, input_dim=input_dim, device=device)
        mean, inv_std, lda_W, lda_b, lda_scale = _compute_lda_from_train(X_train, y_train, num_classes=num_classes, shrinkage=0.2)

        dropout = 0.10
        num_blocks = 5
        hidden_dim = 672

        model = EnsembleNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            mean=mean,
            inv_std=inv_std,
            lda_W=lda_W,
            lda_b=lda_b,
            lda_scale=lda_scale,
        ).to(device)

        while _count_trainable_params(model) > param_limit and hidden_dim > 128:
            hidden_dim -= 16
            model = EnsembleNet(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                mean=mean,
                inv_std=inv_std,
                lda_W=lda_W,
                lda_b=lda_b,
                lda_scale=lda_scale,
            ).to(device)

        if _count_trainable_params(model) > param_limit:
            num_blocks = 4
            hidden_dim = 640
            model = EnsembleNet(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                mean=mean,
                inv_std=inv_std,
                lda_W=lda_W,
                lda_b=lda_b,
                lda_scale=lda_scale,
            ).to(device)
            while _count_trainable_params(model) > param_limit and hidden_dim > 128:
                hidden_dim -= 16
                model = EnsembleNet(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks,
                    dropout=dropout,
                    mean=mean,
                    inv_std=inv_std,
                    lda_W=lda_W,
                    lda_b=lda_b,
                    lda_scale=lda_scale,
                ).to(device)

        train_samples = int(metadata.get("train_samples", len(getattr(train_loader, "dataset", [])) or X_train.size(0)))
        val_samples = int(metadata.get("val_samples", len(getattr(val_loader, "dataset", [])) if val_loader is not None else 0))

        train_bs = min(1024, max(64, train_samples))
        val_bs = min(1024, max(128, val_samples)) if val_loader is not None else 1024

        train_loader2 = _rebuild_loader(train_loader, batch_size=train_bs, shuffle=True, seed=0)
        val_loader2 = _rebuild_loader(val_loader, batch_size=val_bs, shuffle=False, seed=0) if val_loader is not None else None

        max_epochs = 200
        patience = 30

        lr_max = 2.2e-3
        lr_min = 1.0e-5
        wd = 2.0e-2
        label_smoothing = 0.05

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

        steps_per_epoch = max(1, len(train_loader2))
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = max(10, total_steps // 20)

        def lr_at(step: int) -> float:
            if step < warmup_steps:
                return lr_max * float(step + 1) / float(warmup_steps)
            t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))

        mixup_prob = 0.25
        mixup_alpha = 0.2

        best_state = None
        best_acc = -1.0
        bad_epochs = 0
        global_step = 0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader2:
                if not torch.is_tensor(xb):
                    xb = torch.as_tensor(xb)
                if not torch.is_tensor(yb):
                    yb = torch.as_tensor(yb)
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long).view(-1)

                lr = lr_at(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                do_mix = (mixup_prob > 0) and (np.random.rand() < mixup_prob) and (xb.size(0) >= 2)
                if do_mix:
                    perm = torch.randperm(xb.size(0), device=xb.device)
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    if lam < 0.5:
                        lam = 1.0 - lam
                    xb2 = xb[perm]
                    yb2 = yb[perm]
                    xmix = xb.mul(lam).add_(xb2, alpha=(1.0 - lam))

                    y1 = F.one_hot(yb, num_classes=num_classes).to(dtype=torch.float32)
                    y2 = F.one_hot(yb2, num_classes=num_classes).to(dtype=torch.float32)
                    if label_smoothing > 0:
                        y1 = y1 * (1.0 - label_smoothing) + (label_smoothing / num_classes)
                        y2 = y2 * (1.0 - label_smoothing) + (label_smoothing / num_classes)
                    ymix = y1.mul(lam).add_(y2, alpha=(1.0 - lam))

                    logits = model(xmix)
                    loss = _soft_cross_entropy(logits, ymix)
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

            if val_loader2 is not None:
                val_acc = _accuracy(model, val_loader2)
                if val_acc > best_acc + 1e-4:
                    best_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        return model.to(device)