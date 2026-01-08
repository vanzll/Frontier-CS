import math
import os
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn


def _set_threads():
    try:
        n = os.cpu_count() or 8
        torch.set_num_threads(min(8, n))
    except Exception:
        pass


def _collect_xy(loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected batches as (inputs, targets)")
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        xs.append(x.detach().to(device=device, dtype=torch.float32))
        ys.append(y.detach().to(device=device, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return X, y


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


class ResidualBottleneck(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class FixedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.weight.t()) + self.bias


class CosineHead(nn.Module):
    def __init__(self, proto: torch.Tensor, scale: float):
        super().__init__()
        self.register_buffer("proto", proto)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=-1)
        p = torch.nn.functional.normalize(self.proto, dim=-1)
        return (x.matmul(p.t())) * self.scale


class MLPResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int,
        bottleneck: int,
        depth: int,
        dropout: float,
        mu: torch.Tensor,
        inv_std: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("inv_std", inv_std)

        self.inp = nn.Linear(input_dim, width, bias=True)
        self.in_ln = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.in_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[ResidualBottleneck(width, bottleneck, dropout) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes, bias=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) * self.inv_std
        x = self.inp(x)
        x = self.in_ln(x)
        x = self.act(x)
        x = self.in_drop(x)
        x = self.blocks(x)
        x = self.out_ln(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(x)
        return self.head(f)


class CentroidClassifier(nn.Module):
    def __init__(self, mu: torch.Tensor, inv_std: torch.Tensor, proto: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("inv_std", inv_std)
        self.register_buffer("proto", proto)
        bias = -0.5 * (proto * proto).sum(dim=1)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) * self.inv_std
        return x.matmul(self.proto.t()) + self.bias


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _estimate_params(input_dim: int, num_classes: int, width: int, bottleneck: int, depth: int) -> int:
    # inp: D*w + w
    # in_ln: 2w
    # per block: ln 2w + fc1 w*b + b + fc2 b*w + w = 2wb + b + 3w
    # out_ln: 2w
    # head: w*C + C
    D, C, w, b, d = input_dim, num_classes, width, bottleneck, depth
    return (D * w + w) + (2 * w) + d * (2 * w * b + b + 3 * w) + (2 * w) + (w * C + C)


def _choose_arch(input_dim: int, num_classes: int, param_limit: int) -> Tuple[int, int, int]:
    best = None
    best_params = -1

    depths = [2, 1, 3]
    widths = list(range(384, 801, 32))
    for d in depths:
        for w in widths:
            if w < 256:
                continue
            b_candidates = sorted(set([
                max(64, (w // 4 // 32) * 32),
                max(64, (w // 3 // 32) * 32),
                max(64, (w // 2 // 32) * 32),
                160, 192, 224, 256, 288, 320, 352, 384
            ]))
            b_candidates = [b for b in b_candidates if 64 <= b <= w]
            for b in b_candidates:
                p = _estimate_params(input_dim, num_classes, w, b, d)
                if p <= param_limit and p > best_params:
                    best_params = p
                    best = (w, b, d)

    if best is None:
        # fallback very small
        return 256, 128, 1
    return best


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        _set_threads()
        torch.manual_seed(0)

        if metadata is None:
            metadata = {}
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        Xtr, ytr = _collect_xy(train_loader, device)
        Xva, yva = _collect_xy(val_loader, device)

        input_dim = int(metadata.get("input_dim", Xtr.shape[1]))
        num_classes = int(metadata.get("num_classes", int(ytr.max().item()) + 1))
        param_limit = int(metadata.get("param_limit", 1_000_000))

        Xtr = Xtr[:, :input_dim].contiguous()
        Xva = Xva[:, :input_dim].contiguous()

        with torch.no_grad():
            mu = Xtr.mean(dim=0)
            var = (Xtr - mu).pow(2).mean(dim=0)
            inv_std = (var + 1e-6).rsqrt()

        # Simple centroid classifier on input space as a baseline / fallback
        with torch.no_grad():
            Xn = (Xtr - mu) * inv_std
            proto_in = torch.zeros((num_classes, input_dim), device=device, dtype=torch.float32)
            counts = torch.zeros((num_classes,), device=device, dtype=torch.float32)
            proto_in.index_add_(0, ytr, Xn)
            counts.index_add_(0, ytr, torch.ones_like(ytr, dtype=torch.float32))
            proto_in = proto_in / counts.clamp_min(1.0).unsqueeze(1)
            centroid_model = CentroidClassifier(mu, inv_std, proto_in).to(device)
            centroid_model.eval()
            val_logits_centroid = centroid_model(Xva)
            val_acc_centroid = _accuracy_from_logits(val_logits_centroid, yva)

        width, bottleneck, depth = _choose_arch(input_dim, num_classes, param_limit)
        dropout = 0.10

        model = MLPResNet(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            bottleneck=bottleneck,
            depth=depth,
            dropout=dropout,
            mu=mu,
            inv_std=inv_std,
        ).to(device)

        if _param_count(model) > param_limit:
            # Emergency shrink
            width, bottleneck, depth = 512, 256, 1
            model = MLPResNet(
                input_dim=input_dim,
                num_classes=num_classes,
                width=width,
                bottleneck=bottleneck,
                depth=depth,
                dropout=dropout,
                mu=mu,
                inv_std=inv_std,
            ).to(device)

        n = Xtr.shape[0]
        batch_size = 256 if n >= 256 else n
        steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)

        max_epochs = 300
        patience = 45
        label_smoothing = 0.05
        mixup_p = 0.4
        mixup_alpha = 0.15
        noise_std = 0.01

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.02,
        )

        total_steps = max_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-3,
            total_steps=total_steps,
            pct_start=0.10,
            anneal_strategy="cos",
            div_factor=20.0,
            final_div_factor=1000.0,
        )

        best_state = None
        best_val = -1.0
        bad = 0

        beta_dist = torch.distributions.Beta(mixup_alpha, mixup_alpha)

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = Xtr[idx]
                yb = ytr[idx]

                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std

                do_mix = (mixup_alpha > 0) and (torch.rand((), device=device).item() < mixup_p) and (xb.shape[0] >= 2)
                if do_mix:
                    lam = float(beta_dist.sample(()).clamp(0.5, 0.98).item())
                    rperm = torch.randperm(xb.shape[0], device=device)
                    xb2 = xb[rperm]
                    yb2 = yb[rperm]
                    xb = xb * lam + xb2 * (1.0 - lam)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                if do_mix:
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb2)
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)
                val_acc = _accuracy_from_logits(val_logits, yva)

            if val_acc > best_val + 1e-5:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # Prototype head in embedding space (try Euclidean and cosine heads, keep best on val)
        model.eval()
        with torch.no_grad():
            ftr = model.forward_features(Xtr)
            fva = model.forward_features(Xva)

            proto = torch.zeros((num_classes, ftr.shape[1]), device=device, dtype=torch.float32)
            counts = torch.zeros((num_classes,), device=device, dtype=torch.float32)
            proto.index_add_(0, ytr, ftr)
            counts.index_add_(0, ytr, torch.ones_like(ytr, dtype=torch.float32))
            proto = proto / counts.clamp_min(1.0).unsqueeze(1)

            # Euclidean-proto linear logits
            bias_e = -0.5 * (proto * proto).sum(dim=1)
            head_e = FixedLinear(proto, bias_e)
            logits_e = head_e(fva)
            acc_e = _accuracy_from_logits(logits_e, yva)

            # Cosine head
            best_cos_acc = -1.0
            best_cos_scale = 10.0
            for scale in (5.0, 10.0, 15.0, 20.0, 30.0):
                head_c = CosineHead(proto, scale=scale)
                logits_c = head_c(fva)
                acc_c = _accuracy_from_logits(logits_c, yva)
                if acc_c > best_cos_acc:
                    best_cos_acc = acc_c
                    best_cos_scale = scale

            # Original head accuracy
            logits_o = model.head(fva)
            acc_o = _accuracy_from_logits(logits_o, yva)

            # Choose best among (original), (euclidean proto), (cosine proto), and input centroid fallback
            best_choice = "orig"
            best_acc = acc_o

            if acc_e > best_acc + 1e-6:
                best_acc = acc_e
                best_choice = "euc"

            if best_cos_acc > best_acc + 1e-6:
                best_acc = best_cos_acc
                best_choice = "cos"

            if val_acc_centroid > best_acc + 1e-6:
                best_acc = val_acc_centroid
                best_choice = "centroid"

        if best_choice == "centroid":
            centroid_model.eval()
            return centroid_model

        if best_choice == "euc":
            model.head = FixedLinear(proto, bias_e).to(device)
        elif best_choice == "cos":
            model.head = CosineHead(proto, scale=best_cos_scale).to(device)

        model.eval()
        return model