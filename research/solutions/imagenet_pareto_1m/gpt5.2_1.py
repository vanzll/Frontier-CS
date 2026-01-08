import os
import math
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_threads():
    try:
        n = os.cpu_count() or 8
        torch.set_num_threads(max(1, min(8, n)))
    except Exception:
        pass


def _collect_from_loader(loader, device="cpu"):
    xs = []
    ys = []
    if loader is None:
        return None, None
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            continue
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        x = x.detach().to(device=device, dtype=torch.float32, non_blocking=False)
        y = y.detach().to(device=device, dtype=torch.long, non_blocking=False)
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    X = torch.cat(xs, dim=0).contiguous()
    Y = torch.cat(ys, dim=0).contiguous()
    return X, Y


def _accuracy(model, X, Y, batch_size=512):
    model.eval()
    correct = 0
    total = int(Y.numel())
    with torch.inference_mode():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size]
            yb = Y[i : i + batch_size]
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
    return correct / max(1, total)


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.shadow = {}
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
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model: nn.Module, shadow=None):
        sd = self.shadow if shadow is None else shadow
        for name, p in model.named_parameters():
            if p.requires_grad and name in sd:
                p.data.copy_(sd[name])


class ResidualCosineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, mean: torch.Tensor, std: torch.Tensor, dropout: float = 0.12, scale: float = 16.0):
        super().__init__()
        self.register_buffer("mu", mean.view(1, -1).contiguous())
        self.register_buffer("sigma", std.view(1, -1).contiguous())
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.drop = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        self.register_buffer("prototypes", torch.empty(0))
        self.use_prototypes = False
        self.scale = float(scale)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sigma
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.drop(h)
        h2 = self.fc2(h)
        h2 = F.gelu(h2)
        h2 = self.drop(h2)
        h = self.ln(h + h2)
        z = F.normalize(h, dim=1, eps=1e-12)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        if self.use_prototypes and self.prototypes.numel() > 0:
            return z @ self.prototypes.t()
        w = F.normalize(self.classifier.weight, dim=1, eps=1e-12)
        return (z @ w.t()) * self.scale

    @torch.no_grad()
    def build_prototypes(self, X: torch.Tensor, Y: torch.Tensor, num_classes: int):
        self.eval()
        z_list = []
        bs = 512
        for i in range(0, X.shape[0], bs):
            z_list.append(self.embed(X[i : i + bs]))
        Z = torch.cat(z_list, dim=0)
        protos = torch.zeros((num_classes, Z.shape[1]), device=Z.device, dtype=Z.dtype)
        counts = torch.zeros((num_classes,), device=Z.device, dtype=torch.float32)
        for c in range(num_classes):
            mask = (Y == c)
            if mask.any():
                protos[c] = Z[mask].mean(dim=0)
                counts[c] = float(mask.sum().item())
        missing = (counts == 0)
        if missing.any():
            global_mean = Z.mean(dim=0)
            protos[missing] = global_mean
        protos = F.normalize(protos, dim=1, eps=1e-12)
        self.prototypes = protos


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> nn.Module:
        _set_threads()
        torch.manual_seed(0)

        device = "cpu"
        if metadata is not None and isinstance(metadata, dict):
            device = metadata.get("device", "cpu") or "cpu"

        Xtr, Ytr = _collect_from_loader(train_loader, device=device)
        Xva, Yva = _collect_from_loader(val_loader, device=device)

        if Xtr is None or Ytr is None:
            raise RuntimeError("Training loader produced no data.")

        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", int(Xtr.shape[1])))
        num_classes = int(metadata.get("num_classes", int(Ytr.max().item()) + 1))
        param_limit = int(metadata.get("param_limit", 1_000_000))

        mean = Xtr.mean(dim=0)
        std = Xtr.std(dim=0, unbiased=False).clamp_min(1e-6)

        hidden_dim = 773
        model = ResidualCosineMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            mean=mean,
            std=std,
            dropout=0.12,
            scale=16.0,
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            for h in [768, 760, 752, 744, 736, 720, 704, 688, 672, 640, 608, 576, 544, 512]:
                model = ResidualCosineMLP(
                    input_dim=input_dim,
                    hidden_dim=h,
                    num_classes=num_classes,
                    mean=mean,
                    std=std,
                    dropout=0.12,
                    scale=16.0,
                ).to(device)
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if param_count <= param_limit:
                    break

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            for p in model.parameters():
                p.requires_grad_(False)
            return model.to("cpu").eval()

        n = int(Xtr.shape[0])
        batch_size = 256 if n >= 256 else 128
        steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)

        max_epochs = 160
        lr = 3e-3
        weight_decay = 1.2e-4
        label_smoothing = 0.08
        grad_clip = 1.0

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max_epochs * steps_per_epoch,
            pct_start=0.12,
            anneal_strategy="cos",
            final_div_factor=30.0,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        ema = _EMA(model, decay=0.995)
        best_acc = -1.0
        best_shadow = None
        patience = 20
        bad = 0

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb = Xtr[idx]
                yb = Ytr[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                ema.update(model)

            if Xva is not None and Yva is not None:
                tmp_model = model
                state_backup = {name: p.detach().clone() for name, p in tmp_model.named_parameters() if p.requires_grad}
                ema.copy_to(tmp_model)
                acc = _accuracy(tmp_model, Xva, Yva, batch_size=512)
                with torch.no_grad():
                    for name, p in tmp_model.named_parameters():
                        if p.requires_grad and name in state_backup:
                            p.data.copy_(state_backup[name])

                if acc > best_acc + 1e-6:
                    best_acc = acc
                    best_shadow = {k: v.detach().clone() for k, v in ema.shadow.items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if best_shadow is not None:
            ema.copy_to(model, shadow=best_shadow)
        else:
            ema.copy_to(model)

        Xproto, Yproto = Xtr, Ytr
        if Xva is not None and Yva is not None:
            Xproto = torch.cat([Xtr, Xva], dim=0)
            Yproto = torch.cat([Ytr, Yva], dim=0)

        model.build_prototypes(Xproto, Yproto, num_classes=num_classes)

        if Xva is not None and Yva is not None:
            model.use_prototypes = False
            acc_linear = _accuracy(model, Xva, Yva, batch_size=512)
            model.use_prototypes = True
            acc_proto = _accuracy(model, Xva, Yva, batch_size=512)
            model.use_prototypes = (acc_proto >= acc_linear - 1e-6)
        else:
            model.use_prototypes = True

        model.eval().to("cpu")
        return model