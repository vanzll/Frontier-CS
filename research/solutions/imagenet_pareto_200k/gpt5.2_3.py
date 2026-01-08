import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
    raise ValueError("Expected batch to be a tuple/list (inputs, targets, ...)")


@torch.inference_mode()
def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.inference_mode()
def _eval_logits(model: nn.Module, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    targets_list = []
    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
        out = model(x)
        logits_list.append(out.detach().cpu())
        targets_list.append(y.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


def _choose_dims(input_dim: int, num_classes: int, param_limit: int) -> Tuple[int, int]:
    candidates: List[Tuple[int, int]] = [
        (288, 192),
        (272, 184),
        (256, 176),
        (248, 176),
        (240, 168),
        (232, 160),
        (224, 160),
        (216, 152),
        (208, 152),
        (200, 144),
        (192, 144),
        (184, 136),
        (176, 136),
        (168, 128),
        (160, 128),
        (152, 120),
        (144, 120),
    ]

    def param_count(h1: int, h2: int) -> int:
        # Linear1: input->h1
        # Linear2: h1->h2
        # Linear3: h2->h2
        # Linear4: h2->num_classes
        # LayerNorms: on h1, h2, h2 => 2*h1 + 4*h2
        c = 0
        c += input_dim * h1 + h1
        c += h1 * h2 + h2
        c += h2 * h2 + h2
        c += h2 * num_classes + num_classes
        c += 2 * h1 + 4 * h2
        return c

    best = None
    best_params = -1
    for h1, h2 in candidates:
        c = param_count(h1, h2)
        if c <= param_limit and c > best_params:
            best = (h1, h2)
            best_params = c
    if best is None:
        # Fallback: minimal dims
        return 128, 96
    return best


class _InputNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("inv_std", inv_std.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class CentroidClassifier(nn.Module):
    def __init__(self, mean: torch.Tensor, inv_std: torch.Tensor, centroids_norm: torch.Tensor):
        super().__init__()
        self.norm = _InputNorm(mean, inv_std)
        self.register_buffer("centroids", centroids_norm.clone().detach())
        self.register_buffer("centroids_sq", (centroids_norm * centroids_norm).sum(dim=1, keepdim=True).t().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x_sq = (x * x).sum(dim=1, keepdim=True)
        # logits proportional to negative squared distance: -||x-c||^2 = 2 x·c - ||x||^2 - ||c||^2
        logits = 2.0 * (x @ self.centroids.t()) - x_sq - self.centroids_sq
        return logits


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, mean: torch.Tensor, inv_std: torch.Tensor, h1: int, h2: int, dropout: float):
        super().__init__()
        self.norm = _InputNorm(mean, inv_std)
        self.fc1 = nn.Linear(input_dim, h1, bias=True)
        self.ln1 = nn.LayerNorm(h1, elementwise_affine=True)
        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.ln2 = nn.LayerNorm(h2, elementwise_affine=True)
        self.fc3 = nn.Linear(h2, h2, bias=True)
        self.ln3 = nn.LayerNorm(h2, elementwise_affine=True)
        self.fc4 = nn.Linear(h2, num_classes, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.drop(x)

        r = x
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + r

        x = self.fc4(x)
        return x


class BlendClassifier(nn.Module):
    def __init__(self, mlp: nn.Module, centroid: nn.Module, alpha: float, t_mlp: float, t_cent: float):
        super().__init__()
        self.mlp = mlp
        self.centroid = centroid
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.register_buffer("t_mlp", torch.tensor(float(t_mlp), dtype=torch.float32))
        self.register_buffer("t_cent", torch.tensor(float(t_cent), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        lm = self.mlp(x) / self.t_mlp
        lc = self.centroid(x) / self.t_cent
        return a * lm + (1.0 - a) * lc


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        metadata = metadata or {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 200_000))
        device = torch.device(metadata.get("device", "cpu"))

        # Compute global mean/std and class centroids (raw), then convert centroids to normalized space.
        eps = 1e-6
        total_sum = torch.zeros(input_dim, dtype=torch.float64)
        total_sqsum = torch.zeros(input_dim, dtype=torch.float64)
        total_n = 0

        class_sum = torch.zeros(num_classes, input_dim, dtype=torch.float64)
        class_cnt = torch.zeros(num_classes, dtype=torch.float64)

        for batch in train_loader:
            x, y = _unpack_batch(batch)
            x = x.detach().to(dtype=torch.float32)
            y = y.detach().to(dtype=torch.long)
            xb = x.to(dtype=torch.float64)
            total_sum += xb.sum(dim=0)
            total_sqsum += (xb * xb).sum(dim=0)
            total_n += xb.shape[0]

            class_sum.index_add_(0, y.cpu(), xb.cpu())
            ones = torch.ones_like(y, dtype=torch.float64).cpu()
            class_cnt.index_add_(0, y.cpu(), ones)

        if total_n == 0:
            mean = torch.zeros(input_dim, dtype=torch.float32)
            inv_std = torch.ones(input_dim, dtype=torch.float32)
        else:
            mean64 = total_sum / float(total_n)
            var64 = total_sqsum / float(total_n) - mean64 * mean64
            var64 = torch.clamp(var64, min=eps)
            std64 = torch.sqrt(var64)
            mean = mean64.to(dtype=torch.float32)
            inv_std = (1.0 / std64).to(dtype=torch.float32)

        # Avoid division by zero in rare cases
        inv_std = torch.where(torch.isfinite(inv_std), inv_std, torch.ones_like(inv_std))
        inv_std = torch.clamp(inv_std, min=0.0, max=1e6)

        # Class centroids
        class_cnt_safe = torch.clamp(class_cnt, min=1.0)
        class_mean64 = class_sum / class_cnt_safe[:, None]
        centroids_raw = class_mean64.to(dtype=torch.float32)
        centroids_norm = (centroids_raw - mean[None, :]) * inv_std[None, :]

        centroid_model = CentroidClassifier(mean=mean, inv_std=inv_std, centroids_norm=centroids_norm).to(device)

        # Build MLP within param budget
        h1, h2 = _choose_dims(input_dim, num_classes, param_limit)
        dropout = 0.15
        mlp_model = MLPClassifier(input_dim, num_classes, mean, inv_std, h1=h1, h2=h2, dropout=dropout).to(device)

        if _count_trainable_params(mlp_model) > param_limit:
            # Further shrink just in case
            h1, h2 = _choose_dims(input_dim, num_classes, max(1, param_limit - 1024))
            mlp_model = MLPClassifier(input_dim, num_classes, mean, inv_std, h1=h1, h2=h2, dropout=dropout).to(device)

        # If no validation loader, just train and return MLP
        has_val = val_loader is not None

        # Training setup
        label_smoothing = 0.05
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=3e-3, weight_decay=1e-2, betas=(0.9, 0.99))

        max_epochs = 180
        warmup_epochs = 8
        min_lr = 3e-4
        max_lr = 3e-3
        grad_clip = 1.0
        patience = 25 if has_val else 0

        best_state = None
        best_val_acc = -1.0
        no_improve = 0

        def set_lr(epoch: int):
            if epoch < warmup_epochs:
                lr = max_lr * float(epoch + 1) / float(max(1, warmup_epochs))
            else:
                t = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                t = min(1.0, max(0.0, t))
                lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        @torch.inference_mode()
        def eval_acc(model: nn.Module, loader) -> float:
            model.eval()
            correct = 0
            total = 0
            for batch in loader:
                x, y = _unpack_batch(batch)
                x = x.to(device=device, dtype=torch.float32, non_blocking=True)
                y = y.to(device=device, dtype=torch.long, non_blocking=True)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
            return float(correct) / float(max(1, total))

        # Train MLP with early stopping on validation accuracy
        for epoch in range(max_epochs):
            set_lr(epoch)
            mlp_model.train()
            for batch in train_loader:
                x, y = _unpack_batch(batch)
                x = x.to(device=device, dtype=torch.float32, non_blocking=True)
                y = y.to(device=device, dtype=torch.long, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = mlp_model(x)
                loss = criterion(out, y)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=grad_clip)
                optimizer.step()

            if has_val:
                va = eval_acc(mlp_model, val_loader)
                if va > best_val_acc + 1e-6:
                    best_val_acc = va
                    best_state = {k: v.detach().cpu().clone() for k, v in mlp_model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        if best_state is not None:
            mlp_model.load_state_dict(best_state, strict=True)

        mlp_model.eval()
        centroid_model.eval()

        # If no validation, return MLP
        if not has_val:
            mlp_model.to("cpu")
            return mlp_model

        # Choose best of centroid/MLP/blend on validation
        mlp_logits, val_targets = _eval_logits(mlp_model, val_loader, device)
        cent_logits, _ = _eval_logits(centroid_model, val_loader, device)

        base_acc_mlp = _accuracy_from_logits(mlp_logits, val_targets)
        base_acc_cent = _accuracy_from_logits(cent_logits, val_targets)

        best_kind = "mlp" if base_acc_mlp >= base_acc_cent else "centroid"
        best_acc = max(base_acc_mlp, base_acc_cent)
        best_alpha, best_t_mlp, best_t_cent = 1.0, 1.0, 1.0

        alphas = [i / 10.0 for i in range(0, 11)]
        temps = [0.7, 1.0, 1.3, 1.7]
        for a in alphas:
            for tm in temps:
                for tc in temps:
                    logits = a * (mlp_logits / tm) + (1.0 - a) * (cent_logits / tc)
                    acc = _accuracy_from_logits(logits, val_targets)
                    if acc > best_acc + 1e-9:
                        best_acc = acc
                        best_kind = "blend"
                        best_alpha, best_t_mlp, best_t_cent = a, tm, tc

        if best_kind == "centroid":
            centroid_model.to("cpu")
            return centroid_model
        if best_kind == "mlp":
            mlp_model.to("cpu")
            return mlp_model

        blended = BlendClassifier(mlp=mlp_model, centroid=centroid_model, alpha=best_alpha, t_mlp=best_t_mlp, t_cent=best_t_cent)
        blended.to("cpu")
        return blended