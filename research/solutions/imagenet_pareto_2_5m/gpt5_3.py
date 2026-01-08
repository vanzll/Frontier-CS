import math
import copy
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ResidualBottleneck(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.1, drop_path: float = 0.0, layerscale: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(dim) * layerscale)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        y = y * self.gamma
        y = self.drop_path(y)
        return x + y


class MLPResNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, mids: List[int], drop: float = 0.1, drop_path_rate: float = 0.05):
        super().__init__()
        self.ln_in = nn.LayerNorm(input_dim)
        self.fc_in = nn.Linear(input_dim, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        blocks = []
        n_blocks = len(mids)
        for i, m in enumerate(mids):
            dpr = drop_path_rate * float(i) / max(1, n_blocks - 1) if n_blocks > 1 else 0.0
            blocks.append(ResidualBottleneck(width, m, drop=drop, drop_path=dpr, layerscale=1.0))
        self.blocks = nn.ModuleList(blocks)

        self.ln_head = nn.LayerNorm(width)
        self.fc_out = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.ln_in(x)
        x = self.fc_in(x)
        x = self.act(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_head(x)
        x = self.fc_out(x)
        return x


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_params(input_dim: int, num_classes: int, width: int, mids: List[int]) -> int:
    # Manual estimate to ensure under budget, should match real count closely
    total = 0
    # ln_in
    total += 2 * input_dim
    # fc_in
    total += input_dim * width + width
    # blocks
    for m in mids:
        # ln
        total += 2 * width
        # fc1 and fc2
        total += width * m + m
        total += m * width + width
        # layerscale gamma
        total += width
    # ln_head
    total += 2 * width
    # fc_out
    total += width * num_classes + num_classes
    return total


def build_architecture(input_dim: int, num_classes: int, param_limit: int) -> Tuple[int, List[int]]:
    # Try descending widths, add as many blocks as fit under budget
    candidate_widths = [1152, 1088, 1024, 960, 928, 896, 864, 832, 800, 768, 736, 704, 672, 640]
    for width in candidate_widths:
        # Candidates for bottleneck dimensions
        mids_candidates = []
        mids_candidates.extend([max(16, (width * 3) // 4)])  # large
        mids_candidates.extend([max(16, width // 2)])
        mids_candidates.extend([max(16, (width * 3) // 8)])
        mids_candidates.extend([max(16, width // 4)])
        mids_candidates.extend([max(16, width // 8)])
        # Ensure uniqueness and sorted descending
        mids_candidates = sorted(list(set(mids_candidates)), reverse=True)
        mids: List[int] = []
        # Ensure at least 2 blocks if possible
        for m in mids_candidates:
            # round m to multiple of 16 for nicer shapes
            m = max(16, int(round(m / 16) * 16))
            trial_mids = mids + [m]
            est = estimate_params(input_dim, num_classes, width, trial_mids)
            if est <= param_limit - 2048:  # safety margin
                mids = trial_mids
            # Stop if adding more doesn't fit
            if len(mids) >= 4:
                break
        # Guarantee at least 2 blocks. If not, try smaller width
        if len(mids) >= 2:
            # If still have large margin, try to insert smaller extra blocks
            added = True
            while added:
                added = False
                for extra in [max(16, width // 8), max(16, width // 10), max(16, width // 12), max(16, width // 16)]:
                    extra = max(16, int(round(extra / 16) * 16))
                    if extra in mids:
                        continue
                    trial_mids = mids + [extra]
                    if estimate_params(input_dim, num_classes, width, trial_mids) <= param_limit - 1024:
                        mids = trial_mids
                        added = True
                        break
            return width, mids[:4]
    # Fallback small
    width = 640
    mids = [320, 160]
    return width, mids


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        self.collected = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
            self.shadow[name] = new_average

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.copy_(self.shadow[name])


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_mult: float = 0.1):
    def lr_lambda(current_step: int):
        if num_training_steps <= 0:
            return 1.0
        if current_step < num_warmup_steps and num_warmup_steps > 0:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_mult + (1.0 - min_lr_mult) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(1, total)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    indices = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b, lam


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2_500_000))

        width, mids = build_architecture(input_dim, num_classes, param_limit)
        model = MLPResNet(input_dim=input_dim, num_classes=num_classes, width=width, mids=mids, drop=0.15, drop_path_rate=0.05)
        model.to(device)

        # Ensure under parameter limit
        total_params = count_trainable_parameters(model)
        if total_params > param_limit:
            # Fallback to slimmer variant
            width = min(width, 896)
            mids = [width // 2, width // 4]
            model = MLPResNet(input_dim=input_dim, num_classes=num_classes, width=width, mids=mids, drop=0.15, drop_path_rate=0.05)
            model.to(device)

        # Optimizer and scheduler
        base_lr = 3e-3
        weight_decay = 0.05
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        max_epochs = 140
        steps_per_epoch = max(1, len(train_loader))
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = max(10, int(0.06 * total_steps))

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, min_lr_mult=0.08)

        mixup_alpha = 0.2
        label_smoothing = 0.0 if mixup_alpha > 0.0 else 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        ema = EMA(model, decay=0.995)
        ema_model = copy.deepcopy(model)
        ema_model.to(device)

        best_val_acc = -1.0
        best_state = None
        best_epoch = -1
        patience = 30
        no_improve = 0

        global_step = 0

        for epoch in range(max_epochs):
            model.train()
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs, targets = batch[0], batch[1]
                inputs = inputs.to(device, non_blocking=False).float()
                targets = targets.to(device, non_blocking=False).long()

                inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs_mixed)
                if lam < 1.0:
                    loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()
                ema.update(model)

                global_step += 1

            # Validation on EMA weights
            ema.copy_to(ema_model)
            ema_model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                    else:
                        inputs, targets = batch[0], batch[1]
                    inputs = inputs.to(device, non_blocking=False).float()
                    targets = targets.to(device, non_blocking=False).long()
                    logits = ema_model(inputs)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.numel()
            val_acc = val_correct / max(1, val_total)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(ema_model.state_dict())
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            ema.copy_to(model)

        model.eval()
        return model