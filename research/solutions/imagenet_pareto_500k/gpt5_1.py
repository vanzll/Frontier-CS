import math
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class ResidualFFNBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.2, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return x + residual


class MLPResidualNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, drop: float = 0.2, drop_path: float = 0.05,
                 final_norm: bool = True):
        super().__init__()
        self.block = ResidualFFNBlock(input_dim, hidden_dim, drop=drop, drop_path=drop_path)
        self.final_norm = nn.LayerNorm(input_dim) if final_norm else nn.Identity()
        self.head = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.final_norm(x)
        x = self.head(x)
        return x


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def build_cosine_warmup_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


class Solution:
    def solve(self, train_loader, val_loader, metadata: Dict = None) -> torch.nn.Module:
        set_seed(42)

        device = torch.device(metadata.get("device", "cpu") if metadata is not None else "cpu")
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500_000))

        # Determine the largest hidden_dim for a single residual block under the parameter budget
        # Total params formula: P = 769*h + 51200 for final_norm + classifier included
        # Derivation:
        # block: 2*(input_dim*hidden_dim) + hidden_dim + input_dim + 2*input_dim (LayerNorm params)
        #      = 2*d*h + h + d + 2d = (2d+1)h + 3d = (769)h + 1152 for d=384
        # classifier: d*num_classes + num_classes = 384*128 + 128 = 49280
        # final_norm: 2*d = 768
        # sum: (769)h + 1152 + 49280 + 768 = 769h + 51200
        d = input_dim
        # Generalized formula in case d/num_classes differ (but here they match metadata)
        def total_params_for_hidden(h: int) -> int:
            block_params = 2 * d * h + h + d + 2 * d  # fc1 + fc2 + biases + layernorm (weights+bias)
            head_params = d * num_classes + num_classes
            final_norm_params = 2 * d
            return block_params + head_params + final_norm_params

        # Find the maximum hidden_dim not exceeding param_limit
        max_h = max(16, min(1024, (param_limit - (d * num_classes + num_classes) - (3 * d)) // (2 * d + 1)))
        # Adjust down to ensure constraint; also prefer multiples of 8
        best_h = None
        for h in range(max_h, 15, -1):
            if h % 8 != 0:  # prefer multiple of 8 for efficiency
                continue
            if total_params_for_hidden(h) <= param_limit:
                best_h = h
                break
        if best_h is None:
            # Fallback: choose smallest viable hidden dim
            for h in range(16, max_h + 1):
                if total_params_for_hidden(h) <= param_limit:
                    best_h = h
                    break
        # Default to 576 if within limit, else best_h
        proposed_h = 576
        if total_params_for_hidden(proposed_h) <= param_limit:
            hidden_dim = proposed_h
        else:
            hidden_dim = best_h if best_h is not None else max(32, min(384, d))

        model = MLPResidualNet(
            input_dim=d,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            drop=0.2,
            drop_path=0.05,
            final_norm=True,
        ).to(device)

        # Ensure parameter limit
        assert count_trainable_params(model) <= param_limit, f"Model has {count_trainable_params(model)} params, exceeds limit {param_limit}"

        # Optimizer and scheduler
        base_lr = 3e-3
        weight_decay = 0.06
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # Training configuration
        epochs = 120
        steps_per_epoch = max(1, len(train_loader))
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)
        scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)
        grad_clip_norm = 1.0

        ema = EMA(model, decay=0.995)

        best_val_acc = 0.0
        best_state = None
        patience = 25
        no_improve = 0

        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()
                ema.update(model)

            # Validation with EMA weights
            ema.apply_shadow(model)
            val_acc = evaluate(model, val_loader, device)
            ema.restore(model)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

            # Time guard: stop if training takes too long (safety)
            if time.time() - start_time > 1800:  # 30 minutes safety
                break

        # Load best weights (prefer EMA weights around best epoch)
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final EMA smoothing for returned model
        ema.apply_shadow(model)
        return model