import math
import random
import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

        # Initialize last layer to zero for stable residual learning
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class MLPResNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int = 128, num_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout=dropout) for _ in range(num_blocks)])
        self.norm_out = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(width, num_classes)

        # Kaiming init for input and output layers
        nn.init.kaiming_uniform_(self.fc_in.weight, a=math.sqrt(5))
        if self.fc_in.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_in.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc_in.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.fc_out.weight, a=math.sqrt(5))
        if self.fc_out.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_out.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc_out.bias, -bound, bound)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.blocks(x)
        x = self.norm_out(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.997, device: str = "cpu"):
        self.ema = copy.deepcopy(model).to(device)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def to(self, device):
        self.ema.to(device)
        return self


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_accuracy(model: nn.Module, data_loader, device: str) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_total += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    acc = correct / max(1, total)
    loss_avg = loss_total / max(1, total)
    return acc, loss_avg


def params_formula(input_dim: int, num_classes: int, width: int, num_blocks: int) -> int:
    # Input layer
    p = input_dim * width + width
    # Residual blocks: per block: LN(2*width) + Linear(width,width)+bias + Linear(width,width)+bias = 2*width^2 + 4*width
    p += num_blocks * (2 * width * width + 4 * width)
    # Output LN
    p += 2 * width
    # Output layer
    p += width * num_classes + num_classes
    return p


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        seed_everything(42 + (metadata["train_samples"] if metadata and "train_samples" in metadata else 0))

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200_000)
        device = metadata.get("device", "cpu")

        # Search for model configuration near the parameter limit
        best_cfg = None  # (width, num_blocks)
        # Prefer deeper models first
        for num_blocks in [4, 3, 2]:
            # Start from a reasonable upper bound on width
            max_start = min(384, 256)
            chosen = None
            for width in range(max_start, 31, -1):
                p = params_formula(input_dim, num_classes, width, num_blocks)
                if p <= param_limit:
                    chosen = (width, num_blocks, p)
                    break
            if chosen is not None:
                best_cfg = (chosen[0], chosen[1])
                break
        if best_cfg is None:
            # Fallback minimal small network if the limit is extremely small
            best_cfg = (64, 1)

        width, num_blocks = best_cfg
        model = MLPResNet(input_dim=input_dim, num_classes=num_classes, width=width, num_blocks=num_blocks, dropout=0.1).to(device)

        # Hard sanity check for parameter limit
        if count_params(model) > param_limit:
            # Reduce width until within limit
            for w in range(width, 31, -1):
                temp = MLPResNet(input_dim=input_dim, num_classes=num_classes, width=w, num_blocks=num_blocks, dropout=0.1).to(device)
                if count_params(temp) <= param_limit:
                    model = temp
                    width = w
                    break

        # Optimizer and scheduler
        base_lr = 3e-3
        weight_decay = 1e-2
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        # Training settings
        total_epochs = 200 if metadata and metadata.get("train_samples", 2048) <= 10000 else 120
        steps_per_epoch = max(1, len(train_loader))
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = max(10, int(0.1 * total_steps))
        scheduler = cosine_warmup_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # EMA model
        ema_decay = 0.997
        ema = ModelEMA(model, decay=ema_decay, device=device)

        # Mixup configuration
        mixup_alpha = 0.2
        use_mixup = True
        mixup_end_epoch = int(0.6 * total_epochs)

        def mixup_data(x, y, alpha=1.0):
            if alpha > 0.0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        # Training loop
        best_val_acc = 0.0
        best_state = copy.deepcopy(ema.ema.state_dict())
        patience = 30
        epochs_no_improve = 0
        global_step = 0

        for epoch in range(total_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)

                if use_mixup and epoch < mixup_end_epoch:
                    mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()
                ema.update(model)

                global_step += 1

            # Validation using EMA model
            val_acc, _ = evaluate_accuracy(ema.ema, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(ema.ema.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        # Load best EMA weights and return the EMA model
        ema.ema.load_state_dict(best_state)
        ema_model = ema.ema
        # Ensure parameter limit not exceeded
        assert count_params(ema_model) <= param_limit
        return ema_model