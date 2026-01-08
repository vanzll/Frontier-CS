import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)


class ResidualBlockTwoLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h = self.fc1(self.act(self.ln1(x)))
        h = self.drop(h)
        h = self.fc2(self.act(self.ln2(h)))
        h = self.drop(h)
        return x + self.scale * h


class ResidualBlockOneLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h = self.fc(self.act(self.ln(x)))
        h = self.drop(h)
        return x + self.scale * h


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, num_two_layer_blocks: int, num_one_layer_blocks: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, width)
        blocks = []
        for _ in range(num_two_layer_blocks):
            blocks.append(ResidualBlockTwoLayer(width, dropout=dropout))
        for _ in range(num_one_layer_blocks):
            blocks.append(ResidualBlockOneLayer(width, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.final_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

        # Kaiming initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        x = self.head(x)
        return x


def build_model_within_budget(input_dim: int, num_classes: int, param_limit: int, target_layers: tuple, dropout: float):
    # target_layers: (num_two_layer_blocks, num_one_layer_blocks)
    num_two, num_one = target_layers

    # Binary search over width to fit under parameter limit
    low, high = 128, 2048
    best_width = 512
    safety_margin = 4096  # keep some slack for safety
    device = 'cpu'

    while low <= high:
        mid = (low + high) // 2
        model = MLPNet(input_dim, num_classes, mid, num_two, num_one, dropout)
        params = count_parameters(model)
        if params <= param_limit - safety_margin:
            best_width = mid
            low = mid + 1
        else:
            high = mid - 1

    model = MLPNet(input_dim, num_classes, best_width, num_two, num_one, dropout)
    return model


def evaluate_accuracy(model: nn.Module, data_loader, device: str):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=False)
            targets = targets.to(device=device, non_blocking=False)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / total if total > 0 else 0.0


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        set_seed(42)

        input_dim = metadata.get("input_dim", 384) if metadata else 384
        num_classes = metadata.get("num_classes", 128) if metadata else 128
        param_limit = metadata.get("param_limit", 5_000_000) if metadata else 5_000_000
        device = metadata.get("device", "cpu") if metadata else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        # Build a deep residual MLP near budget: 3 two-layer blocks (6 hidden linear layers)
        # Dropout tuned small to avoid underfitting on synthetic data
        dropout = 0.08
        model = build_model_within_budget(input_dim, num_classes, param_limit, target_layers=(3, 0), dropout=dropout)
        model.to(device)

        # Ensure budget
        assert count_parameters(model) <= param_limit, "Parameter limit exceeded"

        # Optimizer with decoupled weight decay, excluding biases and norms
        weight_decay = 0.03
        no_decay, decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "ln" in name.lower() or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=0.003,
            betas=(0.9, 0.999),
        )

        # Scheduling: OneCycleLR over steps
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            steps_per_epoch = max(1, metadata.get("train_samples", 2048) // getattr(train_loader, "batch_size", 64))
        small_steps = steps_per_epoch <= 20

        # Epochs heuristic: more epochs for fewer steps per epoch
        if small_steps:
            epochs = 280
        elif steps_per_epoch <= 40:
            epochs = 220
        else:
            epochs = 170

        total_steps = steps_per_epoch * epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.003,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=6.0,
            final_div_factor=100.0,
        )

        # Loss with mild label smoothing
        label_smoothing = 0.05
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Exponential Moving Average of weights
        ema = EMA(model, decay=0.995)

        # Early stopping
        best_val_acc = -1.0
        best_state_ema = None
        patience = 36
        epochs_no_improve = 0
        grad_clip = 1.0

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=False)
                targets = targets.to(device=device, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                ema.update(model)

            # Validation
            if val_loader is not None:
                # Evaluate EMA weights
                current_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                val_acc = evaluate_accuracy(model, val_loader, device)
                # Restore current weights
                model.load_state_dict(current_state)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state_ema = ema.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Load best EMA weights if available, else last EMA
        if best_state_ema is not None:
            ema.load_state_dict(best_state_ema)
        ema.copy_to(model)

        return model