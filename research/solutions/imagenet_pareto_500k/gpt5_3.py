import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResidualBlock1LinPreLN(nn.Module):
    def __init__(self, dim, dropout=0.1, activation='gelu'):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim, bias=True)
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        # ReZero-style learnable residual scaling
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = self.ln(x)
        z = self.fc(z)
        z = self.act(z)
        z = self.dropout(z)
        return x + self.gate * z


class ResMLP(nn.Module):
    def __init__(self, input_dim, num_classes, width, n_blocks=3, dropout=0.1, activation='gelu'):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.num_classes = num_classes
        self.n_blocks = n_blocks

        self.proj = None
        if input_dim != width:
            self.proj = nn.Linear(input_dim, width)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock1LinPreLN(width, dropout=dropout, activation=activation))
        self.blocks = nn.ModuleList(blocks)

        self.pre_head_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pre_head_ln(x)
        x = self.head(x)
        return x


class EMA:
    def __init__(self, model: nn.Module, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert name in self.shadow
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
                self.shadow[name] = new_avg.clone()

    def store(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()

    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name].data)
        self.backup = {}


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_param_count_estimate(input_dim, num_classes, width, n_blocks=3, use_proj=True):
    # Estimate parameters: projection, blocks (1 linear per block + LayerNorm), head + pre_head ln
    total = 0
    if use_proj and (input_dim != width):
        total += input_dim * width + width  # proj weight + bias
    # Each block: Linear (width*width + width) + LN (2*width) + gate (1)
    for _ in range(n_blocks):
        total += width * width + width  # linear + bias
        total += 2 * width  # layernorm gamma+beta
        total += 1  # gate
    # Pre head layernorm + head linear
    total += 2 * width  # pre_head ln
    total += width * num_classes + num_classes  # head weight + bias
    return total


def best_width_under_budget(input_dim, num_classes, param_limit, n_blocks=3):
    width = min(input_dim, 1024)
    while width > 8:
        est = compute_param_count_estimate(input_dim, num_classes, width, n_blocks=n_blocks, use_proj=True)
        if est <= param_limit:
            return width
        width -= 1
    return max(8, width)


def evaluate_accuracy(model: nn.Module, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        set_seed(42)
        device = torch.device(metadata.get("device", "cpu") if metadata else "cpu")
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500_000))

        n_blocks = 3
        width = best_width_under_budget(input_dim, num_classes, param_limit, n_blocks=n_blocks)

        model = ResMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            n_blocks=n_blocks,
            dropout=0.1,
            activation='gelu',
        ).to(device)

        # Ensure parameter budget
        params = count_trainable_params(model)
        if params > param_limit:
            # Fallback to smaller width if somehow exceeded due to estimation mismatch
            while params > param_limit and width > 8:
                width -= 1
                model = ResMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    width=width,
                    n_blocks=n_blocks,
                    dropout=0.1,
                    activation='gelu',
                ).to(device)
                params = count_trainable_params(model)
        # Training setup
        lr = 3e-3
        weight_decay = 0.02
        epochs = 160
        patience = 30

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        ema = EMA(model, decay=0.995)

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(model)

            scheduler.step()

            # Evaluate with EMA weights
            ema.store(model)
            ema.copy_to(model)
            val_acc = evaluate_accuracy(model, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            ema.restore(model)

            if epochs_no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            # As fallback, use EMA weights
            ema.copy_to(model)

        model.to('cpu')
        model.eval()
        return model