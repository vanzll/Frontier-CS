import math
import random
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.15,
        use_resblock: bool = True,
        input_norm: bool = True,
        out_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout
        self.use_resblock = use_resblock

        self.norm_in = nn.LayerNorm(input_dim) if input_norm else nn.Identity()

        layers = []
        norms_after = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            # add norm after each hidden, except last; the last will be handled by resblock and out_norm
            if i < len(hidden_dims) - 1:
                norms_after.append(nn.LayerNorm(h))
            else:
                norms_after.append(nn.Identity())
            prev = h

        self.features = nn.ModuleList(layers)
        self.norms_after = nn.ModuleList(norms_after)

        last_dim = hidden_dims[-1]
        self.resblock = ResidualBlock(last_dim, dropout=dropout) if use_resblock else nn.Identity()
        self.pre_out_norm = nn.LayerNorm(last_dim) if out_norm else nn.Identity()
        self.dropout_out = nn.Dropout(dropout)
        self.classifier = nn.Linear(last_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.norm_in(x)
        feat_layers = self.features
        norms = self.norms_after
        # iterate per-block
        idx = 0
        for i in range(len(self.hidden_dims)):
            x = feat_layers[idx](x)
            idx += 1
            x = feat_layers[idx](x)
            idx += 1
            x = feat_layers[idx](x)
            idx += 1
            x = norms[i](x)

        x = self.resblock(x)
        x = self.pre_out_norm(x)
        x = self.dropout_out(x)
        x = self.classifier(x)
        return x


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + (1.0 - d) * msd[k])

    def copy_to_model(self, model: nn.Module):
        model.load_state_dict(self.ema.state_dict(), strict=True)


class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = patience
        self.counter = 0
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_state = None
        self.mode = mode

    def step(self, metric: float, model: nn.Module):
        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
            self.best_state = deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def get_best_state(self):
        return self.best_state


def soft_cross_entropy(logits, target_soft):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_soft * log_probs).sum(dim=1).mean()
    return loss


def mixup_data(x, y, num_classes, alpha=0.2, device="cpu"):
    if alpha <= 0.0:
        return x, F.one_hot(y, num_classes=num_classes).float()
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y1 = F.one_hot(y, num_classes=num_classes).float()
    y2 = F.one_hot(y[index], num_classes=num_classes).float()
    mixed_y = lam * y1 + (1.0 - lam) * y2
    return mixed_x, mixed_y


def evaluate_accuracy(model: nn.Module, loader, device: str):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    acc = correct / max(1, total)
    return acc


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Solution:
    def _build_model_under_limit(self, input_dim: int, num_classes: int, limit: int) -> nn.Module:
        # Candidate configurations (descending capacity)
        width1_list = [768, 640, 576, 512, 448, 384]
        width2_list = [512, 448, 384, 320, 256]
        width3_list = [256, 224, 192, 160, 128]
        dropout = 0.15

        for use_res in [True, False]:
            for h1 in width1_list:
                for h2 in [w for w in width2_list if w <= h1]:
                    for h3 in [w for w in width3_list if w <= h2]:
                        model = MLPNet(
                            input_dim=input_dim,
                            hidden_dims=[h1, h2, h3],
                            num_classes=num_classes,
                            dropout=dropout,
                            use_resblock=use_res,
                            input_norm=True,
                            out_norm=True,
                        )
                        params = count_trainable_params(model)
                        if params <= limit:
                            return model
        # Fallback very small model
        model = MLPNet(
            input_dim=input_dim,
            hidden_dims=[min(384, input_dim * 2), 256, 128],
            num_classes=num_classes,
            dropout=0.1,
            use_resblock=False,
            input_norm=True,
            out_norm=True,
        )
        return model

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        set_seed(42)
        device = metadata.get("device", "cpu") if metadata is not None else "cpu"
        input_dim = int(metadata.get("input_dim", 384)) if metadata is not None else 384
        num_classes = int(metadata.get("num_classes", 128)) if metadata is not None else 128
        param_limit = int(metadata.get("param_limit", 1_000_000)) if metadata is not None else 1_000_000

        model = self._build_model_under_limit(input_dim, num_classes, param_limit)
        model.to(device)

        # Ensure parameter constraint
        param_count = count_trainable_params(model)
        if param_count > param_limit:
            # As a final fallback, shrink to a safe small model
            model = MLPNet(
                input_dim=input_dim,
                hidden_dims=[512, 256, 128],
                num_classes=num_classes,
                dropout=0.1,
                use_resblock=False,
                input_norm=True,
                out_norm=True,
            ).to(device)
            param_count = count_trainable_params(model)

        # Optimizer and scheduler
        base_lr = 3e-3
        weight_decay = 5e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        max_epochs = 140
        warmup_epochs = 10

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            # Cosine decay
            t = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Loss setups
        label_smoothing = 0.05
        ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # EMA
        ema = EMA(model, decay=0.999)

        # Early stopping
        early_stop = EarlyStopping(patience=25, mode="max")

        # Mixup settings
        use_mixup = True
        mixup_alpha = 0.2
        mixup_prob = 0.6

        # Optional small Gaussian noise for robustness
        noise_std = 0.01

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.long)

                # Input noise
                if noise_std > 0.0:
                    noise = torch.randn_like(inputs) * noise_std
                    inputs = inputs + noise

                # Decide mixup
                if use_mixup and random.random() < mixup_prob:
                    x_mix, y_soft = mixup_data(inputs, targets, num_classes, alpha=mixup_alpha, device=device)
                    outputs = model(x_mix)
                    loss = soft_cross_entropy(outputs, y_soft)
                else:
                    outputs = model(inputs)
                    loss = ce_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(model)

            scheduler.step()

            # Validate on EMA model for stability
            ema_model = ema.ema
            ema_model.to(device)
            val_acc = evaluate_accuracy(ema_model, val_loader, device)

            stop = early_stop.step(val_acc, ema_model)
            if stop:
                break

        # Load best EMA state
        best_state = early_stop.get_best_state()
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        else:
            # Fall back to EMA weights if early stop didn't store any
            ema.copy_to_model(model)

        model.eval()
        return model