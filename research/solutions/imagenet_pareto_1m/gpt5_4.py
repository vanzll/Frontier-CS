import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim, dropout=0.1, norm_type='ln'):
        super().__init__()
        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(bottleneck_dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.fc1 = nn.Linear(dim, bottleneck_dim, bias=False)
        self.fc2 = nn.Linear(bottleneck_dim, dim, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm1(x)
        y = self.act(y)
        y = self.fc1(y)
        y = self.norm2(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes, width=768, bottleneck=192, num_blocks=2, dropout=0.1, norm_type='ln'):
        super().__init__()
        self.input = nn.Linear(input_dim, width, bias=False)
        if norm_type == 'bn':
            self.in_norm = nn.BatchNorm1d(width)
        else:
            self.in_norm = nn.LayerNorm(width)
        self.act = nn.GELU()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBottleneckBlock(width, bottleneck, dropout=dropout, norm_type=norm_type))
        self.blocks = nn.ModuleList(blocks)
        self.final_drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(width, num_classes, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.input(x)
        x = self.in_norm(x)
        x = self.act(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_drop(x)
        return self.fc_out(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Solution:
    def _build_model_under_budget(self, input_dim, num_classes, param_limit, prefer_norm='ln'):
        # Try a list of widths; pick largest under budget
        candidates = [768, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384]
        num_blocks = 2
        for width in candidates:
            bottleneck = max(64, width // 4)
            model = MLPNet(input_dim, num_classes, width=width, bottleneck=bottleneck, num_blocks=num_blocks, dropout=0.1, norm_type=prefer_norm)
            if count_trainable_params(model) <= param_limit:
                return model
        # Fallback simple model if even smallest exceeds
        hidden = 384
        model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        return model

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 1_000_000))

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Prefer LayerNorm for small batches stability
        prefer_norm = 'ln'

        model = self._build_model_under_budget(input_dim, num_classes, param_limit, prefer_norm=prefer_norm)
        model.to(device)
        total_params = count_trainable_params(model)
        if total_params > param_limit:
            # Last-resort: shrink to baseline 2-layer MLP that surely fits
            hidden = 512
            model = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            ).to(device)

        steps_per_epoch = max(1, len(train_loader))
        # Epoch scheduling based on steps to keep runtime reasonable
        if steps_per_epoch >= 64:
            epochs = 45
        elif steps_per_epoch >= 32:
            epochs = 60
        elif steps_per_epoch >= 16:
            epochs = 70
        else:
            epochs = 80

        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        base_lr = 3e-3
        weight_decay = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        use_onecycle = True
        if use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=base_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.15,
                div_factor=10.0,
                final_div_factor=100.0,
                anneal_strategy='cos'
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = -1.0
        best_state = None
        no_improve = 0
        patience = 15

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if use_onecycle:
                    scheduler.step()

            if val_loader is not None:
                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        if inputs.dim() > 2:
                            inputs = inputs.view(inputs.size(0), -1)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * targets.size(0)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.numel()
                val_acc = correct / max(1, total)
                if val_acc > best_val_acc + 1e-5:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break
            else:
                if not use_onecycle:
                    scheduler.step()

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        return model