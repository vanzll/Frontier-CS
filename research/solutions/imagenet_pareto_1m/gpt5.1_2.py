import math
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.fc3 = nn.Linear(width, width)
        self.bn3 = nn.BatchNorm1d(width)
        self.fc_out = nn.Linear(width, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        residual = x
        y = self.fc2(x)
        y = self.bn2(y)
        y = F.gelu(y)
        y = self.dropout(y)
        x = residual + y

        residual = x
        y = self.fc3(x)
        y = self.bn3(y)
        y = F.gelu(y)
        y = self.dropout(y)
        x = residual + y

        logits = self.fc_out(x)
        return logits


class Solution:
    def _build_model(self, input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
        # Analytic width selection for 3-layer residual MLP with BatchNorm
        # Param formula: P(w) = 2w^2 + w*(input_dim + num_classes + 9) + num_classes
        A = 2.0
        B = float(input_dim + num_classes + 9)
        C = float(num_classes - param_limit)
        disc = B * B - 4 * A * C
        if disc < 0:
            disc = 0.0
        w_max = int((-B + math.sqrt(disc)) / (2 * A))
        if w_max < 32:
            w_max = 32

        # Safety adjustment: verify and, if needed, decrease width until under limit
        width = w_max
        while width > 16:
            model = MLPNet(input_dim, num_classes, width)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if param_count <= param_limit:
                return model
            width -= 1

        # Fallback small model (should never be needed with given limits)
        return MLPNet(input_dim, num_classes, 128)

    def solve(self, train_loader, val_loader, metadata: Dict[str, Any] = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        # Reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 1_000_000))

        model = self._build_model(input_dim, num_classes, param_limit)
        model.to(device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-3,
            weight_decay=1e-4,
        )

        num_epochs = 300
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=5e-5,
        )

        best_val_acc = 0.0
        best_state = None

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()

            if total > 0:
                val_acc = correct / total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()
        return model