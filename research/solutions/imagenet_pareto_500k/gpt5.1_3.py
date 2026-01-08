import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer
        x = self.input_layer(x)
        x = self.norms[0](x)
        x = self.activation(x)
        x = self.dropout(x)

        residual = x
        # Subsequent layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.norms[i + 1](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + residual
            residual = x

        x = self.output_layer(x)
        return x


class Solution:
    def _build_model(self, input_dim: int, num_classes: int, param_limit: int, device: torch.device) -> nn.Module:
        # Try to maximize hidden_dim under parameter budget with a 3-layer residual MLP
        max_start = max(input_dim, num_classes * 2, 64)
        hidden_dim = min(512, max_start)
        best_model = None

        while hidden_dim >= 32:
            model = ResidualMLP(input_dim, num_classes, hidden_dim=hidden_dim, num_layers=3, dropout=0.2)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if param_count <= param_limit:
                best_model = model
                break
            hidden_dim -= 1

        if best_model is None:
            # Fallback to simple linear classifier (will be well under any sensible limit)
            best_model = nn.Linear(input_dim, num_classes)

        best_model.to(device)
        return best_model

    def _evaluate(self, model: nn.Module, data_loader, criterion, device: torch.device):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 500000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        train_samples = metadata.get("train_samples", None)
        if train_samples is None:
            try:
                train_samples = len(train_loader.dataset)
            except Exception:
                train_samples = 2048

        # Build model under parameter constraint
        model = self._build_model(input_dim, num_classes, param_limit, device)

        # Verify parameter constraint (safety check)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # As a very safe fallback, use a single hidden-layer MLP with conservative size
            hidden_dim = min(256, input_dim)
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            ).to(device)

        # Training hyperparameters
        if train_samples <= 4000:
            max_epochs = 220
            patience = 35
            base_lr = 2.5e-3
        elif train_samples <= 20000:
            max_epochs = 120
            patience = 20
            base_lr = 2e-3
        else:
            max_epochs = 60
            patience = 10
            base_lr = 1.5e-3

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=base_lr * 0.1
        )

        best_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_no_improve = 0

        # Training loop with early stopping
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            if scheduler is not None:
                scheduler.step()

            # Validation and early stopping
            if val_loader is not None:
                _, val_acc = self._evaluate(model, val_loader, criterion, device)

                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_state = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

        # Load best validation model (if any improvement was tracked)
        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()
        return model