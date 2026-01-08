import math
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def compute_mlp_param_count(input_dim: int, num_classes: int, width: int, num_hidden_layers: int) -> int:
    """
    Compute parameter count for ResidualMLP architecture used below.
    Architecture:
        fc_in: input_dim -> width
        ln_in: width
        (num_hidden_layers - 1) blocks of:
            fc: width -> width
            ln: width
        fc_out: width -> num_classes
    """
    params = 0
    # fc_in
    params += input_dim * width + width
    # ln_in
    params += 2 * width

    # hidden blocks
    for _ in range(num_hidden_layers - 1):
        # fc
        params += width * width + width
        # ln
        params += 2 * width

    # output layer
    params += width * num_classes + num_classes

    return params


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int,
        num_hidden_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_hidden_layers >= 1

        self.width = width
        self.num_hidden_layers = num_hidden_layers

        self.fc_in = nn.Linear(input_dim, width)
        self.ln_in = nn.LayerNorm(width)

        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(width, width))
            self.hidden_norms.append(nn.LayerNorm(width))

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer (no residual)
        h = self.fc_in(x)
        h = self.ln_in(h)
        h = self.activation(h)
        h = self.dropout(h)

        # Residual hidden layers
        for fc, ln in zip(self.hidden_layers, self.hidden_norms):
            residual = h
            h = fc(h)
            h = ln(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = h + residual

        out = self.fc_out(h)
        return out


def choose_width(
    input_dim: int,
    num_classes: int,
    param_limit: int,
    num_hidden_layers: int,
    min_width: int = 64,
    max_width: int = 2048,
) -> int:
    best_width = min_width
    for width in range(max_width, min_width - 1, -1):
        params = compute_mlp_param_count(input_dim, num_classes, width, num_hidden_layers)
        if params <= param_limit:
            best_width = width
            break
    return best_width


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.numel()

    avg_loss = total_loss / total_samples if (criterion is not None and total_samples > 0) else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


class Solution:
    def solve(self, train_loader, val_loader, metadata: Dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        torch.manual_seed(42)

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 5_000_000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        num_hidden_layers = 3

        width = choose_width(
            input_dim=input_dim,
            num_classes=num_classes,
            param_limit=param_limit,
            num_hidden_layers=num_hidden_layers,
            min_width=64,
            max_width=2048,
        )

        model = ResidualMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            num_hidden_layers=num_hidden_layers,
            dropout=0.2,
        ).to(device)

        # Final safety check to ensure we are within parameter limit
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # Reduce width until under limit (should rarely trigger if choose_width is correct)
            while width > 16:
                width -= 1
                model = ResidualMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    width=width,
                    num_hidden_layers=num_hidden_layers,
                    dropout=0.2,
                ).to(device)
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if param_count <= param_limit:
                    break

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        max_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        patience = 30
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            if val_loader is not None:
                val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                scheduler.step()

                if epochs_no_improve >= patience:
                    break
            else:
                scheduler.step()

        model.load_state_dict(best_state)
        model.to("cpu")
        return model