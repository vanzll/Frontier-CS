import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def estimate_param_count(input_dim: int, num_classes: int, hidden_dim: int, num_layers: int) -> int:
    # Linear + BN layers
    total = 0
    in_dim = input_dim
    for _ in range(num_layers):
        # Linear layer
        total += in_dim * hidden_dim + hidden_dim
        in_dim = hidden_dim
    # Output layer
    total += hidden_dim * num_classes + num_classes
    # BatchNorm layers for each hidden layer: weight + bias = 2 * hidden_dim
    total += num_layers * 2 * hidden_dim
    return total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, data_loader, criterion, device: torch.device):
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
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        set_seed(42)

        if metadata is None:
            metadata = {}

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500_000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        num_layers = 3
        max_hidden_dim = min(input_dim * 2, 1024)

        best_hidden_dim = None
        for h in range(max_hidden_dim, 1, -1):
            if estimate_param_count(input_dim, num_classes, h, num_layers) <= param_limit:
                best_hidden_dim = h
                break

        if best_hidden_dim is None:
            # Fallback to a simple linear classifier if extremely tight budget
            model = nn.Linear(input_dim, num_classes)
            model.to(device)
            return model

        hidden_dim = best_hidden_dim

        model = MLPNet(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
        model.to(device)

        # Safety check in case of any mismatch
        actual_params = count_parameters(model)
        while actual_params > param_limit and hidden_dim > 1:
            hidden_dim -= 1
            model = MLPNet(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1)
            model.to(device)
            actual_params = count_parameters(model)

        # Training setup
        lr = 1e-3
        weight_decay = 1e-3
        max_epochs = 200
        patience = 20

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_state = None
        best_val_acc = -1.0
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                running_correct += (preds == targets).sum().item()
                running_total += batch_size

            if scheduler is not None:
                scheduler.step()

            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()
        return model