import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)

        residual = x

        out = self.fc2(x)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop2(out)

        out = out + residual
        out = self.fc3(out)
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_with_budget(input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
    # First, check simple linear classifier in case budget is extremely low
    linear_params = input_dim * num_classes + num_classes
    if linear_params > param_limit:
        # Cannot satisfy constraint realistically; still return minimal model
        return nn.Linear(input_dim, num_classes)

    # Search for maximum hidden_dim that fits in the budget for ResidualMLP
    # Start from a reasonably high upper bound
    max_hidden_search = min(1024, param_limit)
    best_hidden = None
    for h in range(max_hidden_search, 0, -1):
        tmp_model = ResidualMLP(input_dim, num_classes, h, dropout=0.1)
        if count_params(tmp_model) <= param_limit:
            best_hidden = h
            model = tmp_model
            break

    if best_hidden is None:
        # Fallback to linear classifier if even the smallest residual model does not fit
        model = nn.Linear(input_dim, num_classes)

    return model


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y, y, 1.0
    batch_size = x.size(0)
    if batch_size == 1:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        torch.manual_seed(42)
        np.random.seed(42)

        model = build_model_with_budget(input_dim, num_classes, param_limit)
        model.to(device)

        # Ensure we respect the parameter budget
        if count_params(model) > param_limit:
            # As a last resort, fall back to the simplest linear model
            model = nn.Linear(input_dim, num_classes).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        max_epochs = 200
        patience = 30
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=3e-5
        )

        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0
        best_epoch = 0

        def evaluate(model_eval, loader):
            model_eval.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model_eval(inputs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
            return correct / total if total > 0 else 0.0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)

                optimizer.zero_grad()
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            val_acc = evaluate(model, val_loader)
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if epoch - best_epoch >= patience:
                break

        model.load_state_dict(best_state)
        model.to("cpu")
        return model