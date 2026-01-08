import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ln(x)
        out = self.fc(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return x + out


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.block1 = ResidualBlock(hidden_dim, dropout)
        self.block2 = ResidualBlock(hidden_dim, dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc_out(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    if loader is None:
        return 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 5_000_000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        # Reproducibility
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        # Choose hidden dimension close to limit while staying under param_limit
        target_hidden = 1456  # designed for ~5M limit
        hidden_dim = target_hidden
        dropout = 0.1

        # Adjust hidden_dim downward if needed to satisfy param_limit
        while True:
            model = MLPNet(input_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
            params = count_params(model)
            if params <= param_limit or hidden_dim <= 32:
                break
            hidden_dim = max(32, hidden_dim - 64)

        model = MLPNet(input_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
        model.to(device)

        # Double-check parameter constraint
        if count_params(model) > param_limit:
            # Fallback tiny model to ensure constraint is not violated
            hidden_dim = 64
            model = MLPNet(input_dim, num_classes, hidden_dim=hidden_dim, dropout=0.1)
            model.to(device)

        # Training setup
        train_samples = metadata.get("train_samples", None)
        if train_samples is not None and train_samples <= 4096:
            max_epochs = 220
        else:
            max_epochs = 150

        patience = 35

        # Label smoothing for better generalization
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_val_acc = 0.0
        best_state = deepcopy(model.state_dict())
        best_epoch = 0

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

            scheduler.step()

            _, val_acc = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch

            if epoch - best_epoch >= patience:
                break

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        return model