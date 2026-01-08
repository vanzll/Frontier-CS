import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class LinearModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
    # Try 2-hidden-layer MLP with BatchNorm and Dropout
    hidden_candidates = [512, 384, 320, 256, 224, 192, 160, 128, 96, 64, 48, 32]
    for h in hidden_candidates:
        model = MLPNet(input_dim, num_classes, hidden_dim=h, dropout=0.2)
        if count_trainable_params(model) <= param_limit:
            return model

    # Fallback: simple 1-hidden-layer MLP
    hidden_candidates_simple = [512, 384, 320, 256, 224, 192, 160, 128, 96, 64, 48, 32, 24, 16]
    for h in hidden_candidates_simple:
        model = SimpleMLP(input_dim, num_classes, hidden_dim=h)
        if count_trainable_params(model) <= param_limit:
            return model

    # Last resort: linear classifier
    model = LinearModel(input_dim, num_classes)
    return model


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 200_000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")

        torch.manual_seed(42)

        model = build_model(input_dim, num_classes, param_limit)
        model.to(device)

        # Ensure hard parameter constraint
        if count_trainable_params(model) > param_limit:
            # As a safety fallback, use smallest linear model
            model = LinearModel(input_dim, num_classes).to(device)

        # Training setup
        lr = 1e-3
        weight_decay = 3e-4
        max_epochs = 200
        patience = 25

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

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
                total_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += batch_size

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

            if val_total > 0:
                val_acc = val_correct / val_total
            else:
                val_acc = 0.0

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        model.load_state_dict(best_state)
        model.to(device)
        return model