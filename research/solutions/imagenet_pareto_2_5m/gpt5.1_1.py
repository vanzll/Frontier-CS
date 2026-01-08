import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        return out


class ResMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, depth: int, dropout: float = 0.1):
        super().__init__()
        assert depth >= 2
        self.input = nn.Linear(input_dim, width)
        self.input_bn = nn.BatchNorm1d(width)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dropout=dropout) for _ in range(depth - 1)]
        )
        self.output = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.input_bn(x)
        x = F.relu(x, inplace=True)
        for block in self.blocks:
            residual = x
            out = block(x)
            x = residual + out
        x = self.output(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model: nn.Module, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
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
    def _build_best_model(self, input_dim, num_classes, param_limit):
        best_cfg = None
        best_params = 0

        # Search over reasonable depths and widths
        for depth in range(4, 9):  # depths 4..8
            for width in range(1024, 127, -32):  # widths 1024..128
                model = ResMLP(input_dim, num_classes, width, depth, dropout=0.1)
                params = count_parameters(model)
                if params <= param_limit and params > best_params:
                    best_params = params
                    best_cfg = (depth, width)

        if best_cfg is None:
            # Fallback small model if param_limit is extremely tiny
            width = 128
            depth = 3
            while True:
                model = ResMLP(input_dim, num_classes, width, depth, dropout=0.1)
                params = count_parameters(model)
                if params <= param_limit or width <= 32:
                    return model
                width = max(32, width // 2)

        depth, width = best_cfg
        model = ResMLP(input_dim, num_classes, width, depth, dropout=0.1)
        return model

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            raise ValueError("metadata must be provided")

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2500000))

        meta_device = metadata.get("device", "cpu")
        if torch.cuda.is_available() and meta_device != "cpu":
            device = torch.device(meta_device)
        else:
            device = torch.device("cpu")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = self._build_best_model(input_dim, num_classes, param_limit)
        assert count_parameters(model) <= param_limit, "Model exceeds parameter limit"
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=False,
        )

        max_epochs = 200
        patience = 25  # early stopping on val accuracy
        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0

        has_val = val_loader is not None

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            if has_val:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                scheduler.step(val_acc)

                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break
            else:
                # No validation loader provided, just step scheduler with dummy metric
                scheduler.step(0.0)

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()
        return model