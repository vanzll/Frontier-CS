import math
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class Solution:
    def _build_model(self, input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
        # Start from sqrt of param_limit as a heuristic and search downward
        hidden_dim = int(math.sqrt(max(param_limit, 1)))
        hidden_dim = min(max(hidden_dim, 128), 4096)
        step = 8 if hidden_dim > 512 else 4

        best_model = None
        while hidden_dim >= 128:
            model = MLPClassifier(input_dim, hidden_dim, num_classes, dropout=0.2)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if param_count <= param_limit:
                best_model = model
                break
            hidden_dim -= step

        if best_model is None:
            # Fallback to a small model if param_limit is extremely tight
            hidden_dim = max(32, min(128, input_dim))
            best_model = MLPClassifier(input_dim, hidden_dim, num_classes, dropout=0.2)

        return best_model

    def _evaluate(self, model: nn.Module, data_loader, device: torch.device, criterion=None):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        if total == 0:
            return 0.0, 0.0
        avg_loss = total_loss / total if criterion is not None else 0.0
        acc = correct / total
        return avg_loss, acc

    def _train(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        max_epochs: int = 200,
    ):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=3e-5,
            verbose=False,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epochs_no_improve = 0
        patience = max(20, max_epochs // 4)

        for _ in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            _, val_acc = self._evaluate(model, val_loader, device, criterion)
            scheduler.step(val_acc)

            if val_acc > best_acc + 1e-4:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        model.load_state_dict(best_state)

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        # Reproducibility
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2_500_000))

        model = self._build_model(input_dim, num_classes, param_limit)
        model.to(device)

        train_samples = metadata.get("train_samples", None)
        if train_samples is not None:
            if train_samples <= 2048:
                max_epochs = 220
            elif train_samples <= 8192:
                max_epochs = 180
            else:
                max_epochs = 140
        else:
            max_epochs = 200

        self._train(model, train_loader, val_loader, device, max_epochs=max_epochs)
        model.eval()
        return model