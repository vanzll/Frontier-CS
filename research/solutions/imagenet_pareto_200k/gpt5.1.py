import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Dict


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _create_mlp(input_dim: int, num_classes: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else num_classes
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def solve(self, train_loader, val_loader, metadata: Dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        torch.manual_seed(42)

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # Build model under parameter constraint
        dropout = 0.1
        model = None

        candidate_num_layers = [3, 2]
        candidate_hidden_dims = [256, 224, 192, 160, 128, 96, 64]

        for num_layers in candidate_num_layers:
            for hidden_dim in candidate_hidden_dims:
                tmp_model = self._create_mlp(input_dim, num_classes, hidden_dim, num_layers, dropout)
                if self._count_params(tmp_model) <= param_limit:
                    model = tmp_model
                    break
            if model is not None:
                break

        # Fallback to simple linear classifier if needed
        if model is None:
            model = nn.Linear(input_dim, num_classes)

        model.to(device)

        # Training setup
        lr = 0.002
        weight_decay = 1e-2
        max_epochs = 160
        patience = 30

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = -float("inf")
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

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
                    val_total += targets.numel()

            val_acc = val_correct / val_total if val_total > 0 else 0.0

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            scheduler.step()

        model.load_state_dict(best_state)
        model.to(torch.device("cpu"))
        return model