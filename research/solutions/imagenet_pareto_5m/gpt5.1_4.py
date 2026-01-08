import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 1200, num_hidden_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_hidden_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = F.gelu(x)
            if self.dropout_p > 0.0:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.out(x)
        return x


class Solution:
    def _count_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _build_model(self, input_dim, num_classes, param_limit):
        hidden_dim = 1200
        num_hidden_layers = 4
        dropout = 0.2

        # Adjust model size if param_limit is smaller than expected
        while True:
            model = MLPNet(input_dim, num_classes, hidden_dim=hidden_dim,
                           num_hidden_layers=num_hidden_layers, dropout=dropout)
            param_count = self._count_params(model)
            if param_count <= param_limit:
                return model

            # Reduce width first
            if hidden_dim > 128:
                hidden_dim = max(128, int(hidden_dim * 0.9))
            # Then reduce depth if needed
            elif num_hidden_layers > 2:
                num_hidden_layers -= 1
            else:
                # Cannot shrink further meaningfully; return smallest
                return model

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 5_000_000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        torch.manual_seed(42)

        model = self._build_model(input_dim, num_classes, param_limit)
        model.to(device)

        lr = 1e-3
        weight_decay = 5e-4
        max_epochs = 200
        patience = 30

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_val_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())
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

            val_acc = val_correct / val_total if val_total > 0 else 0.0

            if scheduler is not None:
                scheduler.step()

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        return model