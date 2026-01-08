import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims, dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        layers = []
        bns = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                bns.append(nn.BatchNorm1d(h))
            prev_dim = h
        self.layers = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns) if use_bn else None
        self.out = nn.Linear(prev_dim, num_classes)
        self.activation = nn.GELU()
        self.use_bn = use_bn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.out(x)
        return x


def evaluate(model: nn.Module, data_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / total if (criterion is not None and total > 0) else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 5_000_000)
        device_str = metadata.get("device", "cpu")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)

        # Base architecture close to param_limit but under it
        base_hidden_dims = [1536, 1536, 1024]
        dropout = 0.1
        use_bn = True

        model = MLPNet(input_dim, num_classes, base_hidden_dims, dropout=dropout, use_bn=use_bn)
        current_params = count_parameters(model)

        # Adjust architecture if exceeding param limit (for safety / different metadata)
        if current_params > param_limit:
            scale = math.sqrt(param_limit / current_params) * 0.95
            hidden_dims = base_hidden_dims[:]
            for _ in range(10):
                scaled_hidden = [max(num_classes * 2, int(h * scale)) for h in hidden_dims]
                tmp_model = MLPNet(input_dim, num_classes, scaled_hidden, dropout=dropout, use_bn=use_bn)
                tmp_params = count_parameters(tmp_model)
                if tmp_params <= param_limit:
                    model = tmp_model
                    current_params = tmp_params
                    break
                scale *= 0.9
        model.to(device)

        # Training setup
        try:
            train_samples = metadata.get("train_samples", len(train_loader.dataset))
        except Exception:
            train_samples = metadata.get("train_samples", 2048)

        if train_samples <= 4096:
            max_epochs = 300
            patience = 40
        elif train_samples <= 10000:
            max_epochs = 180
            patience = 25
        else:
            max_epochs = 100
            patience = 15

        base_lr = 3e-3
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)
        try:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=3e-5)
        except TypeError:
            scheduler = None

        try:
            criterion_train = nn.CrossEntropyLoss(label_smoothing=0.1)
        except TypeError:
            criterion_train = nn.CrossEntropyLoss()
        criterion_val = nn.CrossEntropyLoss()

        best_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion_train(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            val_loss, val_acc = evaluate(model, val_loader, device, criterion_val)

            if scheduler is not None:
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
        model.to(torch.device("cpu"))
        return model