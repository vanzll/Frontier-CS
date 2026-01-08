import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from copy import deepcopy


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.2, use_bn=True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Solution:
    def _build_model(self, input_dim, num_classes, param_limit, device):
        hidden_candidates = [
            [640, 320],
            [512, 384],
            [512, 256],
            [512, 192],
            [384, 256],
            [384, 192],
            [320, 160],
            [256, 128],
            [256, 64],
            [192, 96],
            [128, 64],
            [96, 48],
            [64, 32],
            [64],
            [48],
            [32],
            [24],
            [16],
            [8],
            [4],
            [2],
            [1],
            []
        ]
        chosen_model = None
        for hidden_dims in hidden_candidates:
            use_bn = len(hidden_dims) > 0
            dropout = 0.25 if len(hidden_dims) > 0 else 0.0
            model = MLPNet(input_dim, num_classes, hidden_dims, dropout=dropout, use_bn=use_bn)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if param_count <= param_limit:
                chosen_model = model
                break

        if chosen_model is None:
            # Extreme fallback, should rarely happen given reasonable param_limit
            chosen_model = MLPNet(input_dim, num_classes, [1], dropout=0.0, use_bn=False)

        chosen_model.to(device)
        return chosen_model

    def _evaluate(self, model, loader, device, criterion):
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
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
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

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 500000))

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        model = self._build_model(input_dim, num_classes, param_limit, device)

        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        max_epochs = 120
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_state = None
        best_val_acc = 0.0
        patience = 20
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                _, val_acc = self._evaluate(model, val_loader, device, criterion)
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_state = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            scheduler.step()

            if val_loader is not None and epochs_no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()
        return model