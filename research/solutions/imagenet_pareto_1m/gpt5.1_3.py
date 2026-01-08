import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        bns = []

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            bns.append(nn.BatchNorm1d(hidden_dim))

        self.layers = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            out = layer(x)
            out = bn(out)
            out = self.act(out)
            out = self.dropout(out)
            if i > 0:
                x = x + out
            else:
                x = out
        logits = self.fc_out(x)
        return logits


class Solution:
    def _estimate_params(self, input_dim, num_classes, hidden_dim, num_layers):
        if hidden_dim <= 0 or num_layers <= 0:
            return float("inf")
        H = hidden_dim
        L = num_layers
        # Linear layers
        total = input_dim * H + H  # first layer
        if L > 1:
            total += (L - 1) * (H * H + H)  # hidden layers
        total += H * num_classes + num_classes  # output layer
        # BatchNorm layers
        total += 2 * H * L
        return int(total)

    def _choose_arch(self, input_dim, num_classes, param_limit):
        # Search over reasonable hidden sizes and depths to use parameter budget well
        best_hidden = 256
        best_layers = 2
        best_params = 0

        max_hidden = min(1024, max(128, param_limit))  # cap search
        min_hidden = 128
        step = 8

        for L in range(2, 7):  # 2 to 6 hidden layers
            for H in range(max_hidden, min_hidden - 1, -step):
                total = self._estimate_params(input_dim, num_classes, H, L)
                if total <= param_limit and total > best_params:
                    best_params = total
                    best_hidden = H
                    best_layers = L

        if best_params == 0:
            # Fallback simple architecture if search fails
            best_hidden = min(256, max(32, input_dim // 2))
            best_layers = 1

        return best_hidden, best_layers

    def _evaluate(self, model, data_loader, device, criterion=None):
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    loss_sum += loss.item()
                    n_batches += 1
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / n_batches if n_batches > 0 else 0.0
        return acc, avg_loss

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        # Determine input_dim and num_classes
        input_dim = metadata.get("input_dim", None)
        num_classes = metadata.get("num_classes", None)

        if input_dim is None or num_classes is None:
            # Infer from data loaders
            sample_inputs, sample_targets = next(iter(train_loader))
            flat_input = sample_inputs.view(sample_inputs.size(0), -1)
            input_dim = flat_input.size(1)
            num_classes = int(sample_targets.max().item()) + 1

        param_limit = metadata.get("param_limit", 1_000_000)

        # Architecture search within parameter limit
        hidden_dim, num_layers = self._choose_arch(input_dim, num_classes, param_limit)

        # Build model
        dropout = 0.2
        model = ResidualMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        # Ensure parameter constraint; if violated, shrink hidden_dim until it fits
        def count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        param_count = count_params(model)
        while param_count > param_limit and hidden_dim > 32:
            hidden_dim -= 16
            model = ResidualMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)
            param_count = count_params(model)

        # As an additional safeguard, if still too large, reduce layers
        while param_count > param_limit and num_layers > 1:
            num_layers -= 1
            model = ResidualMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)
            param_count = count_params(model)

        # Final check (in extreme case, could fall back to tiny baseline)
        if param_count > param_limit:
            # Tiny fallback network
            hidden_dim = 128
            num_layers = 1
            model = ResidualMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
            ).to(device)
            param_count = count_params(model)

        # Training hyperparameters
        train_samples = metadata.get("train_samples", None)
        if train_samples is not None and train_samples <= 4096:
            max_epochs = 250
            patience = 40
        else:
            max_epochs = 150
            patience = 30

        base_lr = 3e-4
        weight_decay = 1e-2

        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            scheduler.step()

            if val_loader is not None:
                val_acc, _ = self._evaluate(model, val_loader, device, criterion)
                if val_acc > best_val_acc:
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