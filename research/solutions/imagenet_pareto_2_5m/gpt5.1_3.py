import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + residual
        x = self.fc_out(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 2_500_000)

        device_str = metadata.get("device", "cpu")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)

        # Determine maximum hidden dimension under parameter constraint
        # params(H) = H^2 + 518H + 128
        A = 1
        B = 518
        C = 128 - param_limit
        disc = B * B - 4 * A * C
        if disc < 0:
            disc = 0
        hidden_dim = int((-B + math.sqrt(disc)) / (2 * A))
        if hidden_dim < 64:
            hidden_dim = 64

        # Adjust hidden_dim downward until we satisfy param_limit
        while hidden_dim > 0:
            model = MLPNet(input_dim, num_classes, hidden_dim, dropout=0.1)
            n_params = count_params(model)
            if n_params <= param_limit:
                break
            hidden_dim -= 1

        model.to(device)

        max_epochs = 200
        patience = 30
        lr = 3e-3
        weight_decay = 1e-4

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device=device, dtype=torch.long)
                    outputs = model(inputs)
                    _, preds = outputs.max(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

            scheduler.step()

            val_acc = (val_correct / val_total) if val_total > 0 else 0.0

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        return model