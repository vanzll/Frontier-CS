import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + identity
        out = F.relu(out)
        return out


class MLPResNet(nn.Module):
    def __init__(self, input_dim, num_classes, width, num_blocks, dropout=0.2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, width)
        self.ln_in = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dropout=dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.ln_in(x)
        x = F.relu(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_best_width_for_blocks(input_dim, num_classes, blocks, param_limit, dropout=0.2):
    base = max(1, int(math.sqrt(max(param_limit, 1) / (2.0 * max(blocks, 1)))))
    hi = min(base * 2 + 32, param_limit)
    lo = 1
    best_width = 0
    best_params = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        model = MLPResNet(input_dim, num_classes, mid, blocks, dropout=dropout)
        params = count_parameters(model)
        if params <= param_limit:
            best_width = mid
            best_params = params
            lo = mid + 1
        else:
            hi = mid - 1

    return best_width, best_params


def build_best_model(input_dim, num_classes, param_limit, dropout=0.2):
    best_model = None
    best_params = -1
    best_cfg = None

    for blocks in (3, 2, 1):
        width, params = find_best_width_for_blocks(
            input_dim, num_classes, blocks, param_limit, dropout=dropout
        )
        if width > 0 and params <= param_limit and params > best_params:
            best_params = params
            best_cfg = (blocks, width)

    if best_cfg is not None:
        blocks, width = best_cfg
        model = MLPResNet(input_dim, num_classes, width, blocks, dropout=dropout)
        return model

    # Fallback to simple linear model if param limit is extremely small
    model = nn.Linear(input_dim, num_classes)
    if count_parameters(model) > param_limit:
        # Attempt an even smaller two-layer network with bottleneck if needed
        bottleneck = max(1, param_limit // (input_dim + num_classes + 10))
        model = nn.Sequential(
            nn.Linear(input_dim, bottleneck, bias=True),
            nn.ReLU(),
            nn.Linear(bottleneck, num_classes, bias=True),
        )
    return model


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def train_model(model, train_loader, val_loader, device, max_epochs=200):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=3e-5
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    patience = 25
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if val_loader is not None:
            val_acc = evaluate_accuracy(model, val_loader, device)
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                scheduler.step()
                break

        scheduler.step()

    if val_loader is not None:
        model.load_state_dict(best_state)

    model.to("cpu")
    return model


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            raise ValueError("Metadata dictionary is required.")

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 1_000_000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        torch.manual_seed(42)

        model = build_best_model(input_dim, num_classes, param_limit, dropout=0.2)
        model = train_model(model, train_loader, val_loader, device=device, max_epochs=200)
        return model