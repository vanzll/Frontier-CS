import torch
import torch.nn as nn
import torch.optim as optim


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden1, hidden2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def estimate_params_two_hidden(input_dim, num_classes, h1, h2, use_bn=True):
    fc1 = input_dim * h1 + h1
    bn1 = 2 * h1 if use_bn else 0
    fc2 = h1 * h2 + h2
    bn2 = 2 * h2 if use_bn else 0
    fc3 = h2 * num_classes + num_classes
    return fc1 + bn1 + fc2 + bn2 + fc3


def choose_hidden_dims(input_dim, num_classes, param_limit, use_bn=True):
    best_h1, best_h2, best_params = 0, 0, 0
    # Search reasonable hidden sizes; step 16 from 32 to 512
    for h1 in range(32, 513, 16):
        for h2 in range(32, 513, 16):
            p = estimate_params_two_hidden(input_dim, num_classes, h1, h2, use_bn)
            if p <= param_limit and p > best_params:
                best_params = p
                best_h1, best_h2 = h1, h2
    if best_params == 0:
        # Fallback small network if param_limit is extremely tiny (not expected here)
        best_h1, best_h2 = 32, 32
    return best_h1, best_h2


def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                loss_sum += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    avg_loss = loss_sum / total if (criterion is not None and total > 0) else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        torch.manual_seed(42)

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 200000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        hidden1, hidden2 = choose_hidden_dims(input_dim, num_classes, param_limit, use_bn=True)
        model = SimpleMLP(input_dim, num_classes, hidden1, hidden2, dropout=0.2)
        model.to(device)

        # Safety check against param limit; if exceeded, fall back to a smaller fixed model
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            hidden1, hidden2 = 256, 128
            model = SimpleMLP(input_dim, num_classes, hidden1, hidden2, dropout=0.2)
            model.to(device)

        # Final param check (should be under limit now)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # As an ultimate fallback, minimal small model
            hidden1, hidden2 = 128, 64
            model = SimpleMLP(input_dim, num_classes, hidden1, hidden2, dropout=0.2)
            model.to(device)

        train_samples = metadata.get("train_samples", None)
        if train_samples is not None:
            if train_samples < 1000:
                epochs = 80
            elif train_samples > 4000:
                epochs = 60
            else:
                epochs = 120
        else:
            epochs = 120

        lr = 1.5e-3
        weight_decay = 1e-4
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

        best_val_acc = 0.0
        best_state = None
        patience = 25
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            _, val_acc = evaluate(model, val_loader, device, criterion=None)
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            scheduler.step()

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(torch.device("cpu"))
        model.eval()
        return model