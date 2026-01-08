import torch
import torch.nn as nn
import copy


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
    # Configs ordered from largest to smallest (approx), all within or close to 500k for typical settings
    configs = [
        [512, 384, 192],
        [512, 384, 160],
        [512, 320, 160],
        [512, 256, 128],
        [384, 384, 192],
        [384, 384, 128],
        [384, 256, 128],
        [384, 256],
        [256, 256],
        [256, 128],
        [256],
        [],
    ]

    chosen_config = None
    for cfg in configs:
        model = MLPNet(input_dim, num_classes, cfg)
        if count_parameters(model) <= param_limit:
            chosen_config = cfg
            break

    if chosen_config is None:
        chosen_config = []

    model = MLPNet(input_dim, num_classes, chosen_config)
    return model


def evaluate_accuracy(model: nn.Module, data_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / total if total > 0 else 0.0


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            first_batch = next(iter(train_loader))
            input_dim = first_batch[0].shape[1]
            num_classes = int(torch.max(first_batch[1]).item()) + 1
            param_limit = 500000
            device_str = "cpu"
            train_samples = len(getattr(train_loader, "dataset", []))
        else:
            input_dim = metadata.get("input_dim")
            num_classes = metadata.get("num_classes")
            param_limit = metadata.get("param_limit", 500000)
            device_str = metadata.get("device", "cpu")
            train_samples = metadata.get("train_samples", len(getattr(train_loader, "dataset", [])))

        if device_str == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        model = build_model(input_dim, num_classes, param_limit)
        if count_parameters(model) > param_limit:
            # Safety fallback to a small model if for some reason the chosen config exceeds the limit
            model = MLPNet(input_dim, num_classes, [256])

        model.to(device)

        max_epochs = 160
        if train_samples is not None and train_samples > 0:
            if train_samples < 1500:
                max_epochs = 220
            elif train_samples > 10000:
                max_epochs = 80

        lr = 3e-3
        weight_decay = 1e-4

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=3e-4)

        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0
        patience = 40
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
                    break

            scheduler.step()

        model.load_state_dict(best_state)
        model.to("cpu")
        return model