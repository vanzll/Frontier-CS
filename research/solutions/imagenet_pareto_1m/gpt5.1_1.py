import torch
import torch.nn as nn
import numpy as np
import random
import copy


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_256_blocks=5, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim

        # Hidden layer configuration: two 512 layers, then one 256, then several 256 blocks
        hidden_dims = [512, 512, 256] + [256] * num_256_blocks

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim

        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if metadata is None:
            raise ValueError("metadata dictionary is required")

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata.get("param_limit", 1_000_000)

        # Function to build model with given number of 256-dim blocks
        def build_model(num_256_blocks):
            return MLPNet(input_dim, num_classes, num_256_blocks=num_256_blocks, dropout=0.1)

        # Choose the largest model under the parameter limit
        model = None
        for blocks in range(5, 0, -1):
            candidate = build_model(blocks)
            param_count = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
            if param_count <= param_limit:
                model = candidate
                break

        if model is None:
            # Fallback small baseline model if, for some reason, none fit
            model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )

        model.to(device)

        # Training hyperparameters
        max_epochs = 200
        patience = 30
        lr = 1e-3
        weight_decay = 1e-4

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        def evaluate_accuracy(loader):
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == targets).sum().item()
                    total_samples += targets.size(0)
            if total_samples == 0:
                return 0.0
            return total_correct / total_samples

        use_validation = val_loader is not None
        best_state = None
        best_val_acc = -1.0
        epochs_no_improve = 0

        if use_validation:
            best_state = copy.deepcopy(model.state_dict())

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            scheduler.step()

            if use_validation:
                val_acc = evaluate_accuracy(val_loader)
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Load best validation state if available
        if use_validation and best_state is not None:
            model.load_state_dict(best_state)

        model.to(device)
        model.eval()
        return model