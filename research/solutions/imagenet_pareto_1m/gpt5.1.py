import torch
import torch.nn as nn
import random
import numpy as np


class MLPNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 768)
        self.fc2 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = self.act(self.fc3(x))
        x = self.dropout(x)

        residual = x
        out = self.act(self.fc4(x))
        out = self.dropout(out)
        out = self.fc5(out)
        out = out + residual
        out = self.act(out)
        out = self.dropout(out)

        logits = self.out(out)
        return logits


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            raise ValueError("metadata must be provided")

        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        seed = metadata.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata.get("param_limit", 1_000_000)

        model = MLPNet(input_dim, num_classes, dropout=0.1)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > param_limit:
            raise RuntimeError(
                f"Model has {total_params} parameters which exceeds limit {param_limit}"
            )
        model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

        train_samples = metadata.get("train_samples", None)
        if train_samples is None:
            try:
                train_samples = len(train_loader.dataset)
            except Exception:
                train_samples = None

        if train_samples is not None and train_samples <= 5000:
            max_epochs = 350
            early_stopping_patience = 60
        else:
            max_epochs = 120
            early_stopping_patience = 20

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8,
            verbose=False,
            min_lr=1e-5,
        )

        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            if val_loader is not None:
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device, dtype=torch.float32)
                        targets = targets.to(device, dtype=torch.long)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * targets.size(0)
                        preds = outputs.argmax(dim=1)
                        val_correct += (preds == targets).sum().item()
                        val_total += targets.size(0)

                if val_total > 0:
                    val_loss /= val_total
                    val_acc = val_correct / val_total
                else:
                    val_loss = 0.0
                    val_acc = 0.0

                scheduler.step(val_acc)

                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        return model