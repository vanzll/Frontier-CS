import math
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.15, use_weight_norm: bool = True):
        super().__init__()
        self.ln0 = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, num_classes, bias=True)

        if use_weight_norm:
            self.fc1 = nn.utils.weight_norm(self.fc1)
            self.fc2 = nn.utils.weight_norm(self.fc2)
            self.fc3 = nn.utils.weight_norm(self.fc3)

        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.ln0(x)
        x = self.act(self.fc1(x))
        x = self.drop1(x)
        y = self.act(self.fc2(x))
        y = self.drop2(y)
        x = x + torch.sigmoid(self.alpha) * y
        x = self.fc3(x)
        return x


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995, device=None):
        self.ema = copy.deepcopy(model).to(device or next(model.parameters()).device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if k.endswith('num_batches_tracked'):
                esd[k].copy_(msd[k])
            else:
                esd[k].mul_(self.decay).add_(msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return self.ema.state_dict()


class Solution:
    def _evaluate(self, model: nn.Module, data_loader, device: torch.device) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb.float())
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        return correct / max(1, total)

    def _train(self, model: nn.Module, train_loader, val_loader, device: torch.device, epochs: int = 200):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-4)

        ema = ModelEMA(model, decay=0.995, device=device)
        best_val = 0.0
        best_state = None
        patience = 40
        bad_epochs = 0

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb.float())
                loss = criterion(logits, yb.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(model)

            scheduler.step()

            # Evaluate EMA model
            val_acc = self._evaluate(ema.ema, val_loader, device)
            if val_acc >= best_val - 1e-12:
                best_val = val_acc
                best_state = copy.deepcopy(ema.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            model.load_state_dict(ema.state_dict())

        model.eval()
        return model

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 200000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
        except Exception:
            pass

        # Dynamic hidden size selection under parameter budget
        hidden = 256
        dropout = 0.15
        use_weight_norm = True

        def build_model(h):
            return ResidualMLP(input_dim, num_classes, hidden_dim=h, dropout=dropout, use_weight_norm=use_weight_norm)

        model = build_model(hidden)
        while count_trainable_params(model) > param_limit and hidden > 64:
            hidden -= 8
            model = build_model(hidden)

        model.to(device)

        # Train epochs scaled with dataset size (default around 200)
        train_samples = int(metadata.get("train_samples", 2048))
        base_epochs = 200
        # Scale epochs mildly with dataset size
        epochs = int(min(320, max(120, base_epochs * (2048 / max(1, train_samples)))))
        model = self._train(model, train_loader, val_loader, device, epochs=epochs)
        return model