import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader
import copy


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class Solution:
    def _build_model(self, input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
        # Try to get the widest hidden size under the parameter budget
        # using a 3-hidden-layer MLP with BatchNorm and Dropout.
        hidden_dim = 1600
        best_model = None
        best_hidden = None

        while hidden_dim >= 16:
            model = MLPNet(input_dim, num_classes, hidden_dim, dropout=0.2)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if params <= param_limit:
                best_model = model
                best_hidden = hidden_dim
                break
            hidden_dim -= 16

        if best_model is None:
            # Fallback tiny model if param_limit is extremely small
            best_model = MLPNet(input_dim, num_classes, hidden_dim=32, dropout=0.0)

        return best_model

    def _train_one_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    def _evaluate(self, model, loader, criterion, device):
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
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 5_000_000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        torch.manual_seed(42)

        model = self._build_model(input_dim, num_classes, param_limit)
        model.to(device)

        # Double-check parameter constraint
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # As a safety fallback, shrink hidden size aggressively
            model = MLPNet(input_dim, num_classes, hidden_dim=256, dropout=0.1)
            model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        max_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_val_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        patience = 30

        for epoch in range(max_epochs):
            self._train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            if val_loader is not None:
                _, val_acc = self._evaluate(model, val_loader, criterion, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                elif epoch - best_epoch >= patience:
                    break

        # Load best validation model if validation loader exists
        if val_loader is not None and best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        model.to("cpu")
        return model