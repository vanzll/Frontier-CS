import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, use_input_bn: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_input_bn = use_input_bn
        if use_input_bn:
            self.input_bn = nn.BatchNorm1d(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_input_bn:
            x = self.input_bn(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc_out(x)
        return x


class LogisticNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class Solution:
    def _estimate_params_mlp(self, input_dim: int, num_classes: int, hidden_dim: int, use_input_bn: bool = True) -> int:
        params = 0
        if use_input_bn:
            params += 2 * input_dim  # BatchNorm1d: weight + bias

        # fc1 + bn1
        params += input_dim * hidden_dim + hidden_dim  # weights + bias
        params += 2 * hidden_dim  # bn1

        # fc2 + bn2
        params += hidden_dim * hidden_dim + hidden_dim
        params += 2 * hidden_dim

        # output
        params += hidden_dim * num_classes + num_classes

        return params

    def _build_model(self, input_dim: int, num_classes: int, param_limit: int) -> nn.Module:
        # Try 2-hidden-layer MLP with BatchNorm, maximize hidden_dim under param_limit
        max_hidden_search = 512
        selected_hidden = None
        selected_use_input_bn = True

        for use_input_bn in (True, False):
            for h in range(max_hidden_search, 31, -8):
                p = self._estimate_params_mlp(input_dim, num_classes, h, use_input_bn=use_input_bn)
                if p <= param_limit:
                    selected_hidden = h
                    selected_use_input_bn = use_input_bn
                    break
            if selected_hidden is not None:
                break

        if selected_hidden is None:
            # Fallback to simple logistic regression if param_limit very small
            return LogisticNet(input_dim, num_classes)

        model = MLPNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=selected_hidden,
            use_input_bn=selected_use_input_bn,
            dropout=0.1,
        )
        return model

    def _train_one_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            return 0.0
        return total_loss / total_samples

    def _evaluate(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()

        if total_samples == 0:
            return 0.0, 0.0
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        model = self._build_model(input_dim, num_classes, param_limit)
        model.to(device)

        # Ensure parameter constraint
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # As a last resort, use logistic regression which will certainly be under the limit
            model = LogisticNet(input_dim, num_classes).to(device)

        # Training setup
        num_epochs = 120
        patience = 25

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_state = None
        best_val_acc = 0.0
        epochs_no_improve = 0

        has_val = val_loader is not None

        for epoch in range(num_epochs):
            self._train_one_epoch(model, train_loader, optimizer, criterion, device)

            if has_val:
                _, val_acc = self._evaluate(model, val_loader, criterion, device)

                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        scheduler.step()
                        break
            scheduler.step()

        if has_val and best_state is not None:
            model.load_state_dict(best_state)

        model.to("cpu")
        return model