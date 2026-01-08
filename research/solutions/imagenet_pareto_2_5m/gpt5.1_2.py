import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from copy import deepcopy


class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(1024, 1024, 512, 512), dropout=0.2):
        super().__init__()
        assert len(hidden_dims) == 4
        h1, h2, h3, h4 = hidden_dims

        self.input_ln = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(dropout * 0.5)

        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)

        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)

        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)

        self.fc4 = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4)

        self.fc5 = nn.Linear(h4, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.use_res1 = h1 == h2
        self.use_res2 = h3 == h4

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_ln(x)
        x = self.input_dropout(x)

        # Block 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        # Block 2 with optional residual from Block 1
        residual = x if self.use_res1 else None
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        if residual is not None:
            x = x + residual

        # Block 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)

        # Block 4 with optional residual from Block 3
        residual = x if self.use_res2 else None
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout(x)
        if residual is not None:
            x = x + residual

        x = self.fc5(x)
        return x


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 2500000))
        device_str = metadata.get("device", "cpu")

        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        torch.manual_seed(42)

        # Build model under parameter constraint
        candidate_hidden_configs = [
            (1024, 1024, 512, 512),
            (896, 896, 384, 384),
            (768, 768, 256, 256),
            (512, 512, 256, 256),
            (384, 384, 256, 256),
        ]

        model = None
        for hidden in candidate_hidden_configs:
            tmp_model = DeepMLP(input_dim, num_classes, hidden_dims=hidden, dropout=0.2)
            param_count = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
            if param_count <= param_limit:
                model = tmp_model
                break

        if model is None:
            # Fallback to simple linear classifier if param_limit is extremely small
            model = nn.Linear(input_dim, num_classes)

        model.to(device)

        # Training setup
        train_samples = metadata.get("train_samples", None)
        if train_samples is None:
            try:
                train_samples = len(train_loader.dataset)
            except Exception:
                train_samples = None

        if train_samples is not None:
            if train_samples <= 4000:
                num_epochs = 220
            elif train_samples <= 20000:
                num_epochs = 100
            else:
                num_epochs = 50
        else:
            num_epochs = 150

        patience = max(15, num_epochs // 5)
        base_lr = 5e-4

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        def evaluate(model_eval, loader, criterion_eval, device_eval):
            model_eval.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(device_eval, dtype=torch.float32)
                    targets = targets.to(device_eval, dtype=torch.long)
                    outputs = model_eval(inputs)
                    loss = criterion_eval(outputs, targets)
                    total_loss += loss.item() * targets.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            if total == 0:
                return 0.0, 0.0
            return total_loss / total, correct / total

        best_state = deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        scheduler.step()
                        break
            scheduler.step()

        if val_loader is not None:
            model.load_state_dict(best_state)

            # Optional fine-tuning on train + val with lower LR
            try:
                train_dataset = train_loader.dataset
                val_dataset = val_loader.dataset
                combined_dataset = ConcatDataset([train_dataset, val_dataset])
                batch_size = getattr(train_loader, "batch_size", None) or 64
                combined_loader = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                )

                fine_tune_epochs = max(5, num_epochs // 10)
                optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr * 0.25, weight_decay=1e-2)

                for _ in range(fine_tune_epochs):
                    model.train()
                    for inputs, targets in combined_loader:
                        inputs = inputs.to(device, dtype=torch.float32)
                        targets = targets.to(device, dtype=torch.long)

                        optimizer.zero_grad(set_to_none=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
            except Exception:
                pass

        model.to(torch.device("cpu"))
        model.eval()
        return model