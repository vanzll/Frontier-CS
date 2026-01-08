import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out + residual


class MLPResNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_blocks=4, head_dim=256, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.GELU()

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim, head_dim)
        self.fc1_act = nn.GELU()
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(head_dim, num_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.input_act(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x


def build_model(input_dim, num_classes, param_limit=None):
    hidden_dim = 512
    num_blocks = 4
    head_dim = 256
    dropout = 0.1

    while True:
        model = MLPResNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            head_dim=head_dim,
            dropout=dropout,
        )
        if param_limit is None:
            return model
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count <= param_limit:
            return model

        # Reduce capacity if over limit
        if num_blocks > 2:
            num_blocks -= 1
        elif hidden_dim > 256:
            hidden_dim -= 64
            head_dim = max(hidden_dim // 2, num_classes * 2)
        elif head_dim > num_classes:
            head_dim = max(head_dim // 2, num_classes)
        else:
            return model


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 2500000)
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        torch.manual_seed(42)

        model = build_model(input_dim, num_classes, param_limit)
        model.to(device)

        train_samples = metadata.get("train_samples", None)
        if train_samples is not None and train_samples <= 5000:
            num_epochs = 200
        else:
            num_epochs = 100

        learning_rate = 3e-3
        weight_decay = 1e-2

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_state = None
        best_acc = 0.0

        for _ in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.view(-1).to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, dtype=torch.float32)
                    targets = targets.view(-1).to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

            if total > 0:
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(model.state_dict())

            scheduler.step()

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to("cpu")
        model.eval()
        return model