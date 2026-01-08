import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


class ResidualLinearBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ln(x)
        y = self.fc(self.act(y))
        y = self.drop(y)
        return x + y


class MLPResidualNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, depth: int, dropout: float = 0.1):
        super().__init__()
        self.in_ln = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResidualLinearBlock(width, dropout=dropout, activation="gelu") for _ in range(depth)])
        self.out_ln = nn.LayerNorm(width)
        self.out_drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)

        # Initialize head to small weights for stability
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.in_proj(self.in_ln(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.out_ln(x)
        x = self.out_drop(x)
        x = self.head(x)
        return x


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_params(input_dim: int, num_classes: int, width: int, depth: int) -> int:
    # in_ln: 2*input_dim
    # in_proj: input_dim*width + width
    # each block: ln(2*width) + fc(width*width + width)
    # out_ln: 2*width
    # head: width*num_classes + num_classes
    params = 2 * input_dim
    params += input_dim * width + width
    for _ in range(depth):
        params += 2 * width
        params += width * width + width
    params += 2 * width
    params += width * num_classes + num_classes
    return params


def choose_architecture(input_dim: int, num_classes: int, param_limit: int):
    best = None
    best_params = -1
    # Search reasonable grid
    for width in range(512, 1793, 32):
        for depth in range(1, 12):  # number of residual blocks
            p = estimate_params(input_dim, num_classes, width, depth)
            if p <= param_limit and p > best_params:
                best_params = p
                best = (width, depth)
    if best is None:
        # Fallback minimal model
        return 256, 1
    return best


@torch.no_grad()
def evaluate_acc(model: nn.Module, data_loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    for xb, yb in data_loader:
        xb = xb.to(device, non_blocking=False)
        yb = yb.to(device, non_blocking=False)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        set_seed(42)

        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 5_000_000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        # Choose architecture close to parameter limit without exceeding
        width, depth = choose_architecture(input_dim, num_classes, param_limit - 8_000)  # small safety margin

        # Reasonable dropout tuned by width/depth
        if width >= 1280 or depth >= 6:
            dropout = 0.2
        elif width >= 1024 or depth >= 4:
            dropout = 0.15
        else:
            dropout = 0.1

        model = MLPResidualNet(input_dim=input_dim, num_classes=num_classes, width=width, depth=depth, dropout=dropout).to(device)

        # Ensure parameter constraint
        total_params = count_trainable_params(model)
        if total_params > param_limit:
            # As a final safeguard, progressively reduce depth then width
            while total_params > param_limit and depth > 1:
                depth -= 1
                model = MLPResidualNet(input_dim=input_dim, num_classes=num_classes, width=width, depth=depth, dropout=dropout).to(device)
                total_params = count_trainable_params(model)
            while total_params > param_limit and width > 256:
                width -= 32
                model = MLPResidualNet(input_dim=input_dim, num_classes=num_classes, width=width, depth=depth, dropout=dropout).to(device)
                total_params = count_trainable_params(model)

        # Optimizer and scheduler
        base_lr = 1e-3
        weight_decay = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        # Training setup
        epochs = 140
        steps_per_epoch = max(1, len(train_loader))
        # OneCycleLR provides robust training for various settings
        max_lr = 6e-3 if width <= 1024 else 4e-3
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=100.0,
        )

        # Loss with label smoothing
        # Using CE with smoothing provides regularization akin to mixup but simpler and faster
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.06)

        best_val_acc = -1.0
        best_state = None
        patience = 25
        epochs_no_improve = 0
        grad_clip = 1.0

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=False)
                yb = yb.to(device, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = ce_loss(logits, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            # Validation
            if val_loader is not None:
                val_acc = evaluate_acc(model, val_loader, device)
            else:
                # If no val loader provided, use train loader for monitoring
                val_acc = evaluate_acc(model, train_loader, device)

            if val_acc > best_val_acc + 1e-5:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
        model.to(device)
        model.eval()
        return model