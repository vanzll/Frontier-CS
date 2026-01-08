import math
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, width: int, bottleneck: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, bottleneck, bias=True)
        self.fc2 = nn.Linear(bottleneck, width, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x):
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int, bottleneck: int, num_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.stem = nn.Linear(input_dim, width, bias=True)
        self.blocks = nn.ModuleList([BottleneckBlock(width, bottleneck, dropout=dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes, bias=True)
        self.act = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.stem.weight, nonlinearity='linear')
        nn.init.zeros_(self.stem.bias)
        nn.init.ones_(self.in_norm.weight)
        nn.init.zeros_(self.in_norm.bias)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)
        # Head initialized near zero to stabilize early training
        nn.init.zeros_(self.head.bias)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity='linear')

    def forward(self, x):
        x = self.in_norm(x)
        x = self.stem(x)
        x = self.act(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        x = self.head(x)
        return x


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def choose_arch(param_limit: int, input_dim: int, num_classes: int):
    # Search over number of blocks, widths, and compute best bottleneck to maximize params under limit.
    best = None  # (params, n, w, m)
    # Reasonable ranges for CPU training
    n_candidates = [4, 3, 2]
    w_candidates = list(range(768, 383, -32))  # 768 down to 384 by 32
    for n in n_candidates:
        for w in w_candidates:
            # derive maximum m for given (n, w) using param formula including input LayerNorm
            # T = 2*n*w*m + n*m + w*(input_dim + num_classes + 3*n + 3) + 2*input_dim + num_classes
            denom = 2 * n * w + n
            base = w * (input_dim + num_classes + 3 * n + 3) + 2 * input_dim + num_classes
            remaining = param_limit - base
            if remaining <= 0 or denom <= 0:
                continue
            m_max = remaining // denom
            # Clamp m to [64, w] and to multiple of 16 for hardware-friendliness
            m_max = max(16, min(int(m_max), w))
            m_max = (m_max // 16) * 16
            if m_max < 16:
                continue

            # Try few candidates around m_max to utilize budget optimally
            for delta in [0, -16, -32, -48]:
                m = max(16, min(w, m_max + delta))
                # Compute exact param count by formula
                T = 2 * n * w * m + n * m + w * (input_dim + num_classes + 3 * n + 3) + 2 * input_dim + num_classes
                if T <= param_limit:
                    if best is None or T > best[0]:
                        best = (T, n, w, m)
            # Early exit if near limit
            if best is not None and best[0] > param_limit * 0.995:
                break
        if best is not None and best[0] > param_limit * 0.995:
            break

    # Fallback if none found (should not happen)
    if best is None:
        # Use a safe small architecture
        n, w, m = 2, 512, 128
        return n, w, m
    _, n, w, m = best
    return n, w, m


def get_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
            yb = yb.to(device=device, dtype=torch.long, non_blocking=False)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            loss_sum += loss.item() * yb.numel()
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    return acc, avg_loss


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        input_dim = int(metadata.get("input_dim", 384))
        num_classes = int(metadata.get("num_classes", 128))
        param_limit = int(metadata.get("param_limit", 1_000_000))
        device_str = metadata.get("device", "cpu")
        device = torch.device(device_str)

        # Reproducibility
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        # Choose architecture to utilize parameter budget
        n_blocks, width, bottleneck = choose_arch(param_limit, input_dim, num_classes)

        # Build model
        model = MLPNet(
            input_dim=input_dim,
            num_classes=num_classes,
            width=width,
            bottleneck=bottleneck,
            num_blocks=n_blocks,
            dropout=0.1
        ).to(device)

        # Ensure param constraint
        params = count_trainable_params(model)
        if params > param_limit:
            # As a safety fallback, reduce bottleneck and/or width until under limit
            # Reduce bottleneck first
            b = bottleneck
            w = width
            n = n_blocks
            while params > param_limit and b >= 32:
                b = max(16, (b // 2))
                model = MLPNet(input_dim, num_classes, w, b, n, dropout=0.1).to(device)
                params = count_trainable_params(model)
            # If still too large, reduce width
            while params > param_limit and w >= 256:
                w = max(256, w - 32)
                b = min(b, w)
                model = MLPNet(input_dim, num_classes, w, b, n, dropout=0.1).to(device)
                params = count_trainable_params(model)
            # If still too large, reduce blocks
            while params > param_limit and n > 1:
                n -= 1
                model = MLPNet(input_dim, num_classes, w, b, n, dropout=0.1).to(device)
                params = count_trainable_params(model)
            # Final check (should be satisfied)
            assert params <= param_limit, "Parameter limit could not be satisfied."

        # Training setup
        train_steps_per_epoch = max(1, len(train_loader))
        max_epochs = 220
        patience = 40

        base_lr = 3e-3
        weight_decay = 0.05
        param_groups = get_param_groups(model, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.99))
        # OneCycleLR for smooth warmup and cosine annealing
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            epochs=max_epochs,
            steps_per_epoch=train_steps_per_epoch,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=25.0
        )

        # Loss with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # EMA for better generalization
        ema = EMA(model, decay=0.995)

        best_acc = 0.0
        best_epoch = -1
        best_ema_state = None

        # Training loop
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            for xb, yb in train_loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=False)
                yb = yb.to(device=device, dtype=torch.long, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                ema.update(model)

                bs = yb.size(0)
                running_loss += loss.item() * bs
                total_samples += bs

            # Validation with EMA weights
            ema.apply_to(model)
            val_acc, val_loss = evaluate(model, val_loader, device)
            ema.restore(model)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_ema_state = ema.state_dict()

            # Early stopping
            if epoch - best_epoch >= patience:
                break

        # Load best EMA weights
        if best_ema_state is not None:
            ema.load_state_dict(best_ema_state)
            ema.apply_to(model)  # Keep EMA weights in model for evaluation

        model.to('cpu')
        model.eval()
        return model