import math
import copy
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _collect_loader(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Loader must yield (inputs, targets).")
        x = torch.as_tensor(x).detach().cpu()
        y = torch.as_tensor(y).detach().cpu()
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    if x.dtype != torch.float32:
        x = x.float()
    if y.dtype != torch.long:
        y = y.long()
    return x, y


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _MLP200K(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.register_buffer("x_mean", mean.view(1, -1).contiguous())
        self.register_buffer("x_std", std.view(1, -1).contiguous())

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.ln1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.ln2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

        self.ln_out = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes, bias=True)

        self.drop = nn.Dropout(p=float(dropout))
        self.act = nn.GELU(approximate="tanh")

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="linear")
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = (x - self.x_mean) / self.x_std

        x = self.fc1(x)
        x = self.act(x)
        x = self.ln1(x)
        x = self.drop(x)

        res = x
        x = self.fc2(x)
        x = self.act(x)
        x = self.ln2(x)
        x = self.drop(x)

        x = x + res
        x = self.ln_out(x)

        x = self.fc3(x)
        return x


@torch.no_grad()
def _accuracy_from_tensors(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    n = y.numel()
    if n == 0:
        return 0.0
    correct = 0
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
    return float(correct) / float(n)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}

        device = metadata.get("device", "cpu")
        if device != "cpu":
            device = "cpu"

        torch.manual_seed(0)

        train_x, train_y = _collect_loader(train_loader)
        val_x, val_y = _collect_loader(val_loader)

        input_dim = int(metadata.get("input_dim", train_x.shape[-1]))
        num_classes = int(metadata.get("num_classes", int(train_y.max().item()) + 1))
        param_limit = int(metadata.get("param_limit", 200_000))

        if train_x.ndim != 2 or train_x.shape[1] != input_dim:
            train_x = train_x.view(train_x.shape[0], -1)
        if val_x.ndim != 2 or val_x.shape[1] != input_dim:
            val_x = val_x.view(val_x.shape[0], -1)

        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)

        # Choose the widest 2-hidden-layer MLP (H->H) that fits within param_limit:
        # params = D*H + H*H + H*C + C (+ biases for fc3 only)
        # (fc1, fc2 bias=False; fc3 bias=True)
        D, C = input_dim, num_classes
        best_h = None
        best_params = -1
        for h in range(512, 31, -1):
            params = D * h + h * h + h * C + C
            if params <= param_limit:
                best_h = h
                best_params = params
                break

        if best_h is None:
            # Fallback to linear classifier
            model = nn.Linear(input_dim, num_classes, bias=True).to(device)
            if _count_trainable_params(model) > param_limit:
                # Extreme fallback: bias-less
                model = nn.Linear(input_dim, num_classes, bias=False).to(device)
            return model.cpu().eval()

        model = _MLP200K(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=int(best_h),
            mean=mean,
            std=std,
            dropout=0.20,
        ).to(device)

        if _count_trainable_params(model) > param_limit:
            # Should not happen, but keep safe.
            for h in range(best_h - 1, 31, -1):
                tmp = _MLP200K(input_dim, num_classes, h, mean, std, dropout=0.20).to(device)
                if _count_trainable_params(tmp) <= param_limit:
                    model = tmp
                    break

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        batch_size = 128
        if hasattr(train_loader, "batch_size") and train_loader.batch_size is not None:
            try:
                batch_size = int(train_loader.batch_size)
            except Exception:
                batch_size = 128
        batch_size = max(32, min(256, batch_size))

        train_ds = TensorDataset(train_x, train_y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        max_epochs = 260
        warmup_epochs = 10
        patience = 35

        lr = 5e-3
        wd = 2e-2
        criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98), eps=1e-8)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            t = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        best_state = None
        best_acc = -1.0
        best_epoch = -1
        bad = 0

        noise_std = 0.08
        grad_clip = 1.0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_dl:
                if noise_std > 0.0:
                    xb = xb + noise_std * torch.randn_like(xb)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            scheduler.step()

            val_acc = _accuracy_from_tensors(model, val_x, val_y, batch_size=1024)
            if val_acc > best_acc + 1e-6:
                best_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience and epoch >= warmup_epochs + 15:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Light fine-tuning on train+val (more data), small LR, few epochs
        all_x = torch.cat([train_x, val_x], dim=0)
        all_y = torch.cat([train_y, val_y], dim=0)
        all_ds = TensorDataset(all_x, all_y)
        all_dl = DataLoader(all_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        ft_epochs = 25
        ft_lr = 1.2e-3
        ft_wd = 1.5e-2
        ft_opt = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=ft_wd, betas=(0.9, 0.98), eps=1e-8)

        def ft_lr_lambda(epoch: int) -> float:
            t = float(epoch) / float(max(1, ft_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))

        ft_sched = torch.optim.lr_scheduler.LambdaLR(ft_opt, lr_lambda=ft_lr_lambda)

        for _ in range(ft_epochs):
            model.train()
            for xb, yb in all_dl:
                if noise_std > 0.0:
                    xb = xb + (noise_std * 0.75) * torch.randn_like(xb)

                ft_opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                ft_opt.step()
            ft_sched.step()

        model = model.cpu()
        model.eval()

        if _count_trainable_params(model) > param_limit:
            # Final hard safety fallback
            fallback = nn.Linear(input_dim, num_classes, bias=True)
            if _count_trainable_params(fallback) <= param_limit:
                fallback.eval()
                return fallback
            fallback = nn.Linear(input_dim, num_classes, bias=False)
            fallback.eval()
            return fallback

        return model