import os
import math
import torch
import torch.nn as nn


class _RandomFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, m_relu: int, m_rff: int, seed: int = 0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.m_relu = int(m_relu)
        self.m_rff = int(m_rff)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        self.register_buffer("mean_x", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("std_x", torch.ones(self.input_dim, dtype=torch.float32))

        if self.m_relu > 0:
            scale = 1.0 / math.sqrt(max(1, self.input_dim))
            R_relu = torch.randn(self.input_dim, self.m_relu, generator=g, dtype=torch.float32) * scale
        else:
            R_relu = torch.empty(self.input_dim, 0, dtype=torch.float32)
        self.register_buffer("R_relu", R_relu)

        if self.m_rff > 0:
            scale = 1.0 / math.sqrt(max(1, self.input_dim))
            R_rff = torch.randn(self.input_dim, self.m_rff, generator=g, dtype=torch.float32) * scale
            b_rff = torch.rand(self.m_rff, generator=g, dtype=torch.float32) * (2.0 * math.pi)
        else:
            R_rff = torch.empty(self.input_dim, 0, dtype=torch.float32)
            b_rff = torch.empty(0, dtype=torch.float32)
        self.register_buffer("R_rff", R_rff)
        self.register_buffer("b_rff", b_rff)

        raw_dim = 2 * self.input_dim + self.m_relu + self.m_rff
        self.raw_dim = int(raw_dim)
        self.register_buffer("mean_f", torch.zeros(self.raw_dim, dtype=torch.float32))
        self.register_buffer("std_f", torch.ones(self.raw_dim, dtype=torch.float32))

    @torch.no_grad()
    def fit_normalizers(self, X: torch.Tensor) -> None:
        X = X.detach().to("cpu", dtype=torch.float32)
        mean_x = X.mean(dim=0)
        var_x = X.var(dim=0, unbiased=False)
        std_x = torch.sqrt(var_x + 1e-6)

        self.mean_x.copy_(mean_x)
        self.std_x.copy_(std_x)

        raw = self._raw_features(X)
        mean_f = raw.mean(dim=0)
        var_f = raw.var(dim=0, unbiased=False)
        std_f = torch.sqrt(var_f + 1e-6)
        std_f = torch.clamp(std_f, min=1e-4)

        self.mean_f.copy_(mean_f)
        self.std_f.copy_(std_f)

    def _raw_features(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=torch.float32)
        Xn = (X - self.mean_x) / self.std_x
        X2 = Xn * Xn

        feats = [Xn, X2]

        if self.m_relu > 0:
            z = torch.relu(Xn @ self.R_relu)
            feats.append(z)

        if self.m_rff > 0:
            z = Xn @ self.R_rff
            z = z + self.b_rff
            rff = torch.cos(z) * math.sqrt(2.0 / float(self.m_rff))
            feats.append(rff)

        return torch.cat(feats, dim=1)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        raw = self._raw_features(X)
        return (raw - self.mean_f) / self.std_f


class _RFClassifier(nn.Module):
    def __init__(self, extractor: _RandomFeatureExtractor, num_classes: int):
        super().__init__()
        self.extractor = extractor
        self.num_classes = int(num_classes)
        self.head = nn.Linear(self.extractor.raw_dim, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extractor.features(x)
        return self.head(f)


def _gather_from_loader(loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict) and "inputs" in batch and "targets" in batch:
            x, y = batch["inputs"], batch["targets"]
        else:
            raise TypeError("Unsupported batch format")
        xs.append(x.detach().to("cpu"))
        ys.append(y.detach().to("cpu"))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0).to(dtype=torch.long)
    return X, y


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        try:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        except Exception:
            pass

        if metadata is None:
            metadata = {}

        X_train, y_train = _gather_from_loader(train_loader)
        if val_loader is not None:
            X_val, y_val = _gather_from_loader(val_loader)
        else:
            X_val, y_val = None, None

        input_dim = int(metadata.get("input_dim", X_train.shape[1]))
        num_classes = int(metadata.get("num_classes", int(y_train.max().item()) + 1))
        param_limit = int(metadata.get("param_limit", 1_000_000))

        fdim_max = max(16, (param_limit // max(1, num_classes)) - 1)

        desired_total_random = 1024
        base_dim = 2 * input_dim
        rem = max(0, min(desired_total_random, fdim_max - base_dim))
        m_relu = int(round(rem * 0.625))
        m_rff = int(rem - m_relu)

        extractor = _RandomFeatureExtractor(input_dim=input_dim, m_relu=m_relu, m_rff=m_rff, seed=0)
        extractor.fit_normalizers(X_train)

        with torch.no_grad():
            f_train = extractor.features(X_train)
            if X_val is not None:
                f_val = extractor.features(X_val)

        N = f_train.shape[0]
        ones_train = torch.ones((N, 1), dtype=f_train.dtype)
        phi_train = torch.cat([f_train, ones_train], dim=1)

        Y = torch.zeros((N, num_classes), dtype=torch.float32)
        Y.scatter_(1, y_train.view(-1, 1), 1.0)

        Gram = (phi_train.T @ phi_train) / float(N)
        B = (phi_train.T @ Y) / float(N)

        lams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
        best_lam = None
        best_W = None
        best_val_acc = -1.0

        I = torch.eye(Gram.shape[0], dtype=Gram.dtype)

        if X_val is not None:
            phi_val = torch.cat([f_val, torch.ones((f_val.shape[0], 1), dtype=f_val.dtype)], dim=1)

            for lam in lams:
                lam_eff = float(lam)
                W = None
                for _ in range(5):
                    A = Gram + lam_eff * I
                    L, info = torch.linalg.cholesky_ex(A)
                    if int(info.item()) == 0:
                        W = torch.cholesky_solve(B, L)
                        break
                    lam_eff *= 10.0
                if W is None:
                    continue

                logits_val = phi_val @ W
                acc = _accuracy(logits_val, y_val)
                if acc > best_val_acc + 1e-8:
                    best_val_acc = acc
                    best_lam = lam_eff
                    best_W = W
        else:
            lam_eff = 1e-2
            W = None
            for _ in range(6):
                A = Gram + lam_eff * I
                L, info = torch.linalg.cholesky_ex(A)
                if int(info.item()) == 0:
                    W = torch.cholesky_solve(B, L)
                    break
                lam_eff *= 10.0
            best_lam = lam_eff
            best_W = W

        model = _RFClassifier(extractor=extractor, num_classes=num_classes)

        if best_W is None:
            with torch.no_grad():
                model.head.weight.zero_()
                model.head.bias.zero_()
            model.eval()
            return model

        with torch.no_grad():
            Ww = best_W[:-1, :].T.contiguous()
            Wb = best_W[-1, :].contiguous()
            model.head.weight.copy_(Ww)
            model.head.bias.copy_(Wb)

        if X_val is not None:
            head = model.head
            head.train()

            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            opt = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=1e-4)
            max_epochs = 60
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

            best_state = {
                "w": head.weight.detach().clone(),
                "b": head.bias.detach().clone(),
            }
            best_acc = best_val_acc
            patience = 10
            bad = 0

            for _ in range(max_epochs):
                opt.zero_grad(set_to_none=True)
                logits = head(f_train)
                loss = criterion(logits, y_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                opt.step()
                scheduler.step()

                head.eval()
                with torch.no_grad():
                    acc = _accuracy(head(f_val), y_val)
                head.train()

                if acc > best_acc + 1e-6:
                    best_acc = acc
                    best_state["w"] = head.weight.detach().clone()
                    best_state["b"] = head.bias.detach().clone()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            with torch.no_grad():
                head.weight.copy_(best_state["w"])
                head.bias.copy_(best_state["b"])

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > param_limit:
            safe_model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )
            safe_params = sum(p.numel() for p in safe_model.parameters() if p.requires_grad)
            if safe_params <= param_limit:
                model = safe_model

        model.eval()
        return model