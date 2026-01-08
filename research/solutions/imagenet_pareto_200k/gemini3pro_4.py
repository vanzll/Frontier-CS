import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ParetoModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ParetoModel, self).__init__()
        # Optimized architecture for ~195k parameters with input_dim=384, num_classes=128
        # L1: 384->256 (98,560)
        # L2: 256->192 (49,344)
        # L3: 192->144 (27,792)
        # L4: 144->128 (18,560)
        # BN: 1,184
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(192, 144),
            nn.BatchNorm1d(144),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(144, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = torch.device(metadata.get("device", "cpu"))
        
        # Initialize model
        model = ParetoModel(input_dim, num_classes).to(device)
        
        # Strict parameter check and fallback
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        limit = metadata.get("param_limit", 200000)
        
        if param_count > limit:
            # Fallback to single hidden layer if dimensions differ and cause overflow
            # H(I + O + 1) + O <= Limit => H <= (Limit - O) / (I + O + 1)
            max_h = int((limit - num_classes) / (input_dim + num_classes + 1)) - 2
            model = nn.Sequential(
                nn.Linear(input_dim, max_h),
                nn.BatchNorm1d(max_h),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(max_h, num_classes)
            ).to(device)
            
        # Optimization setup
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        epochs = 80
        steps = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.002, epochs=epochs, steps_per_epoch=steps, pct_start=0.3
        )
        
        # EMA setup
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_decay = 0.99
        
        best_acc = 0.0
        best_weights = copy.deepcopy(model.state_dict())
        
        model.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Noise injection
                if epoch < epochs - 10:
                    inputs = inputs + torch.randn_like(inputs) * 0.05
                
                # Mixup regularization
                do_mixup = inputs.size(0) > 1 and np.random.random() < 0.6
                if do_mixup:
                    alpha = 0.4
                    lam = np.random.beta(alpha, alpha)
                    idx = torch.randperm(inputs.size(0), device=device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[idx]
                    target_a, target_b = targets, targets[idx]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # EMA Update
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
            
            # Validation
            if epoch % 5 == 0 or epoch > epochs - 10:
                acc = self.evaluate(ema_model, val_loader, device)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = copy.deepcopy(ema_model.state_dict())
        
        model.load_state_dict(best_weights)
        return model

    def evaluate(self, model, loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        model.train()
        return correct / total if total > 0 else 0