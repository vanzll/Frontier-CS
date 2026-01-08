import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ParetoEqualNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, dropout=0.2):
        super().__init__()
        # Initial projection
        self.entry = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.res1 = ResBlock(hidden_dim, dropout)
        self.res2 = ResBlock(hidden_dim, dropout)
        
        # Output head
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.entry(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # 1. Parse Metadata
        if metadata is None:
            metadata = {}
            
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 1000000)
        device = metadata.get("device", "cpu")
        
        # 2. Determine Architecture (Maximize H within budget)
        # Architecture:
        # Entry: BN(In) + Lin(In,H) + BN(H) -> 2*In + In*H + H + 2*H
        # Res1: Lin(H,H) + BN(H)            -> H^2 + H + 2*H
        # Res2: Lin(H,H) + BN(H)            -> H^2 + H + 2*H
        # Head: Lin(H, Out)                 -> H*Out + Out
        # Total approx: 2H^2 + (In + Out + 9)H + (2*In + Out)
        
        # Initial estimate
        a = 2
        b = input_dim + num_classes + 9
        c = (2 * input_dim + num_classes) - param_limit
        delta = b**2 - 4*a*c
        h_est = int((-b + math.sqrt(delta)) / (2*a))
        
        # Refine H to strictly satisfy constraint
        hidden_dim = h_est
        while True:
            model = ParetoEqualNet(input_dim, num_classes, hidden_dim)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if params < param_limit - 500: # Safety buffer
                break
            hidden_dim -= 4
            
        # 3. Setup Model and Training
        model = ParetoEqualNet(input_dim, num_classes, hidden_dim, dropout=0.4)
        model = model.to(device)
        
        # Optimization
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
        epochs = 120
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        # Mixup parameters
        mixup_alpha = 0.4
        
        best_acc = 0.0
        best_model_state = None
        
        # 4. Training Loop
        for epoch in range(epochs):
            model.train()
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Mixup Augmentation
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(inputs.size(0)).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets[index])
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # 5. Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            acc = correct / total
            
            # Checkpoint
            if acc >= best_acc:
                best_acc = acc
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # 6. Finalize
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Ensure model is on the correct device
        model.to(device)
        model.eval()
        
        return model