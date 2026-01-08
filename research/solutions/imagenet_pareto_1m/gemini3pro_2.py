import torch
import torch.nn as nn
import torch.optim as optim
import math

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # 1. Define Architecture Components
        class ResBlock(nn.Module):
            def __init__(self, dim, dropout=0.25):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )

            def forward(self, x):
                return x + self.net(x)

        class ParamEfficientModel(nn.Module):
            def __init__(self, in_d, out_d, h_d):
                super().__init__()
                # Normalize input features as they might not be standardized
                self.input_norm = nn.BatchNorm1d(in_d)
                
                # Projection to hidden dimension
                self.entry = nn.Sequential(
                    nn.Linear(in_d, h_d),
                    nn.BatchNorm1d(h_d),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                # Residual blocks for depth
                self.blocks = nn.Sequential(
                    ResBlock(h_d),
                    ResBlock(h_d)
                )
                
                # Classification head
                self.head = nn.Linear(h_d, out_d)

            def forward(self, x):
                x = self.input_norm(x)
                x = self.entry(x)
                x = self.blocks(x)
                return self.head(x)

        # 2. Calculate Optimal Hidden Dimension
        # Constraints: Maximize width within 1M parameter budget
        # Params calculation:
        # InputBN: 2*in
        # Entry: (in*h + h) [Linear] + 2*h [BN] = h(in + 3)
        # ResBlock (x2): 2 * [ (h*h + h) [Linear] + 2*h [BN] ] = 2 * (h^2 + 3h) = 2h^2 + 6h
        # Head: h*out + out
        # Total: 2h^2 + (in + out + 9)h + (2*in + out) <= 1,000,000
        
        target_limit = 998000  # Safety buffer
        a = 2
        b = input_dim + num_classes + 9
        c_fixed = 2 * input_dim + num_classes
        c = c_fixed - target_limit
        
        # Quadratic formula: ax^2 + bx + c = 0 -> x = (-b + sqrt(b^2 - 4ac)) / 2a
        delta = b**2 - 4 * a * c
        hidden_dim = int((-b + math.sqrt(delta)) / (2 * a))
        
        # Initialize model
        model = ParamEfficientModel(input_dim, num_classes, hidden_dim).to(device)
        
        # Safety check: Reduce width if we accidentally exceed limit due to rounding
        while sum(p.numel() for p in model.parameters() if p.requires_grad) > 1000000:
            hidden_dim -= 4
            model = ParamEfficientModel(input_dim, num_classes, hidden_dim).to(device)

        # 3. Setup Training
        # Use AdamW with weight decay for regularization on small dataset
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 80
        # OneCycleLR typically converges faster and better for fixed epoch budgets
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.002, 
            epochs=epochs, 
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        # Tracking best model
        best_acc = 0.0
        best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # 4. Training Loop
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Augmentation: Injection of random noise to prevent overfitting
                # Since inputs are features, this acts as data augmentation
                noise = torch.randn_like(inputs) * 0.05
                inputs_aug = inputs + noise
                
                optimizer.zero_grad()
                outputs = model(inputs_aug)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validation
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
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Restore best model
        model.load_state_dict(best_weights)
        
        return model