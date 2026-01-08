import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class Model(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_blocks, dropout=0.2):
        super().__init__()
        # Initial BN to normalize input features from synthetic source
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Projection to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(hidden_dim, dropout))
        self.body = nn.Sequential(*layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_bn(x)
        x = self.proj(x)
        x = self.body(x)
        x = self.head(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 1000000)
        device = metadata.get("device", "cpu")
        
        # Determine model configuration to maximize usage of parameter budget
        # We target ~98% of the limit to be safe but efficient
        target_params = int(param_limit * 0.98)
        num_blocks = 3  # Depth chosen based on efficiency analysis
        
        # Binary search for maximum feasible hidden dimension
        low = 64
        high = 2048
        best_dim = 128
        
        while low <= high:
            mid = (low + high) // 2
            # Temporarily instantiate to count parameters
            temp_model = Model(input_dim, num_classes, mid, num_blocks)
            cnt = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
            
            if cnt <= target_params:
                best_dim = mid
                low = mid + 1
            else:
                high = mid - 1
                
        # Instantiate the final model with optimized width
        model = Model(input_dim, num_classes, best_dim, num_blocks, dropout=0.3).to(device)
        
        # Optimization setup
        # Higher weight decay for regularization on small dataset
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-2)
        epochs = 80 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        best_model_state = copy.deepcopy(model.state_dict())
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Mixup Augmentation
                # Essential for small dataset (16 samples/class) to prevent overfitting
                if np.random.random() < 0.7: 
                    alpha = 0.4
                    lam = np.random.beta(alpha, alpha)
                    idx = torch.randperm(inputs.size(0)).to(device)
                    
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[idx]
                    y_a, y_b = targets, targets[idx]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
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
            
            val_acc = correct / total
            # Save best model
            if val_acc >= best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        
        # Return best model
        model.load_state_dict(best_model_state)
        return model