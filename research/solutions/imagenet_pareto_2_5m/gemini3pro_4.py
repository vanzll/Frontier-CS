import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=590, num_blocks=3):
        super(ResNetMLP, self).__init__()
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.input_relu = nn.ReLU()
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate=0.2) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # Initialize model
        # Calculation for parameters:
        # Width 590, 3 Blocks
        # Input: 384*590 + 590 + BN(590*2) = ~228k
        # Block: 2*(590*590 + 590 + BN(590*2)) = ~700k
        # 3 Blocks = ~2.1M
        # Output: 590*128 + 128 = ~75k
        # Total ~2.4M < 2.5M Limit
        model = ResNetMLP(input_dim, num_classes, hidden_dim=590, num_blocks=3)
        model = model.to(device)
        
        # Optimization setup
        # AdamW with weight decay for regularization
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Label smoothing helps with generalization on synthetic data
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        epochs = 120
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_acc = 0.0
        best_weights = copy.deepcopy(model.state_dict())
        
        # Mixup parameters
        mixup_alpha = 0.4
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Apply Mixup Augmentation
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(inputs.size(0)).to(device)
                    
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    y_a, y_b = targets, targets[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(inputs)
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
            if acc >= best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        
        # Restore best model weights
        model.load_state_dict(best_weights)
        
        return model