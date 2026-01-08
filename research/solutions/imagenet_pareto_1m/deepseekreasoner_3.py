import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from copy import deepcopy

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.stride = stride
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual
        return out

class EfficientMLP(nn.Module):
    def __init__(self, input_dim=384, num_classes=128, expansion=4):
        super().__init__()
        # Stage 1: Initial projection
        hidden1 = 512
        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Stage 2: Bottleneck residual blocks
        hidden2 = 384
        self.stage2 = nn.Sequential(
            ResidualBlock(hidden1, hidden2, 0.1),
            ResidualBlock(hidden2, hidden2, 0.1),
            ResidualBlock(hidden2, hidden2, 0.1),
        )
        
        # Stage 3: Deeper processing
        hidden3 = 320
        self.stage3 = nn.Sequential(
            ResidualBlock(hidden2, hidden3, 0.15),
            ResidualBlock(hidden3, hidden3, 0.15),
            ResidualBlock(hidden3, hidden3, 0.15),
        )
        
        # Stage 4: Final processing
        hidden4 = 256
        self.stage4 = nn.Sequential(
            ResidualBlock(hidden3, hidden4, 0.2),
            ResidualBlock(hidden4, hidden4, 0.2),
        )
        
        # Head
        self.final_bn = nn.BatchNorm1d(hidden4)
        self.head = nn.Linear(hidden4, num_classes)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_bn(x)
        x = F.relu(x)
        x = self.head(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
            
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        param_limit = metadata.get("param_limit", 1000000)
        
        # Create model
        model = EfficientMLP(input_dim, num_classes)
        model.to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > param_limit:
            # Fallback to smaller model if initial design is too large
            return self._create_smaller_model(input_dim, num_classes, device, param_limit)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        
        # Training parameters
        num_epochs = 150
        best_acc = 0.0
        best_model = None
        patience = 20
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # Early stopping and model selection
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stop if no improvement for patience epochs
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
        
        model.eval()
        return model
    
    def _create_smaller_model(self, input_dim, num_classes, device, param_limit):
        """Create a smaller model that definitely fits within parameter limit"""
        class SmallMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(384, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = SmallMLP(input_dim, num_classes)
        model.to(device)
        
        # Quick training for smaller model
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001)
        
        for epoch in range(50):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model