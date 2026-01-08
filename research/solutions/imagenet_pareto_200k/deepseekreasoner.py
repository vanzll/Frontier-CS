import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from typing import Optional
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, dropout=0.1):
        super().__init__()
        self.stride = stride
        self.use_skip = (in_dim != out_dim) or (stride != 1)
        
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.act1 = nn.GELU()
        self.conv1 = nn.Linear(in_dim, out_dim)
        
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Linear(out_dim, out_dim)
        
        if self.use_skip:
            self.skip = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if stride == 1 else nn.Identity()
            )
        
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.use_skip:
            identity = self.skip(x)
        
        return out + identity

class EfficientNet(nn.Module):
    def __init__(self, input_dim=384, num_classes=128, dropout_rate=0.2):
        super().__init__()
        
        # Initial projection - reduce dimension gradually
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Depthwise separable blocks with residual connections
        self.blocks = nn.Sequential(
            ResidualBlock(256, 256, dropout=0.1),
            ResidualBlock(256, 256, dropout=0.1),
            ResidualBlock(256, 192, dropout=0.15),
            ResidualBlock(192, 192, dropout=0.15),
            ResidualBlock(192, 128, dropout=0.2),
            ResidualBlock(128, 128, dropout=0.2),
        )
        
        # Head with bottleneck
        self.head = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> nn.Module:
        device = metadata.get("device", "cpu")
        num_classes = metadata["num_classes"]
        input_dim = metadata["input_dim"]
        param_limit = metadata["param_limit"]
        
        # Create model with parameter constraint
        model = None
        for dropout in [0.2, 0.25, 0.3]:
            model = EfficientNet(input_dim, num_classes, dropout_rate=dropout)
            params = count_parameters(model)
            if params <= param_limit:
                break
        
        if model is None:
            # Fallback to simpler model
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 192),
                nn.BatchNorm1d(192),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(192, num_classes)
            )
        
        model.to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0015,
            weight_decay=0.05
        )
        
        # Schedulers
        scheduler1 = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        scheduler2 = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Early stopping
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        epochs = 120
        
        for epoch in range(epochs):
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
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Update schedulers
            scheduler1.step()
            scheduler2.step(val_acc)
            
            # Early stopping logic
            if val_acc > best_val_acc + 0.1:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final validation to ensure model is in eval mode
        model.eval()
        return model