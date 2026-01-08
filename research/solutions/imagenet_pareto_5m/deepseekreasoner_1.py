import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.norm1 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm2 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.norm1(x)
        out = F.gelu(out)
        out = self.linear1(out)
        
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        out += identity
        return out

class Model(nn.Module):
    def __init__(self, input_dim=384, num_classes=128, param_limit=5000000):
        super().__init__()
        
        # Calculate dimensions to stay under 5M parameters
        # Using a pyramid structure with residual blocks
        hidden_dims = [1024, 896, 768, 640, 512, 384, 256]
        
        # Adjust to stay under parameter limit
        while True:
            self.blocks = nn.ModuleList()
            current_dim = input_dim
            
            # Build network
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(ResidualBlock(current_dim, hidden_dim))
                current_dim = hidden_dim
            
            # Final layers
            final_layers = [
                nn.BatchNorm1d(current_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(current_dim, num_classes)
            ]
            
            self.blocks = nn.ModuleList(layers)
            self.final = nn.Sequential(*final_layers)
            
            # Calculate parameters
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            if total_params <= param_limit:
                break
            else:
                # Reduce hidden dimensions proportionally
                hidden_dims = [int(d * 0.9) for d in hidden_dims]
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        if metadata is None:
            metadata = {}
        
        device = metadata.get('device', 'cpu')
        num_classes = metadata.get('num_classes', 128)
        input_dim = metadata.get('input_dim', 384)
        param_limit = metadata.get('param_limit', 5000000)
        
        # Initialize model
        model = Model(input_dim=input_dim, num_classes=num_classes, param_limit=param_limit)
        model.to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > param_limit:
            # Emergency reduction - create simpler model
            model = nn.Sequential(
                nn.Linear(input_dim, 1536),
                nn.BatchNorm1d(1536),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1536, 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 768),
                nn.BatchNorm1d(768),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(768, num_classes)
            ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
        
        # Early stopping
        best_val_acc = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        # Training loop
        num_epochs = 100
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            
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
            
            val_acc = 100.0 * val_correct / val_total
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping and model checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model