import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        
        # Calculate optimal architecture dimensions to maximize capacity within 2.5M params
        # Using a residual MLP with bottleneck structure for efficiency
        base_dim = 1024
        bottleneck_factor = 0.25
        
        class EfficientResNet(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Initial projection
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, base_dim),
                    nn.BatchNorm1d(base_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Residual blocks with gradual dimension changes
                self.block1 = ResidualBlock(base_dim, int(base_dim * bottleneck_factor))
                self.block2 = ResidualBlock(int(base_dim * bottleneck_factor), base_dim)
                self.block3 = ResidualBlock(base_dim, int(base_dim * bottleneck_factor))
                self.block4 = ResidualBlock(int(base_dim * bottleneck_factor), base_dim)
                self.block5 = ResidualBlock(base_dim, int(base_dim * bottleneck_factor))
                self.block6 = ResidualBlock(int(base_dim * bottleneck_factor), base_dim)
                
                # Final classifier with bottleneck
                bottleneck = 512
                self.classifier = nn.Sequential(
                    nn.Linear(base_dim, bottleneck),
                    nn.BatchNorm1d(bottleneck),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(bottleneck, num_classes)
                )
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
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
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                x = self.block5(x)
                x = self.block6(x)
                x = self.classifier(x)
                return x
        
        model = EfficientResNet().to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > param_limit:
            # Scale down if needed
            scale = math.sqrt(param_limit / total_params)
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    new_size = max(1, int(param.shape[0] * scale))
                    if name == 'classifier.0.weight':
                        new_size = min(new_size, param.shape[1])
                    
                    if new_size != param.shape[0]:
                        # Replace layer with smaller one
                        parts = name.split('.')
                        obj = model
                        for part in parts[:-1]:
                            if part.isdigit():
                                obj = obj[int(part)]
                            else:
                                obj = getattr(obj, part)
                        
                        old_layer = getattr(obj, parts[-1][:-7])
                        in_features = old_layer.in_features
                        out_features = old_layer.out_features
                        
                        if parts[-1].startswith('fc1'):
                            new_layer = nn.Linear(in_features, new_size)
                        elif parts[-1].startswith('fc2'):
                            new_layer = nn.Linear(in_features, new_size)
                        elif parts[-1] == 'weight' and hasattr(old_layer, 'in_features'):
                            if parts[-2] == '0':  # classifier first layer
                                new_layer = nn.Linear(in_features, new_size)
                            else:
                                new_layer = nn.Linear(in_features, out_features)
                        
                        nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')
                        if new_layer.bias is not None:
                            nn.init.constant_(new_layer.bias, 0)
                        
                        setattr(obj, parts[-1][:-7], new_layer)
            
            model = model.to(device)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.05
        )
        
        # Mixed schedulers
        scheduler1 = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        scheduler2 = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        best_state_dict = None
        
        num_epochs = 100
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
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
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * correct / total
            val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler1.step()
            scheduler2.step(val_acc)
            
            # Early stopping with patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
            
            # Dynamic dropout adjustment based on validation performance
            if val_acc > 90.0:
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = min(0.05, module.p * 0.95)
        
        # Load best model
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        
        # Final validation check
        model.eval()
        return model