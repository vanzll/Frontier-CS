import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.norm1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual

class EfficientNet(nn.Module):
    def __init__(self, input_dim=384, num_classes=128, param_limit=200000):
        super().__init__()
        # Design parameters to stay under 200K
        hidden1 = 320  # Optimized for parameter efficiency
        hidden2 = 256
        hidden3 = 192
        dropout = 0.3
        
        # Calculate parameters
        params = 0
        params += input_dim * hidden1 + hidden1  # Layer 1
        params += hidden1 * hidden2 + hidden2    # Layer 2
        params += hidden2 * hidden3 + hidden3    # Layer 3
        params += hidden3 * num_classes + num_classes  # Output layer
        params += 2 * hidden1 + 2 * hidden2 + 2 * hidden3  # BatchNorm params
        params += hidden1 + hidden2 + hidden3  # Residual connections
        
        # Adjust if needed
        if params > param_limit:
            # Scale down if needed
            scale = (param_limit / params) ** 0.5
            hidden1 = int(hidden1 * scale)
            hidden2 = int(hidden2 * scale)
            hidden3 = int(hidden3 * scale)
        
        # Build network
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Block 1
        self.block1 = ResidualBlock(input_dim, hidden1, dropout)
        
        # Block 2
        self.block2 = ResidualBlock(hidden1, hidden2, dropout)
        
        # Block 3
        self.block3 = ResidualBlock(hidden2, hidden3, dropout)
        
        # Final layers
        self.final_norm = nn.BatchNorm1d(hidden3)
        self.final_act = nn.ReLU()
        self.final_dropout = nn.Dropout(dropout * 0.5)
        self.classifier = nn.Linear(hidden3, num_classes)
        
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
        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_dropout(x)
        return self.classifier(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        if metadata is None:
            metadata = {}
        
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        
        # Create model
        model = EfficientNet(input_dim, num_classes, param_limit)
        model.to(device)
        
        # Verify parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params > param_limit:
            # Scale down if still over
            model = self._create_smaller_model(input_dim, num_classes, param_limit)
            model.to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        
        # Training loop
        best_acc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
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
            
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            
            # Early stopping check
            if val_acc > best_acc:
                best_acc = val_acc
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
    
    def _create_smaller_model(self, input_dim, num_classes, param_limit):
        """Create a smaller model if initial one exceeds parameter limit"""
        hidden1 = 256
        hidden2 = 192
        hidden3 = 128
        dropout = 0.25
        
        model = nn.Sequential(OrderedDict([
            ('input_norm', nn.BatchNorm1d(input_dim)),
            ('fc1', nn.Linear(input_dim, hidden1)),
            ('bn1', nn.BatchNorm1d(hidden1)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(dropout)),
            
            ('fc2', nn.Linear(hidden1, hidden2)),
            ('bn2', nn.BatchNorm1d(hidden2)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(dropout)),
            
            ('fc3', nn.Linear(hidden2, hidden3)),
            ('bn3', nn.BatchNorm1d(hidden3)),
            ('relu3', nn.ReLU()),
            ('dropout3', nn.Dropout(dropout * 0.5)),
            
            ('classifier', nn.Linear(hidden3, num_classes))
        ]))
        
        return model