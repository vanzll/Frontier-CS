import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2, use_bn=True):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Expanded initial layer
        hidden_dim1 = 1024
        
        # Bottleneck layer for efficiency
        hidden_dim2 = 768
        
        # Deep narrow layers
        hidden_dim3 = 512
        hidden_dim4 = 384
        
        # Architecture designed to maximize capacity within 5M params
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            ResidualBlock(hidden_dim1, hidden_dim2, dropout_rate=0.25),
            ResidualBlock(hidden_dim2, hidden_dim3, dropout_rate=0.25),
            ResidualBlock(hidden_dim3, hidden_dim4, dropout_rate=0.3),
            
            # Final projection
            nn.Linear(hidden_dim4, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        device = metadata.get("device", "cpu")
        
        # Create model with parameter budget check
        model = Model(input_dim, num_classes).to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > 5000000:
            # If too large, create a smaller model
            return self._create_smaller_model(input_dim, num_classes, device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        
        # Use AdamW with weight decay for better generalization
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,
            weight_decay=0.01
        )
        
        # Cosine annealing learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
        # Early stopping setup
        best_val_acc = 0.0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        num_epochs = 100
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
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
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def _create_smaller_model(self, input_dim, num_classes, device):
        """Create a smaller model if initial one exceeds parameter budget"""
        class SmallerModel(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                hidden1 = 896
                hidden2 = 512
                hidden3 = 256
                
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.BatchNorm1d(hidden1),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(hidden1, hidden2),
                    nn.BatchNorm1d(hidden2),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    
                    nn.Linear(hidden2, hidden3),
                    nn.BatchNorm1d(hidden3),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(hidden3, num_classes)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = SmallerModel(input_dim, num_classes).to(device)
        return model