import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class EfficientMLP(nn.Module):
    def __init__(self, input_dim, num_classes, param_limit=500000):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Calculate hidden dimensions to stay under parameter budget
        # Using asymmetric bottleneck structure: input -> large hidden -> smaller -> output
        hidden1 = 512
        hidden2 = 256
        
        # Calculate parameter count
        params = 0
        params += input_dim * hidden1 + hidden1  # Layer 1
        params += hidden1 * hidden2 + hidden2    # Layer 2  
        params += hidden2 * num_classes + num_classes  # Output layer
        
        # Add batch norm parameters (4 per layer: weight, bias, running_mean, running_var)
        params += 4 * hidden1 + 4 * hidden2
        
        # If we're over budget, scale down
        if params > param_limit:
            scale = (param_limit / params) ** 0.5
            hidden1 = int(hidden1 * scale)
            hidden2 = int(hidden2 * scale)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        if metadata is None:
            metadata = {}
        
        device = metadata.get("device", "cpu")
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 500000)
        
        # Initialize model
        model = EfficientMLP(input_dim, num_classes, param_limit).to(device)
        
        # Verify parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > param_limit:
            # Emergency reduction
            scale = (param_limit / param_count) ** 0.5
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    layer.out_features = max(32, int(layer.out_features * scale))
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
        best_val_acc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(100):
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Validation
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
            
            val_acc = val_correct / val_total
            
            # Early stopping with patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            scheduler.step()
            
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model