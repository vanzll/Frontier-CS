import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class EfficientMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[384, 256, 192], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata=None):
        if metadata is None:
            metadata = {}
            
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 500000)
        device = metadata.get("device", "cpu")
        
        # Design model within parameter budget
        hidden_configs = [
            [384, 256, 192],
            [384, 256, 256, 128],
            [512, 256, 128],
        ]
        
        best_model = None
        for hidden_dims in hidden_configs:
            model = EfficientMLP(input_dim, num_classes, hidden_dims=hidden_dims, dropout=0.3)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if params <= param_limit:
                best_model = model
                break
                
        if best_model is None:
            # Fallback to minimal model
            best_model = EfficientMLP(input_dim, num_classes, hidden_dims=[256, 128], dropout=0.2)
        
        model = best_model.to(device)
        
        # Verify parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params > param_limit:
            # Emergency fallback
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            ).to(device)
        
        # Training setup
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        num_epochs = 50
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
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
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # Early stopping with patience
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
        
        # Final validation to ensure model is properly trained
        model.eval()
        return model