import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2, use_batchnorm=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
            )
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out += residual
        out = self.dropout2(F.relu(out))
        return out

class EfficientNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Carefully designed architecture to maximize capacity within 2.5M params
        hidden_dims = [512, 512, 512, 512, 512, 512]  # 6 layers
        dropout_rates = [0.3, 0.3, 0.3, 0.2, 0.2, 0.1]
        
        layers = OrderedDict()
        
        # Input layer
        layers['fc0'] = nn.Linear(input_dim, hidden_dims[0])
        layers['bn0'] = nn.BatchNorm1d(hidden_dims[0])
        layers['relu0'] = nn.ReLU()
        layers['dropout0'] = nn.Dropout(0.3)
        
        # Residual blocks
        for i in range(len(hidden_dims) - 1):
            layers[f'res{i+1}'] = ResidualBlock(
                hidden_dims[i], 
                hidden_dims[i+1], 
                dropout_rates[i],
                use_batchnorm=True
            )
        
        # Final layers with bottleneck
        layers['bottleneck'] = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Classifier
        layers['classifier'] = nn.Linear(128, num_classes)
        
        self.network = nn.Sequential(layers)
        
    def forward(self, x):
        return self.network(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        
        # Create model
        model = EfficientNet(input_dim, num_classes).to(device)
        
        # Verify parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {params:,} (limit: {param_limit:,})")
        
        # Enhanced training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use AdamW with weight decay for better regularization
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0005,
            betas=(0.9, 0.999)
        )
        
        # Combined schedulers
        scheduler1 = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        scheduler2 = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=10, 
            verbose=False
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(200):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Update schedulers
            scheduler1.step()
            scheduler2.step(val_acc)
            
            # Early stopping with patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress occasionally
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return model