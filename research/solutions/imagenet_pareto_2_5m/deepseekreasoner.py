import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class EfficientClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=128):
        super().__init__()
        # Architecture designed to stay under 2.5M params
        self.expand = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks with bottleneck structure
        self.block1 = ResidualBlock(512, 768, dropout_rate=0.3)
        self.block2 = ResidualBlock(768, 1024, dropout_rate=0.3)
        self.block3 = ResidualBlock(1024, 768, dropout_rate=0.3)
        self.block4 = ResidualBlock(768, 512, dropout_rate=0.2)
        
        # Final layers
        self.final = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
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
        x = self.expand(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        device = torch.device(metadata["device"] if metadata and "device" in metadata else "cpu")
        num_classes = metadata["num_classes"] if metadata and "num_classes" in metadata else 128
        input_dim = metadata["input_dim"] if metadata and "input_dim" in metadata else 384
        
        model = EfficientClassifier(input_dim, num_classes).to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_limit = metadata["param_limit"] if metadata and "param_limit" in metadata else 2500000
        
        if total_params > param_limit:
            # Reduce model size if needed
            model = self._create_smaller_model(input_dim, num_classes, param_limit).to(device)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = 10
        
        num_epochs = 80
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
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
            
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def _create_smaller_model(self, input_dim, num_classes, param_limit):
        """Create a smaller model if initial one exceeds parameter limit"""
        class SmallerClassifier(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(384, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(512, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(384, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                return self.net(x)
        
        return SmallerClassifier(input_dim, num_classes)