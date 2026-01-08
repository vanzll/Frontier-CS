import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import copy

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Identity()
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

class EfficientNet(nn.Module):
    def __init__(self, input_dim, num_classes, param_limit):
        super().__init__()
        
        # Calculate dimensions to stay under parameter limit
        # We'll use a bottleneck architecture
        hidden1 = 1024
        hidden2 = 768
        hidden3 = 512
        hidden4 = 384
        
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # Initial projection
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        # Residual blocks
        self.res1 = ResidualBlock(hidden1, hidden2, 0.2)
        self.res2 = ResidualBlock(hidden2, hidden3, 0.2)
        self.res3 = ResidualBlock(hidden3, hidden4, 0.2)
        
        # Final classifier with bottleneck
        self.fc_final = nn.Linear(hidden4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Verify parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if total_params > param_limit:
            raise ValueError(f"Model exceeds parameter limit: {total_params} > {param_limit}")
    
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
        x = self.bn_input(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2, training=self.training)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.fc_final(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        
        # Create model
        model = EfficientNet(input_dim, num_classes, param_limit).to(device)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Scheduler with warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
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
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
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
                
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100. * train_correct / train_total
                    })
            
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print epoch statistics
            print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_acc:.2f}%, '
                  f'Best Val Acc: {best_val_acc:.2f}%')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final verification
        model.eval()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Final model parameters: {total_params:,}')
        
        if total_params > param_limit:
            raise ValueError(f"Model exceeds parameter limit: {total_params} > {param_limit}")
        
        return model