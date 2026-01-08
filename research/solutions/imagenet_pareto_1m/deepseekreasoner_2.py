import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.norm1(x)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + identity

class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        hidden_dims = [512, 512, 384, 256]
        
        layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(ResidualBlock(prev_dim, h_dim, dropout=0.1 if i < 2 else 0.05))
            prev_dim = h_dim
        
        self.features = nn.Sequential(*layers)
        
        self.final_norm = nn.BatchNorm1d(prev_dim)
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.features(x)
        x = self.final_norm(x)
        x = self.classifier(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        if metadata is None:
            metadata = {}
        
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        model = CustomModel(input_dim, num_classes).to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
        
        best_acc = 0
        best_model = None
        patience_counter = 0
        max_patience = 15
        
        num_epochs = 80
        
        for epoch in range(num_epochs):
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
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if best_model is not None:
            model.load_state_dict(best_model)
        
        model.eval()
        return model