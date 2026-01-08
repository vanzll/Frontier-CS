import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
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

class EfficientNet(nn.Module):
    def __init__(self, input_dim, num_classes, expansion_factor=4, depth=3, dropout=0.2):
        super().__init__()
        hidden_dim = 512
        
        layers = []
        in_features = input_dim
        
        for i in range(depth):
            out_features = hidden_dim if i == depth-1 else hidden_dim // 2
            layers.append(ResidualBlock(in_features, out_features, dropout))
            in_features = out_features
            
        self.features = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        device = torch.device(metadata["device"] if metadata and "device" in metadata else "cpu")
        input_dim = metadata["input_dim"] if metadata else 384
        num_classes = metadata["num_classes"] if metadata else 128
        
        model = EfficientNet(input_dim, num_classes)
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > 1000000:
            current_params = total_params
            reduction_factor = (1000000 / current_params) ** 0.5
            hidden_dim = int(512 * reduction_factor)
            
            model = EfficientNet(
                input_dim, 
                num_classes,
                depth=2,
                dropout=0.15
            )
            model.to(device)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if total_params > 1000000:
                model = nn.Sequential(
                    nn.Linear(input_dim, 768),
                    nn.BatchNorm1d(768),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(768, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(512, num_classes)
                )
                model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        
        best_acc = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        epochs = 100
        for epoch in range(epochs):
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            scheduler.step()
            
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
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        final_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if final_params > 1000000:
            model = nn.Sequential(
                nn.Linear(input_dim, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(384, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
            model.to(device)
        
        model.eval()
        return model