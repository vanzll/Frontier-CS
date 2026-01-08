import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from copy import deepcopy
import time

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.size()
        y = self.fc(x)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features)
            )
        
        self.se = SEBlock(out_features, se_reduction)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out += residual
        return out

class EfficientNet(nn.Module):
    def __init__(self, input_dim=384, num_classes=128):
        super(EfficientNet, self).__init__()
        
        widths = [512, 768, 1024, 768, 512]
        depths = [2, 3, 4, 2, 1]
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, widths[0]),
            nn.BatchNorm1d(widths[0]),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        layers = []
        for i in range(len(widths)):
            in_ch = widths[i-1] if i > 0 else widths[0]
            out_ch = widths[i]
            for j in range(depths[i]):
                layers.append(ResidualBlock(
                    in_ch if j == 0 else out_ch,
                    out_ch,
                    dropout_rate=0.2 - i*0.03
                ))
        
        self.layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(widths[-1]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(widths[-1], num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        device = torch.device(metadata["device"] if metadata and "device" in metadata else "cpu")
        input_dim = metadata["input_dim"] if metadata else 384
        num_classes = metadata["num_classes"] if metadata else 128
        
        model = EfficientNet(input_dim, num_classes)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        target_param = 2500000
        
        if param_count > target_param:
            scale = (target_param / param_count) ** 0.5
            self._scale_model(model, scale)
        
        model.to(device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.05
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        num_epochs = 300
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            avg_loss = total_loss / len(train_loader)
            
            val_acc = self._evaluate(model, val_loader, device)
            
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        return model
    
    def _evaluate(self, model, loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
    
    def _scale_model(self, model, scale):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    new_out_features = max(int(module.out_features * scale), 128)
                    if hasattr(module, 'in_features'):
                        new_in_features = max(int(module.in_features * scale), 128)
                    else:
                        new_in_features = module.weight.size(1)
                    
                    if new_out_features != module.out_features:
                        new_weight = module.weight.data[:new_out_features, :new_in_features].clone()
                        if module.bias is not None:
                            new_bias = module.bias.data[:new_out_features].clone()
                        
                        module.out_features = new_out_features
                        if hasattr(module, 'in_features'):
                            module.in_features = new_in_features
                        
                        module.weight = nn.Parameter(new_weight)
                        if module.bias is not None:
                            module.bias = nn.Parameter(new_bias)