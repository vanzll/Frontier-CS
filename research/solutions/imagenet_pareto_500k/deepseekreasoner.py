import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        
        class EfficientNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                
                # Stage 1: Initial projection
                self.stem = nn.Sequential(
                    nn.Linear(input_dim, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
                
                # Stage 2: Depthwise separable blocks
                self.block1 = self._make_ds_block(384, 256, 0.2)
                self.block2 = self._make_ds_block(256, 256, 0.2)
                
                # Stage 3: Bottleneck
                self.block3 = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3)
                )
                
                # Stage 4: Final layers
                self.block4 = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3)
                )
                
                # Classifier
                self.classifier = nn.Linear(128, num_classes)
                
                # Initialize weights
                self._initialize_weights()
                
            def _make_ds_block(self, in_dim, out_dim, dropout):
                """Depthwise separable block"""
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(out_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
                
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
                x = self.stem(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                return self.classifier(x)
        
        # Create and verify model size
        model = EfficientNet(input_dim, num_classes).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Ensure we're under the limit with some safety margin
        if total_params > param_limit:
            # Scale down model if needed
            model = self._create_smaller_model(input_dim, num_classes, param_limit).to(device)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
        # Early stopping
        best_val_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        num_epochs = 100
        
        for epoch in range(num_epochs):
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
                
                # Gradient clipping
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
            
            val_acc = val_correct / val_total
            
            # Early stopping logic
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
    
    def _create_smaller_model(self, input_dim, num_classes, param_limit):
        """Create an even smaller model if needed"""
        class CompactNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 192),
                    nn.BatchNorm1d(192),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    nn.Linear(192, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, num_classes)
                )
                
                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                return self.net(x)
        
        return CompactNet(input_dim, num_classes)