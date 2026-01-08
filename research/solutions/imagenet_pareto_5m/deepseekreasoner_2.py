import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import math
from collections import OrderedDict

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        
        # Enhanced architecture with better parameter efficiency
        class EfficientNet(nn.Module):
            def __init__(self, input_dim, num_classes, dropout_rate=0.3):
                super().__init__()
                
                # Initial projection with residual connection
                self.initial_proj = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate)
                )
                
                # Bottleneck blocks with increasing capacity
                self.blocks = nn.ModuleList([
                    self._make_block(512, 512, dropout_rate),  # Block 1
                    self._make_block(512, 512, dropout_rate),  # Block 2
                    self._make_block(512, 768, dropout_rate),  # Block 3
                    self._make_block(768, 768, dropout_rate),  # Block 4
                    self._make_block(768, 1024, dropout_rate), # Block 5
                    self._make_block(1024, 1024, dropout_rate), # Block 6
                ])
                
                # Final classification head
                self.final_norm = nn.BatchNorm1d(1024)
                self.final_dropout = nn.Dropout(dropout_rate)
                self.classifier = nn.Linear(1024, num_classes)
                
                # Initialize weights
                self._initialize_weights()
            
            def _make_block(self, in_dim, out_dim, dropout_rate):
                return nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=False),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                )
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                x = self.initial_proj(x)
                
                for block in self.blocks:
                    x = block(x)
                
                x = self.final_norm(x)
                x = self.final_dropout(x)
                x = self.classifier(x)
                return x
        
        # Create model and move to device
        model = EfficientNet(input_dim, num_classes)
        model = model.to(device)
        
        # Calculate parameters and ensure constraint
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        
        if total_params > param_limit:
            # Scale down if needed
            print(f"Model exceeds parameter limit ({param_limit:,}), creating smaller model")
            model = self._create_smaller_model(input_dim, num_classes, param_limit, device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with warmup
        warmup_epochs = 5
        total_epochs = 100
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=(device != "cpu"))
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(total_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward
                with autocast(enabled=(device != "cpu")):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Mixed precision backward
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            train_loss = train_loss / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    with autocast(enabled=(device != "cpu")):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / val_total
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1:3d} | '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        model.eval()
        
        return model
    
    def _create_smaller_model(self, input_dim, num_classes, param_limit, device):
        """Create a smaller model if the initial one exceeds the parameter limit"""
        class CompactNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                
                self.features = nn.Sequential(
                    nn.Linear(input_dim, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    
                    nn.Linear(384, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                    
                    nn.Linear(512, 640),
                    nn.BatchNorm1d(640),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    nn.Linear(640, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                )
                
                self.classifier = nn.Linear(512, num_classes)
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = CompactNet(input_dim, num_classes)
        model = model.to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Compact model parameters: {total_params:,}")
        
        if total_params > param_limit:
            # Emergency fallback - minimal model
            class MinimalNet(nn.Module):
                def __init__(self, input_dim, num_classes):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            model = MinimalNet(input_dim, num_classes)
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Minimal model parameters: {total_params:,}")
        
        return model