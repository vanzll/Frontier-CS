import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
import math
from collections import OrderedDict

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        
        # Design an efficient model within 5M parameters
        # Use depthwise separable convolutions (1D) and residual connections
        # Optimized for 384-dimensional feature vectors
        class EfficientNet(nn.Module):
            def __init__(self, input_dim=384, num_classes=128):
                super().__init__()
                self.input_dim = input_dim
                self.num_classes = num_classes
                
                # Initial projection
                hidden1 = 512
                self.stem = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.BatchNorm1d(hidden1),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
                
                # Depthwise separable blocks (more efficient than dense layers)
                self.blocks = nn.ModuleList()
                block_channels = [512, 512, 384, 384, 256, 256]
                dropout_rates = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1]
                
                for in_ch, out_ch, drop_rate in zip(
                    [hidden1] + block_channels[:-1], 
                    block_channels, 
                    dropout_rates
                ):
                    block = DepthwiseSeparableBlock(
                        in_ch, out_ch, 
                        expand_ratio=2,
                        dropout_rate=drop_rate
                    )
                    self.blocks.append(block)
                
                # Final layers
                final_hidden = 512
                self.final = nn.Sequential(
                    nn.Linear(block_channels[-1], final_hidden),
                    nn.BatchNorm1d(final_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(final_hidden, final_hidden // 2),
                    nn.BatchNorm1d(final_hidden // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.05),
                )
                
                self.classifier = nn.Linear(final_hidden // 2, num_classes)
                
                # Initialize weights
                self._initialize_weights()
                
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
                for block in self.blocks:
                    x = block(x)
                x = self.final(x)
                return self.classifier(x)
        
        class DepthwiseSeparableBlock(nn.Module):
            def __init__(self, in_channels, out_channels, expand_ratio=2, dropout_rate=0.1):
                super().__init__()
                expanded_channels = in_channels * expand_ratio
                
                self.conv = nn.Sequential(
                    # Pointwise expansion
                    nn.Linear(in_channels, expanded_channels),
                    nn.BatchNorm1d(expanded_channels),
                    nn.ReLU(inplace=True),
                    
                    # Depthwise convolution (simulated with 1x1 conv)
                    nn.Linear(expanded_channels, expanded_channels),
                    nn.BatchNorm1d(expanded_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                    
                    # Pointwise projection
                    nn.Linear(expanded_channels, out_channels),
                    nn.BatchNorm1d(out_channels),
                )
                
                self.shortcut = nn.Sequential()
                if in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.BatchNorm1d(out_channels),
                    )
                
                self.activation = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                identity = self.shortcut(x)
                out = self.conv(x)
                out += identity
                out = self.activation(out)
                out = self.dropout(out)
                return out
        
        # Create model
        model = EfficientNet(input_dim, num_classes).to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        if total_params > param_limit:
            raise ValueError(f"Model exceeds parameter limit: {total_params:,} > {param_limit:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,
            weight_decay=0.05
        )
        
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=50,
            eta_min=1e-6
        )
        
        # Mixed precision training (emulated on CPU)
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # Disabled for CPU
        
        # Training loop
        num_epochs = 100
        best_acc = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=False):
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
            
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            scheduler.step()
            
            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final validation
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
        
        final_val_acc = 100. * val_correct / val_total
        print(f"Final validation accuracy: {final_val_acc:.2f}%")
        
        return model