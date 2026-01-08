import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        device = metadata.get("device", "cpu")
        
        class EfficientNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(EfficientNet, self).__init__()
                
                # Bottleneck architecture with depthwise separable convolutions
                # Start with linear layer to compress input
                self.features = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    # Bottleneck block 1
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    # Bottleneck block 2 with residual connection
                    ResidualBlock(256, 256),
                    
                    # Bottleneck block 3
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    # Final compression
                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
                
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
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_features, out_features):
                super(ResidualBlock, self).__init__()
                self.linear1 = nn.Linear(in_features, out_features)
                self.bn1 = nn.BatchNorm1d(out_features)
                self.relu = nn.ReLU(inplace=True)
                self.linear2 = nn.Linear(out_features, out_features)
                self.bn2 = nn.BatchNorm1d(out_features)
                self.dropout = nn.Dropout(0.2)
                
                # Skip connection
                self.skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
            
            def forward(self, x):
                identity = self.skip(x)
                out = self.linear1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.linear2(out)
                out = self.bn2(out)
                out += identity
                out = self.relu(out)
                return out
        
        # Create model
        model = EfficientNet(input_dim, num_classes).to(device)
        
        # Verify parameter count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params > param_limit:
            # Reduce model size dynamically if needed
            return self._create_simpler_model(input_dim, num_classes, device, param_limit)
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
        # Training loop with early stopping
        best_val_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0
        num_epochs = 100
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
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
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            val_acc = 100. * correct / total
            scheduler.step()
            
            # Early stopping and model selection
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model
    
    def _create_simpler_model(self, input_dim, num_classes, device, param_limit):
        """Fallback model if primary model exceeds parameter limit"""
        class SimpleNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(SimpleNet, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    nn.Linear(384, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, num_classes)
                )
                
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
                return self.net(x)
        
        model = SimpleNet(input_dim, num_classes).to(device)
        
        # Verify it's under limit
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # If still over limit, create minimal model
        if trainable_params > param_limit:
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ).to(device)
        
        return model