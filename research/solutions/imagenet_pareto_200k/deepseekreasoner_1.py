import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

class EfficientNet(nn.Module):
    def __init__(self, input_dim=384, num_classes=128):
        super().__init__()
        
        # Bottleneck architecture to maximize capacity within 200K params
        hidden1 = 384  # Keep same as input to preserve information
        hidden2 = 320  # Gradual reduction
        hidden3 = 256  # Further reduction
        hidden4 = 192  # Compact representation
        
        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3 with residual connection
            ResidualBlock(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 4
            nn.Linear(hidden3, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(hidden4, num_classes)
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
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += residual
        return self.activation(out)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        
        # Create model
        model = EfficientNet(input_dim, num_classes).to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params <= 200000, f"Model has {total_params} parameters, exceeds 200K limit"
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
        
        # Training loop
        num_epochs = 100
        best_val_acc = 0.0
        best_model = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
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
            
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
            
            # Early stopping check
            if epoch > 20 and val_acc < 60.0:  # If not learning after 20 epochs
                break
        
        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
        
        return model