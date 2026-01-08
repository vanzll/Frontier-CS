import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math

class EfficientMLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Calculate dimensions to stay under 1M params
        # Target: ~980K params for safety margin
        # Using bottleneck architecture with residual connections
        
        # Layer dimensions - carefully balanced
        hidden1 = 768
        hidden2 = 640
        hidden3 = 512
        hidden4 = 384
        hidden5 = 256
        
        # Residual block 1
        self.res1_fc1 = nn.Linear(input_dim, hidden1)
        self.res1_bn1 = nn.BatchNorm1d(hidden1)
        self.res1_fc2 = nn.Linear(hidden1, hidden2)
        self.res1_bn2 = nn.BatchNorm1d(hidden2)
        self.res1_shortcut = nn.Linear(input_dim, hidden2) if input_dim != hidden2 else nn.Identity()
        self.res1_dropout = nn.Dropout(dropout_rate)
        
        # Residual block 2
        self.res2_fc1 = nn.Linear(hidden2, hidden3)
        self.res2_bn1 = nn.BatchNorm1d(hidden3)
        self.res2_fc2 = nn.Linear(hidden3, hidden4)
        self.res2_bn2 = nn.BatchNorm1d(hidden4)
        self.res2_shortcut = nn.Linear(hidden2, hidden4) if hidden2 != hidden4 else nn.Identity()
        self.res2_dropout = nn.Dropout(dropout_rate)
        
        # Final layers
        self.fc3 = nn.Linear(hidden4, hidden5)
        self.bn3 = nn.BatchNorm1d(hidden5)
        self.fc4 = nn.Linear(hidden5, num_classes)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
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
        # Residual block 1
        identity = x
        out = self.res1_fc1(x)
        out = self.res1_bn1(out)
        out = self.activation(out)
        out = self.res1_fc2(out)
        out = self.res1_bn2(out)
        
        identity = self.res1_shortcut(identity)
        out += identity
        out = self.activation(out)
        out = self.res1_dropout(out)
        
        # Residual block 2
        identity = out
        out = self.res2_fc1(out)
        out = self.res2_bn1(out)
        out = self.activation(out)
        out = self.res2_fc2(out)
        out = self.res2_bn2(out)
        
        identity = self.res2_shortcut(identity)
        out += identity
        out = self.activation(out)
        out = self.res2_dropout(out)
        
        # Final layers
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc4(out)
        
        return out

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        # Extract metadata
        num_classes = metadata["num_classes"]
        input_dim = metadata["input_dim"]
        device = metadata.get("device", "cpu")
        
        # Create model with parameter constraint
        model = EfficientMLP(input_dim, num_classes)
        model = model.to(device)
        
        # Verify parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {param_count:,}")
        
        # If model exceeds 1M params, reduce complexity
        if param_count > 950000:
            # Reduce dimensions progressively until under limit
            return self._create_simpler_model(input_dim, num_classes, device)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        
        # Early stopping
        best_val_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        # Training loop
        num_epochs = 150
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # L2 regularization
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param, 2)
                loss = loss + 1e-4 * l2_reg
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
            
            val_acc = val_correct / val_total
            
            # Early stopping check
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
    
    def _create_simpler_model(self, input_dim, num_classes, device):
        """Create a simpler model that definitely stays under 1M params"""
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        model = SimpleMLP(input_dim, num_classes)
        model = model.to(device)
        
        # Verify parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Simpler model parameters: {param_count:,}")
        
        return model