import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.25):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        residual = x
        
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        
        out = self.ln2(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.drop(out)
        
        return residual + out

class ParetoNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Architecture designed to stay under 2.5M parameters
        # Calculation:
        # Input Block: ~198k
        # 4x ResBlocks (width 512): ~2.11M
        # Output Block: ~67k
        # Total: ~2.37M params
        
        self.hidden_dim = 512
        
        # Input Projection
        self.input_ln = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_act = nn.GELU()
        
        # Deep Residual Backbone
        self.blocks = nn.Sequential(
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim)
        )
        
        # Output Head
        self.final_ln = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
            
        x = self.input_ln(x)
        x = self.input_proj(x)
        x = self.input_act(x)
        
        x = self.blocks(x)
        
        x = self.final_ln(x)
        return self.head(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        
        model = ParetoNet(input_dim, num_classes).to(device)
        
        # Robust training configuration for small datasets
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training parameters
        epochs = 70  # Sufficient for convergence on small data
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_acc = 0.0
        best_model_state = copy.deepcopy(model.state_dict())
        
        # Mixup parameters
        mixup_alpha = 0.4
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply Mixup for regularization
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                else:
                    lam = 1
                
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                target_a, target_b = targets, targets[index]
                
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_acc = correct / total if total > 0 else 0
            
            # Checkpoint best model
            if val_acc >= best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        
        # Load and return best found model
        model.load_state_dict(best_model_state)
        return model