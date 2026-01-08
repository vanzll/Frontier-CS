import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model on synthetic ImageNet-like data within 500k parameter budget.
        """
        if metadata is None:
            metadata = {}
            
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 500000)
        device = metadata.get("device", "cpu")
        
        # Dynamic Architecture Sizing
        # We use a 3-layer MLP (Input -> Hidden -> Hidden -> Output) with Batch Norm and Dropout.
        # Structure:
        # Layer 1: Linear(In, H) + BN(H)
        # Layer 2: Linear(H, H) + BN(H)
        # Layer 3: Linear(H, Out)
        # Total Params = (In*H + H) + 2*H + (H*H + H) + 2*H + (H*Out + Out)
        #              = H^2 + H*(In + Out + 6) + Out
        
        # Solve quadratic equation for H: H^2 + b*H + c <= limit
        limit = param_limit - 500  # Safety buffer
        b_coeff = input_dim + num_classes + 6
        c_coeff = num_classes - limit
        
        delta = b_coeff**2 - 4 * c_coeff
        max_h = (-b_coeff + math.sqrt(delta)) / 2
        hidden_dim = int(max_h)
        
        class AdaptiveMLP(nn.Module):
            def __init__(self, in_d, h_d, out_d, dropout_rate=0.4):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_d, h_d),
                    nn.BatchNorm1d(h_d),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(h_d, h_d),
                    nn.BatchNorm1d(h_d),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(h_d, out_d)
                )
                
                # Robust initialization
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

        # Initialize Model
        model = AdaptiveMLP(input_dim, hidden_dim, num_classes).to(device)
        
        # Training Hyperparameters
        epochs = 100
        lr = 0.002
        weight_decay = 5e-4  # Higher decay for small dataset
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Mixup Params
        use_mixup = True
        alpha = 0.4
        
        best_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                # Apply Mixup for regularization, disable near end of training
                if use_mixup and epoch < epochs - 10:
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(inputs.size(0)).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    y_a, y_b = targets, targets[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
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
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            val_acc = correct / total if total > 0 else 0
            
            if val_acc >= best_acc:
                best_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best performing model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        model.to(device)
        model.eval()
        return model