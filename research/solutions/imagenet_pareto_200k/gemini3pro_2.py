import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        Parameter limit: 200,000 (Hard Constraint)
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        device = metadata.get("device", "cpu")
        
        # Calculate max hidden size H for the 2-hidden-layer MLP architecture
        # Architecture: BN(in) -> Linear(in, h) -> BN(h) -> ReLU -> Drop -> Linear(h, h) -> BN(h) -> ReLU -> Drop -> Linear(h, out)
        #
        # Parameter Count Calculation:
        # BN_in params: 2 * input_dim
        # Layer 1 (Linear + Bias): input_dim * h + h
        # BN1 params: 2 * h
        # Layer 2 (Linear + Bias): h * h + h
        # BN2 params: 2 * h
        # Layer 3 (Linear + Bias): h * num_classes + num_classes
        #
        # Total = h^2 + h*(input_dim + num_classes + 6) + (2*input_dim + num_classes)
        # We need Total <= param_limit
        
        a = 1
        b = input_dim + num_classes + 6
        c_const = 2 * input_dim + num_classes
        c = c_const - param_limit
        
        # Solve quadratic equation ax^2 + bx + c = 0 for h
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            h = int((-b + math.sqrt(discriminant)) / (2*a))
        else:
            h = 64 # Fallback safe value
            
        # Verify and adjust to strictly meet constraint
        def count_params(hid):
            return hid**2 + hid*(input_dim + num_classes + 6) + (2*input_dim + num_classes)
            
        while count_params(h) > param_limit:
            h -= 1
            
        # Define Model Architecture
        class OptimizedNet(nn.Module):
            def __init__(self, in_d, out_d, hid_d):
                super().__init__()
                # Normalize input features
                self.norm_in = nn.BatchNorm1d(in_d)
                
                self.features = nn.Sequential(
                    # Layer 1
                    nn.Linear(in_d, hid_d),
                    nn.BatchNorm1d(hid_d),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    
                    # Layer 2
                    nn.Linear(hid_d, hid_d),
                    nn.BatchNorm1d(hid_d),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    
                    # Output Layer
                    nn.Linear(hid_d, out_d)
                )
                
            def forward(self, x):
                x = self.norm_in(x)
                return self.features(x)
        
        # Instantiate model
        model = OptimizedNet(input_dim, num_classes, h).to(device)
        
        # Optimization Setup
        # Using AdamW with weight decay to prevent overfitting on small data
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
        
        # CrossEntropy with Label Smoothing for regularization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Scheduler
        epochs = 60
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Track best model
        best_acc = -1.0
        best_weights = copy.deepcopy(model.state_dict())
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Skip batch size 1 to avoid BN errors
                if inputs.size(0) <= 1:
                    continue
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            acc = correct / total if total > 0 else 0.0
            
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        
        # Restore best weights
        model.load_state_dict(best_weights)
        
        return model