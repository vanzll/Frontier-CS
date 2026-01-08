import torch
import torch.nn as nn
import torch.optim as optim
import math

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it, ensuring parameter count stays within 200,000.
        """
        metadata = metadata or {}
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        device = metadata.get("device", "cpu")
        
        # Calculate maximum hidden dimension 'h' to maximize capacity under the limit
        # Architecture: 
        #   Linear(in, h) -> BN(h) -> GELU -> Dropout
        #   Linear(h, h)  -> BN(h) -> GELU -> Dropout
        #   Linear(h, out)
        #
        # Parameter Count Equation:
        #   L1: in*h + h
        #   BN1: 2*h (weight + bias)
        #   L2: h*h + h
        #   BN2: 2*h (weight + bias)
        #   L3: h*out + out
        #
        #   Total = h^2 + (in + out + 6)h + out
        
        b_coeff = input_dim + num_classes + 6
        c_coeff = num_classes - param_limit
        
        # Quadratic formula: (-b + sqrt(b^2 - 4ac)) / 2
        discriminant = b_coeff**2 - 4 * 1 * c_coeff
        h = int((-b_coeff + math.sqrt(discriminant)) / 2)
        
        # Helper to verify exact parameter count
        def count_params(hid):
            return (hid**2) + (input_dim + num_classes + 6)*hid + num_classes
            
        # Ensure strict adherence to limit
        while count_params(h) > param_limit:
            h -= 1
            
        class EfficientNet(nn.Module):
            def __init__(self, in_d, out_d, hid_d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_d, hid_d),
                    nn.BatchNorm1d(hid_d),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hid_d, hid_d),
                    nn.BatchNorm1d(hid_d),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hid_d, out_d)
                )

            def forward(self, x):
                return self.net(x)

        model = EfficientNet(input_dim, num_classes, h).to(device)
        
        # Training Setup
        # AdamW with weight decay helps generalization on small/synthetic data
        # Label smoothing improves performance on classification tasks with noise
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        epochs = 50
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
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
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_acc = correct / total if total > 0 else 0.0
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model