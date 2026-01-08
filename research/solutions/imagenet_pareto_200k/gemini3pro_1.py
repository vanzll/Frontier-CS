import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it, maximizing accuracy within parameter budget.
        """
        # 1. Parse Metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 200000)
        device = metadata.get("device", "cpu")

        # 2. Dynamic Architecture Design
        # Design a 3-layer MLP: Input -> H -> H -> Output
        # Structure:
        #   L1: Linear(In, H) + BN(H) + Activation
        #   L2: Linear(H, H) + BN(H) + Activation
        #   L3: Linear(H, Out)
        #
        # Parameter Count Formula:
        #   L1: In*H + H (bias) + 2*H (BN params) = H*(In + 3)
        #   L2: H*H + H (bias) + 2*H (BN params) = H*(H + 3)
        #   L3: H*Out + Out (bias)
        #   Total = H^2 + H*(In + Out + 6) + Out
        
        # Solving for H: H^2 + bH + c <= limit
        # Reserve slight buffer (e.g., 500 params) for safety
        limit = param_limit - 500
        a = 1
        b_coef = input_dim + num_classes + 6
        c_coef = num_classes - limit
        
        # Quadratic formula: (-b + sqrt(b^2 - 4ac)) / 2a
        discriminant = b_coef**2 - 4 * a * c_coef
        h_float = (-b_coef + math.sqrt(discriminant)) / 2
        hidden_dim = int(h_float)
        
        # Ensure hidden_dim is even (optional optimization for memory alignment)
        if hidden_dim % 2 != 0:
            hidden_dim -= 1

        # 3. Model Definition
        class AdaptiveNetwork(nn.Module):
            def __init__(self, in_d, h_d, out_d):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(in_d, h_d),
                    nn.BatchNorm1d(h_d),
                    nn.SiLU(), # SiLU (Swish) often outperforms ReLU
                    nn.Dropout(0.25),
                    
                    nn.Linear(h_d, h_d),
                    nn.BatchNorm1d(h_d),
                    nn.SiLU(),
                    nn.Dropout(0.25),
                    
                    nn.Linear(h_d, out_d)
                )

            def forward(self, x):
                return self.features(x)

        model = AdaptiveNetwork(input_dim, hidden_dim, num_classes).to(device)

        # 4. Verify Constraints
        current_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if current_params > param_limit:
            # Fallback: reduce hidden dimension if calculation was slightly off
            hidden_dim -= 10
            model = AdaptiveNetwork(input_dim, hidden_dim, num_classes).to(device)

        # 5. Training Setup
        # Use Label Smoothing to prevent overfitting on synthetic/noisy data
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Training config
        epochs = 75
        try:
            steps_per_epoch = len(train_loader)
        except:
            # Fallback estimate if len not available
            steps_per_epoch = 2048 // 32
            
        # OneCycleLR for super-convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.004, 
            epochs=epochs, 
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3
        )

        # 6. Training Loop with Mixup
        best_acc = 0.0
        best_state = None
        
        # Mixup alpha
        alpha = 0.4

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Mixup implementation
                if alpha > 0:
                    lam = np.random.beta(alpha, alpha)
                else:
                    lam = 1
                    
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                targets_a, targets_b = targets, targets[index]
                
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
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
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            acc = correct / total
            
            # Save best model
            if acc >= best_acc:
                best_acc = acc
                # Save to CPU to avoid memory issues (though unlikely on this scale)
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # 7. Finalize
        if best_state is not None:
            model.load_state_dict(best_state)
        
        model.to(device)
        return model