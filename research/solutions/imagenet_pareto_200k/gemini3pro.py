import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model on synthetic ImageNet-like data within 200k parameters.
        """
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # Architecture Design for ~199k parameters
        # Constraints: < 200,000 parameters
        # Configuration: 384 -> 280 -> 220 -> 128
        #
        # Parameter Calculation:
        # Layer 1 (384 -> 280):
        #   Weights: 384 * 280 = 107,520
        #   Bias: 280
        #   BatchNorm: 280 * 2 = 560
        #   Subtotal: 108,360
        #
        # Layer 2 (280 -> 220):
        #   Weights: 280 * 220 = 61,600
        #   Bias: 220
        #   BatchNorm: 220 * 2 = 440
        #   Subtotal: 62,260
        #
        # Layer 3 (220 -> 128):
        #   Weights: 220 * 128 = 28,160
        #   Bias: 128
        #   Subtotal: 28,288
        #
        # Total: 108,360 + 62,260 + 28,288 = 198,908 parameters (Safe margin)
        
        h1 = 280
        h2 = 220
        
        class ParetoNet(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h1),
                    nn.BatchNorm1d(h1),
                    nn.SiLU(),  # Swish activation often better than ReLU for deep MLPs
                    nn.Dropout(0.3),
                    nn.Linear(h1, h2),
                    nn.BatchNorm1d(h2),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(h2, out_dim)
                )

            def forward(self, x):
                return self.net(x)

        model = ParetoNet(input_dim, num_classes).to(device)
        
        # Safety check for parameter limit
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > 200000:
            # Fallback to smaller architecture if calculation fails
            h1, h2 = 256, 192
            model = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.BatchNorm1d(h1),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(h2, num_classes)
            ).to(device)

        # Training hyperparameters
        epochs = 150
        learning_rate = 0.001
        weight_decay = 0.02  # Stronger regularization for small dataset
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        # Label smoothing helps generalization on small datasets
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = -1.0
        best_model_state = copy.deepcopy(model.state_dict())
        
        # Mixup parameters
        use_mixup = True
        mixup_alpha = 0.4
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if use_mixup:
                    # Mixup augmentation
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(inputs.size(0)).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    target_a, target_b = targets, targets[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Evaluate on validation set
            # Check every 2 epochs initially, every epoch later
            if epoch % 2 == 0 or epoch > epochs - 30:
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
                
                val_acc = correct / total if total > 0 else 0.0
                
                # Save best model
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
        
        # Return best model found
        model.load_state_dict(best_model_state)
        return model