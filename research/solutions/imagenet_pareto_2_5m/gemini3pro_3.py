import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # 1. Configuration & Metadata
        if metadata is None:
            metadata = {}
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # Hyperparameters optimized for 2.5M budget and small dataset
        # H=720 with depth 6 (Stem + 2x2 ResBlock + Head) ~= 2.45M params
        hidden_dim = 720 
        epochs = 70
        lr = 0.001
        
        # 2. Model Definition
        class ResMLPBlock(nn.Module):
            """
            Residual Block containing two Linear layers.
            Structure: x + (Linear -> BN -> ReLU -> Drop -> Linear -> BN -> ReLU -> Drop)
            """
            def __init__(self, dim, dropout=0.2):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
            def forward(self, x):
                return x + self.net(x)

        class ParetoModel(nn.Module):
            def __init__(self, in_features, out_features, hidden_size):
                super().__init__()
                # Stem: Projection to hidden dimension
                self.stem = nn.Sequential(
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Body: Deep Residual MLP
                self.layer1 = ResMLPBlock(hidden_size, dropout=0.25)
                self.layer2 = ResMLPBlock(hidden_size, dropout=0.25)
                
                # Head: Classifier
                self.head = nn.Linear(hidden_size, out_features)

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                return self.head(x)
        
        # 3. Model Initialization
        model = ParetoModel(input_dim, num_classes, hidden_dim).to(device)
        
        # Constraint Enforcement: dynamically reduce size if over budget
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > 2500000:
            # Fallback to a safe width if overhead calculations were off
            hidden_dim = 600
            model = ParetoModel(input_dim, num_classes, hidden_dim).to(device)

        # 4. Training Setup
        # Label Smoothing helps with generalization on small/synthetic data
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # AdamW with weight decay for regularization
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # OneCycleLR for super-convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr, 
            epochs=epochs, 
            steps_per_epoch=len(train_loader)
        )
        
        best_acc = -1.0
        best_weights = copy.deepcopy(model.state_dict())
        
        # 5. Training Loop
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Mixup Augmentation (active for first 85% of training)
                # Helps synthesize new samples from the small pool
                if epoch < int(epochs * 0.85):
                    # Sample lambda from Uniform(0,1) roughly equivalent to Beta(1,1)
                    lam = torch.rand(1).item()
                    # Force major contribution from one sample to maintain label semantics
                    lam = max(lam, 1.0 - lam)
                    
                    index = torch.randperm(inputs.size(0)).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets[index])
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # 6. Validation
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
            
            # Checkpoint
            if val_acc >= best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
        
        # 7. Finalize
        model.load_state_dict(best_weights)
        return model