import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math

class ResidualBlock(nn.Module):
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

class OptimizedModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Architecture tuned for 2.5M parameter budget
        # Calculation for H=600, input=384, output=128:
        # Stem: 384*600 + 600 + 2*600 (BN) = 232,200
        # Block: 2 * (600*600 + 600 + 2*600) = 723,600
        # 3 Blocks: 3 * 723,600 = 2,170,800
        # Head: 600*128 + 128 = 76,928
        # Total: 2,479,928 parameters (Safe under 2,500,000)
        
        self.hidden_dim = 600
        
        self.stem = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.body = nn.Sequential(
            ResidualBlock(self.hidden_dim, dropout=0.2),
            ResidualBlock(self.hidden_dim, dropout=0.2),
            ResidualBlock(self.hidden_dim, dropout=0.2)
        )
        
        self.head = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Metadata extraction
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # Initialize model
        model = OptimizedModel(input_dim, num_classes).to(device)
        
        # Training hyperparameters
        epochs = 70
        learning_rate = 1e-3
        weight_decay = 1e-4
        
        # Optimization setup
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Cosine annealing for smooth convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        # Label smoothing for regularization on synthetic data
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0.0
        best_model_state = copy.deepcopy(model.state_dict())
        
        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Augmentation: Add Gaussian noise to feature vectors
                with torch.no_grad():
                    noise = torch.randn_like(inputs) * 0.02
                    inputs_aug = inputs + noise
                
                optimizer.zero_grad()
                outputs = model(inputs_aug)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # --- Validation Phase ---
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            val_acc = correct / total
            
            # Checkpoint best model
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        
        # Load best weights before returning
        model.load_state_dict(best_model_state)
        return model