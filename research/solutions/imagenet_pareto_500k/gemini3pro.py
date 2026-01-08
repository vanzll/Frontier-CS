import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ResBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.net(x)

class ImageNetModel(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout_rate):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.body = nn.Sequential(
            ResBlock(h_dim, dropout_rate),
            ResBlock(h_dim, dropout_rate)
        )
        self.tail = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 500000)
        device = metadata.get("device", "cpu")
        
        # Configuration
        # We aim to maximize model capacity within the 500k parameter budget.
        # Based on calculations, a hidden dimension around 386 fits the architecture:
        # Input(384->H) + 2xResBlock(H->H) + Output(H->128)
        hidden_dim = 386
        dropout = 0.25
        
        # Initialize model and ensure it fits in budget
        model = ImageNetModel(input_dim, hidden_dim, num_classes, dropout).to(device)
        
        while True:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if param_count <= param_limit:
                break
            hidden_dim -= 1
            model = ImageNetModel(input_dim, hidden_dim, num_classes, dropout).to(device)
        
        # Optimization Setup
        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training parameters
        epochs = 120
        steps_per_epoch = len(train_loader)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=0.002, 
            epochs=epochs, 
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3
        )
        
        best_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())
        alpha = 0.4 # For Mixup
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Skip batch size 1 to prevent BatchNorm errors
                if inputs.size(0) <= 1:
                    continue
                
                # Mixup Augmentation
                if np.random.random() < 0.5:
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(inputs.size(0)).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    y_a, y_b = targets, targets[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
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
            
            if total > 0:
                acc = correct / total
                if acc >= best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(model.state_dict())
        
        # Return best model
        model.load_state_dict(best_state)
        return model