import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.act(out)
        return out

class ResNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # Architecture configuration derived to stay under 500k parameters
        # Calculation for H=244, N=3:
        # Input: 384*244 + 244 = 93,940
        # Block: 2*(244^2 + 244) + 4*244 = 120,536
        # Output: 244*128 + 128 = 31,360
        # Total: 93,940 + 3*120,536 + 31,360 = 486,908 < 500,000
        HIDDEN_DIM = 244
        NUM_BLOCKS = 3
        DROPOUT = 0.25
        
        model = ResNetMLP(input_dim, HIDDEN_DIM, num_classes, NUM_BLOCKS, DROPOUT)
        model = model.to(device)
        
        # Training hyperparameters
        EPOCHS = 60
        LR = 0.003
        WEIGHT_DECAY = 1e-2
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Scheduler
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LR, 
            epochs=EPOCHS, 
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Mixup Data Augmentation
                if np.random.random() < 0.5:
                    alpha = 0.4
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
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            val_acc = correct / total
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        # Load best weights
        model.load_state_dict(best_model_wts)
        return model