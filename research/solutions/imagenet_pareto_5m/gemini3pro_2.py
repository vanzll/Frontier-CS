import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # Hyperparameters
        HIDDEN_DIM = 1200
        DROPOUT = 0.15
        EPOCHS = 50
        LR_MAX = 2e-3
        WEIGHT_DECAY = 0.01
        
        # Define model architecture using nn.Sequential for standard serialization
        # Architecture: 5-layer MLP with BatchNorm and GELU
        # Layer 1: 384 -> 1200
        # Layer 2: 1200 -> 1200
        # Layer 3: 1200 -> 1200
        # Layer 4: 1200 -> 1200
        # Layer 5: 1200 -> 128
        # Parameter estimation:
        # L1: ~462k
        # L2-L4: 3 * ~1.44M = ~4.32M
        # L5: ~154k
        # Total: ~4.94M (< 5.0M limit)
        
        model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(HIDDEN_DIM, num_classes)
        )
        
        model = model.to(device)
        
        # Verify parameter count constraint
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if param_count > 5000000:
            # Fallback to smaller width if calculation fails (safety net)
            HIDDEN_DIM = 1100
            model = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, num_classes)
            )
            model = model.to(device)
            
        # Optimization setup
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        # Attempt to determine steps per epoch
        try:
            steps_per_epoch = len(train_loader)
        except:
            # Fallback if __len__ not implemented
            steps_per_epoch = 2048 // 32 
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LR_MAX, 
            epochs=EPOCHS, 
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3
        )
        
        # Training loop
        best_acc = 0.0
        best_model_state = copy.deepcopy(model.state_dict())
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            if total > 0:
                val_acc = correct / total
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
        
        # Return best model
        model.load_state_dict(best_model_state)
        return model