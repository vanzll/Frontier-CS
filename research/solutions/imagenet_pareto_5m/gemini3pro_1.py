import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 5000000)
        device = metadata.get("device", "cpu")
        
        # Define architecture builder
        def build_model(width, depth=6, dropout=0.2):
            layers = []
            # Input block
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
            # Hidden blocks
            for _ in range(depth - 2):
                layers.append(nn.Linear(width, width))
                layers.append(nn.BatchNorm1d(width))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                
            # Output layer
            layers.append(nn.Linear(width, num_classes))
            
            return nn.Sequential(*layers)
            
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        # Optimize width to maximize parameter usage within the limit
        # We fix depth to 6 (Input -> 4x Hidden -> Output) which is effective for this scale
        target_layers = 6 
        min_w, max_w = 100, 2000
        best_width = min_w
        
        # Binary search for maximum width
        while min_w <= max_w:
            mid_w = (min_w + max_w) // 2
            temp_model = build_model(mid_w, depth=target_layers)
            params = count_params(temp_model)
            
            if params <= param_limit:
                best_width = mid_w
                min_w = mid_w + 1
            else:
                max_w = mid_w - 1
        
        # Instantiate final optimized model
        model = build_model(best_width, depth=target_layers).to(device)
        
        # Training Hyperparameters
        # 50 epochs is sufficient for the small dataset size (2048 samples)
        # while fitting within the 1-hour time limit on CPU
        epochs = 60
        learning_rate = 1e-3
        weight_decay = 1e-4
        
        # Optimizer, Loss, and Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Label smoothing improves generalization on synthetic data
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training Loop
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(epochs):
            # Train Phase
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Validation Phase
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
            
            val_acc = correct / total
            
            # Model Checkpointing
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        # Load best weights before returning
        model.load_state_dict(best_model_wts)
        
        return model