import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import numpy as np

class ParetoModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mean, std):
        super(ParetoModel, self).__init__()
        # Register normalization statistics as buffers so they are saved with state_dict
        # but do not count as trainable parameters
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        # Architecture: 3-Layer MLP with BatchNorm, SiLU (Swish), and Dropout
        # Designed to maximize capacity within 1M parameter budget
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Apply normalization using registered buffers
        x = (x - self.mean) / (self.std + 1e-6)
        return self.net(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 1000000)
        device = metadata.get("device", "cpu")
        
        # Load all training data into memory for efficiency and Mixup
        train_inputs = []
        train_targets = []
        for inputs, targets in train_loader:
            train_inputs.append(inputs)
            train_targets.append(targets)
        
        X_train = torch.cat(train_inputs).to(device)
        y_train = torch.cat(train_targets).to(device)
        
        # Compute normalization statistics
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0)
        
        # Load validation data
        val_inputs = []
        val_targets = []
        for inputs, targets in val_loader:
            val_inputs.append(inputs)
            val_targets.append(targets)
        X_val = torch.cat(val_inputs).to(device)
        y_val = torch.cat(val_targets).to(device)
        
        # Dynamic Architecture Sizing
        # We solve for Hidden Dimension (H) to maximize parameter usage under the limit.
        # Params = L1(In*H + H) + BN1(2H) + L2(H*H + H) + BN2(2H) + L3(H*Out + Out)
        # Total = H^2 + H(In + Out + 6) + Out
        
        safety_margin = 2000 # Buffer to ensure we never accidentally exceed
        budget = param_limit - safety_margin
        
        a = 1
        b = input_dim + num_classes + 6
        c = num_classes - budget
        
        # Quadratic formula solution for positive H
        delta = b**2 - 4*a*c
        hidden_dim = int((-b + math.sqrt(delta)) / (2*a))
        
        # Initialize model
        model = ParetoModel(input_dim, hidden_dim, num_classes, mean, std).to(device)
        
        # Optimization setup
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        epochs = 80
        batch_size = 64
        num_samples = X_train.shape[0]
        num_batches = math.ceil(num_samples / batch_size)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        best_model_state = copy.deepcopy(model.state_dict())
        
        mixup_alpha = 0.4
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(num_samples, device=device)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]
                
                inputs = X_train[batch_idx]
                targets = y_train[batch_idx]
                
                # Apply Mixup Augmentation
                if mixup_alpha > 0:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                else:
                    lam = 1.0
                
                curr_bs = inputs.size(0)
                mix_idx = torch.randperm(curr_bs, device=device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[mix_idx, :]
                targets_a, targets_b = targets, targets[mix_idx]
                
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
            
            model.train()
            
        # Return best model
        model.load_state_dict(best_model_state)
        model.eval()
        return model