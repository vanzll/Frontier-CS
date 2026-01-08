import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.ln1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.act1(out)
        out = self.ln1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.ln2(out)
        out = self.dropout(out)
        return identity + out

class ParetoModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_blocks, dropout_rate=0.1):
        super(ParetoModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout_rate))
        self.blocks = nn.Sequential(*layers)
        
        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.final_act = nn.GELU()
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.final_bn(x)
        x = self.final_act(x)
        x = self.output_head(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        param_limit = metadata["param_limit"]
        device = metadata.get("device", "cpu")
        train_samples = metadata.get("train_samples", 2048)
        
        # Calculate optimal architecture to maximize parameter usage under budget
        # We aim to maximize width and depth within the 5M limit
        best_config = (256, 4) 
        max_params = 0
        safe_limit = param_limit - 5000 # Safety buffer
        
        # Architecture search
        # Range logic: Widths from 384 to 600, Depths from 4 to 20
        for w in range(384, 600, 8): 
            for d in range(4, 25):
                # Parameter Calculation
                # 1. Input projection: (input * w) + w
                p_stem = input_dim * w + w
                
                # 2. Residual Blocks
                # Each block:
                # BN1: 2*w (weight+bias)
                # L1: w*w + w
                # BN2: 2*w
                # L2: w*w + w
                # Total per block = 2*w^2 + 6*w
                p_block = 2 * (w * w) + 6 * w
                p_blocks = d * p_block
                
                # 3. Head
                # BN: 2*w
                # Linear: w*num_classes + num_classes
                p_head = 2 * w + (w * num_classes + num_classes)
                
                total = p_stem + p_blocks + p_head
                
                if total <= safe_limit and total > max_params:
                    max_params = total
                    best_config = (w, d)

        hidden_dim, num_blocks = best_config
        
        # Initialize model
        model = ParetoModel(input_dim, num_classes, hidden_dim, num_blocks, dropout_rate=0.2).to(device)
        
        # Training Hyperparameters
        epochs = 120
        lr = 2e-3
        weight_decay = 0.05  # Strong regularization for small dataset
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Scheduler
        if hasattr(train_loader, "__len__"):
            steps_per_epoch = len(train_loader)
        else:
            steps_per_epoch = max(1, train_samples // 32)
            
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        best_acc = 0.0
        best_state = None
        
        # Mixup parameters
        mixup_alpha = 0.4
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Regularization: Noise Injection (decaying over time)
                if epoch < epochs * 0.8:
                    noise_level = 0.05 * (1 - epoch/epochs)
                    noise = torch.randn_like(inputs) * noise_level
                    inputs = inputs + noise
                
                optimizer.zero_grad()
                
                # Mixup
                if epoch < epochs * 0.9 and mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(inputs.size(0)).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    target_a, target_b = targets, targets[index]
                    
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                else:
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
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            val_acc = correct / total
            
            # Save best model
            if val_acc >= best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Load best model weights
        if best_state is not None:
            model.load_state_dict(best_state)
            
        return model