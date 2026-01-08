import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ResNetMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_blocks):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it within the 5M parameter budget.
        """
        # Extract metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 5000000)
        device = metadata.get("device", "cpu")
        
        # Configuration for architecture to fit under 5M params
        # Initial calculation:
        # Params ~= Input + K * Block + Output
        # Block(750) ~= 1.13M. 4 Blocks ~= 4.5M. + Overhead ~= 4.9M
        hidden_dim = 750
        num_blocks = 4
        
        # Dynamic adjustment to guarantee constraint satisfaction
        while True:
            model = ResNetMLP(input_dim, num_classes, hidden_dim, num_blocks)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if total_params <= param_limit:
                break
            hidden_dim -= 4
        
        model = model.to(device)
        
        # Training Hyperparameters
        # High regularization (decay, dropout, mixup) because dataset is small (2048) relative to model size
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        epochs = 55
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0.0
        best_model_state = None
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Mixup Augmentation
                alpha = 0.4
                lam = np.random.beta(alpha, alpha)
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
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
            
            val_acc = correct / total
            
            # Save best model
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                # Store state dict on CPU to avoid reference issues
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Load best weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model.to(device)