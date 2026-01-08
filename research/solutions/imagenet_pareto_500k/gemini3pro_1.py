import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class ParetoNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Parameter Budget Analysis:
        # Limit: 500,000
        #
        # Input LayerNorm: 384 * 2 = 768
        #
        # Residual Block 1:
        #   Linear: 384 * 384 + 384 = 147,840
        #   LayerNorm: 384 * 2 = 768
        #   Total Block: 148,608
        #
        # Residual Block 2: 148,608
        # Residual Block 3: 148,608
        #
        # Head:
        #   Linear: 384 * 128 + 128 = 49,280
        #
        # Total Model:
        # 768 + (148,608 * 3) + 49,280 = 495,872
        # Margin: 4,128 (Safe)
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.blocks = nn.Sequential(
            ResidualBlock(input_dim),
            ResidualBlock(input_dim),
            ResidualBlock(input_dim)
        )
        
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.blocks(x)
        return self.head(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model to maximize accuracy within 500k parameters.
        """
        # Extract metadata
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        device = metadata.get("device", "cpu")
        
        # Initialize model
        model = ParetoNet(input_dim, num_classes).to(device)
        
        # Training Configuration
        # Uses cosine annealing and AdamW for efficient convergence on CPU
        epochs = 70
        lr = 1e-3
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Best model tracking
        best_val_acc = 0.0
        best_weights = model.state_dict()
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Input Noise Injection
                # Adds Gaussian noise to features during early training
                # Critical for regularization given small dataset (2048 samples) vs params (500k)
                if epoch < 50:
                    noise_scale = 0.03 * (1 - epoch / 50)
                    noise = torch.randn_like(inputs) * noise_scale
                    inputs = inputs + noise
                
                optimizer.zero_grad()
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
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_weights = model.state_dict()
        
        # Restore best weights before returning
        model.load_state_dict(best_weights)
        return model