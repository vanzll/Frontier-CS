import torch
import torch.nn as nn
import torch.optim as optim

# Helper classes for the model architecture
# A standard residual block for MLPs
class ResidualBlock(nn.Module):
    def __init__(self, features: int, dropout_rate: float):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))

# The main model: a ResNet-style MLP
class MLPResNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_blocks: int, dropout_rate: float):
        super(MLPResNet, self).__init__()
        # Initial projection layer
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        
        # A sequence of residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        
        # Final classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        
        Args:
            train_loader: PyTorch DataLoader with training data
            val_loader: PyTorch DataLoader with validation data
            metadata: Dict with keys:
                - num_classes: int (128)
                - input_dim: int (384)
                - param_limit: int (1,000,000)
                - baseline_accuracy: float (0.8)
                - train_samples: int
                - val_samples: int
                - test_samples: int
                - device: str ("cpu")
        
        Returns:
            Trained torch.nn.Module ready for evaluation
        """
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        
        # --- Model Configuration ---
        # These hyperparameters are carefully tuned to create a model with
        # approximately 986,273 parameters, getting close to the 1M limit.
        hidden_dim = 435 
        num_blocks = 2
        dropout_rate = 0.25

        model = MLPResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate
        ).to(device)
        
        # --- Training Configuration ---
        epochs = 160
        lr = 3e-3 
        weight_decay = 1e-4
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # A learning rate scheduler is crucial for stable training and good convergence.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # --- Training Loop ---
        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()

        # Set the model to evaluation mode before returning
        model.eval()
        return model