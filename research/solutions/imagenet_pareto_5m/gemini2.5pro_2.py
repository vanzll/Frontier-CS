import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class ResBlock(nn.Module):
    """
    A residual block for an MLP.
    This block consists of two linear layers with BatchNorm, GELU activation, and Dropout.
    The input is added to the output of the block (skip connection), and an activation
    is applied to the sum.
    """
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.final_activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        out += identity
        out = self.final_activation(out)
        out = self.dropout(out)
        return out

class ParetoOptModel(nn.Module):
    """
    A deep MLP with residual connections, architected to have a parameter count
    just under the 5,000,000 limit. The calculated parameter count is ~4,994,251.
    
    The architecture:
    1. An input layer to project the input dimension to the hidden dimension.
    2. A series of residual blocks to learn complex features.
    3. An output layer to project the hidden dimension to the number of classes.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 869, num_blocks: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
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
                - param_limit: int (5,000,000)
                - baseline_accuracy: float (0.88)
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

        # The model architecture and hyperparameters are tuned to maximize capacity
        # and performance within the given constraints.
        model = ParetoOptModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=869,
            num_blocks=3,
            dropout=0.15
        ).to(device)

        # Training Hyperparameters
        epochs = 450
        max_lr = 8e-4
        weight_decay = 0.05
        
        optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # OneCycleLR is a modern scheduler that often leads to faster convergence and
        # better final performance by cyclically varying the learning rate.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            epochs=epochs, 
            steps_per_epoch=len(train_loader),
            pct_start=0.25,
            div_factor=10,
            final_div_factor=1e4
        )

        best_val_acc = -1.0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation loop to find the best performing model checkpoint
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_acc = correct / total
            
            # Save the state of the model if it has the best validation accuracy so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
        
        # Load the best model state before returning
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model