import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        
        # --- Model Definition ---
        # We define the model classes inside the solve method to keep the solution self-contained.
        
        class ResBlock(nn.Module):
            """A residual block for the MLP."""
            def __init__(self, dim, dropout):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim)
                self.bn1 = nn.BatchNorm1d(dim)
                self.act1 = nn.GELU()
                self.dropout1 = nn.Dropout(dropout)
                
                self.linear2 = nn.Linear(dim, dim)
                self.bn2 = nn.BatchNorm1d(dim)
                
                # Final activation and dropout after adding the residual connection
                self.act2 = nn.GELU()
                self.dropout2 = nn.Dropout(dropout)

            def forward(self, x):
                residual = x
                
                out = self.linear1(x)
                out = self.bn1(out)
                out = self.act1(out)
                out = self.dropout1(out)
                
                out = self.linear2(out)
                out = self.bn2(out)
                
                out += residual
                
                out = self.act2(out)
                out = self.dropout2(out)
                return out

        class DeepMLP(nn.Module):
            """A deep MLP with residual connections."""
            def __init__(self, input_dim, hidden_dim, num_classes, num_blocks, dropout):
                super().__init__()
                
                # Input projection layer
                layers = [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ]
                
                # Stack of residual blocks
                for _ in range(num_blocks):
                    layers.append(ResBlock(hidden_dim, dropout))
                
                self.network = nn.Sequential(*layers)
                
                # Output layer
                self.output_layer = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.network(x)
                x = self.output_layer(x)
                return x
        
        # --- Hyperparameters & Setup ---
        
        device = metadata.get("device", "cpu")
        num_classes = metadata["num_classes"]
        input_dim = metadata["input_dim"]
        
        # Model architecture parameters are chosen to maximize capacity within the 5M limit.
        # This architecture has ~4.96M parameters.
        hidden_dim = 1050
        num_res_blocks = 2
        dropout_rate = 0.2
        
        # Training parameters are selected for robust convergence on a small dataset.
        epochs = 400
        learning_rate = 3e-4
        weight_decay = 0.01
        label_smoothing = 0.1

        # --- Model Initialization ---
        
        model = DeepMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_blocks=num_res_blocks,
            dropout=dropout_rate,
        ).to(device)
        
        # --- Training Setup ---
        
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Cosine annealing is a robust learning rate schedule, ideal for a fixed number of epochs.
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # --- Training Loop ---
        
        model.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Step the scheduler after each epoch
            scheduler.step()
        
        return model