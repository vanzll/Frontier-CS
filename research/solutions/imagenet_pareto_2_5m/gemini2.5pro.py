import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

# Model Definition
# The architecture is a ResNet-style MLP (ResMLP) designed to maximize parameter
# count under the 2.5M limit while providing strong representation capacity.
# We use a hidden dimension of 727 and 2 residual blocks to get close to the limit.

class ResBlock(nn.Module):
    """
    A residual block for an MLP, using a pre-activation-like structure.
    It follows the pattern: (Norm -> Activation -> Linear) x 2 + Skip Connection.
    """
    def __init__(self, dim: int, dropout_p: float):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.act1 = nn.GELU()
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.norm1(x)
        out = self.act1(out)
        out = self.fc1(out)
        
        out = self.norm2(out)
        out = self.act2(out)
        out = self.fc2(out)
        
        out = self.dropout(out)
        
        return identity + out

class ResMLP(nn.Module):
    """
    A ResNet-style MLP for high-performance classification.
    Parameter count for (input_dim=384, hidden_dim=727, num_blocks=2, num_classes=128):
    - Linear layers: 2,489,887
    - BatchNorm layers: 7,270
    - Total: 2,497,157 (safely under 2,500,000)
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, num_blocks: int, dropout_p: float):
        super().__init__()
        # Initial linear layer to project input to the hidden dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # A sequence of residual blocks to learn deep features
        self.blocks = nn.Sequential(
            *[ResBlock(dim=hidden_dim, dropout_p=dropout_p) for _ in range(num_blocks)]
        )
        
        # Final classifier head with pre-activation
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        """
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # --- Model Architecture ---
        # A 2-block ResMLP with hidden_dim=727 gets to ~2.497M parameters.
        model = ResMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=727,
            num_blocks=2,
            dropout_p=0.2
        ).to(device)

        # --- Training Hyperparameters ---
        num_epochs = 350
        learning_rate = 8e-4
        weight_decay = 0.05
        label_smoothing = 0.1
        early_stopping_patience = 40

        # --- Optimizer, Scheduler, and Loss Function ---
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # --- Training Loop with Early Stopping ---
        best_val_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # --- Training Phase ---
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # --- Validation Phase ---
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
            
            # --- Scheduler and Early Stopping ---
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                break
        
        # --- Load Best Model ---
        # Restore the model to the state with the best validation accuracy
        if best_model_state:
            model.load_state_dict(best_model_state)

        return model