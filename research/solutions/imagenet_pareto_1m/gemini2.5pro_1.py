import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# Helper classes for the model architecture, defined at the module level for clarity.

class ResidualBlock(nn.Module):
    """
    A residual block for an MLP, designed to improve gradient flow in deep networks.
    It consists of two linear layers with BatchNorm, ReLU activation, and Dropout.
    """
    def __init__(self, dim: int, dropout_p: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with a residual connection.
        """
        residual = self.block(x)
        return self.relu(x + residual)

class ParameterizedModel(nn.Module):
    """
    A deep MLP featuring residual blocks. The architecture is carefully tuned 
    to maximize parameter count while staying under the 1,000,000 parameter limit.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, num_blocks: int, dropout_p: float):
        super().__init__()
        # Initial projection layer
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        
        # A sequence of residual blocks
        blocks = [ResidualBlock(hidden_dim, dropout_p) for _ in range(num_blocks)]
        self.residual_blocks = nn.Sequential(*blocks)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.residual_blocks(x)
        x = self.classifier(x)
        return x

class Solution:
    """
    Solution for the ImageNet Pareto Optimization problem.
    """
    def solve(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, metadata: dict = None) -> torch.nn.Module:
        """
        Trains a neural network model to maximize accuracy on a synthetic dataset,
        adhering to a strict parameter budget.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            metadata: A dictionary containing problem-specific parameters like
                      input dimensions, number of classes, and device.

        Returns:
            A trained torch.nn.Module, ready for evaluation.
        """
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # --- Hyperparameters ---
        # Architecture tuned to be just under 1,000,000 parameters (approx. 999,153)
        HIDDEN_DIM = 438
        NUM_BLOCKS = 2
        DROPOUT_P = 0.4
        
        # Training parameters optimized for this task
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 1e-4
        NUM_EPOCHS = 500  # A generous number of epochs
        PATIENCE = 50     # Early stopping patience

        # --- Model Initialization ---
        model = ParameterizedModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P
        ).to(device)

        # --- Optimizer, Loss Function, and Learning Rate Scheduler ---
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # --- Training Loop with Early Stopping ---
        best_val_accuracy = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        for _ in range(NUM_EPOCHS):
            # Training phase
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_accuracy = val_correct / val_total
            
            # Update learning rate
            scheduler.step()

            # Early stopping logic: save the best model and stop if no improvement
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Use deepcopy to ensure the state is fully saved at this point
                best_model_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= PATIENCE:
                # Stop training if validation accuracy has not improved for 'PATIENCE' epochs
                break
        
        # Load the best performing model state before returning
        if best_model_state:
            model.load_state_dict(best_model_state)

        return model