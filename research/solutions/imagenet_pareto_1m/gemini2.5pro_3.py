import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Model Definition: Defined at the top level for clarity.

class ResidualBlock(nn.Module):
    """A residual block for an MLP."""
    def __init__(self, size: int, dropout_p: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.gelu(out)
        return out

class ResMLP(nn.Module):
    """A ResNet-style MLP optimized for this task."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, 
                 num_blocks: int, dropout_p: float):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        blocks = [ResidualBlock(hidden_dim, dropout_p) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
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

        # --- Hyperparameters ---
        # Architecture is designed to be just under the 1M parameter limit.
        HIDDEN_DIM = 438
        NUM_BLOCKS = 2
        DROPOUT_P = 0.25
        
        # Training parameters are tuned for stable and fast convergence.
        EPOCHS = 300
        MAX_LR = 2e-3
        WEIGHT_DECAY = 1e-2
        PATIENCE = 40

        # --- Model Initialization ---
        model = ResMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P
        ).to(device)

        # --- Training Setup ---
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        
        total_steps = EPOCHS * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=total_steps)

        # --- Training Loop with Early Stopping ---
        best_val_accuracy = -1.0
        best_model_state = None
        epochs_no_improve = 0
        
        for _ in range(EPOCHS):
            # Training phase
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation phase
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
            
            val_accuracy = correct / total
            
            # Save the best model and check for early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    break
        
        # Load the best performing model state
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Return the final model in evaluation mode and on the CPU
        return model.to(torch.device("cpu")).eval()