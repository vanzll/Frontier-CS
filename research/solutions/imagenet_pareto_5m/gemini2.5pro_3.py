import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

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

        # Define model architecture within the solve method to keep it self-contained
        class Block(nn.Module):
            """A residual block for the MLP."""
            def __init__(self, dim, dropout_p=0.1):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.final_act = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.net(x)
                out += identity
                out = self.final_act(out)
                return out

        class ResMLP(nn.Module):
            """A Residual MLP model designed to maximize capacity within parameter limits."""
            def __init__(self, input_dim, hidden_dim, num_classes, num_blocks, dropout_p):
                super().__init__()
                self.input_layer = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                )
                self.blocks = nn.Sequential(
                    *[Block(hidden_dim, dropout_p) for _ in range(num_blocks)]
                )
                self.output_layer = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.input_layer(x)
                x = self.blocks(x)
                x = self.output_layer(x)
                return x

        # --- Hyperparameters and Architecture Configuration ---
        INPUT_DIM = metadata["input_dim"]
        NUM_CLASSES = metadata["num_classes"]
        
        # Architecture carefully tuned to stay just under the 5M parameter limit
        # This configuration yields approx. 4,983,276 parameters.
        HIDDEN_DIM = 868
        NUM_BLOCKS = 3
        
        # Regularization and Training Hyperparameters
        DROPOUT_P = 0.15
        NUM_EPOCHS = 450
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 5e-5
        LABEL_SMOOTHING = 0.1
        
        model_params = {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES,
            "num_blocks": NUM_BLOCKS,
            "dropout_p": DROPOUT_P,
        }
        
        model = ResMLP(**model_params).to(device)
        
        # --- Training Setup ---
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        best_val_acc = 0.0
        best_model_state = None

        # --- Training Loop ---
        for epoch in range(NUM_EPOCHS):
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
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_acc = correct / total if total > 0 else 0
            
            # Save the best model state based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())

            scheduler.step()

        # --- Return Best Model ---
        # Load the state of the best performing model on the validation set
        if best_model_state:
            final_model = ResMLP(**model_params).to(device)
            final_model.load_state_dict(best_model_state)
            return final_model
        else:
            # Fallback to the last model state if no improvement was ever recorded
            return model