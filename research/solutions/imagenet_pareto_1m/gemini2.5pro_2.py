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

        # A 4-layer MLP with BatchNorm and Dropout, designed to maximize parameter usage
        # within the 1,000,000 limit.
        # Parameter count for this architecture is 997,844.
        hidden_dim = 588
        dropout_p = 0.4

        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, num_classes)
        ).to(device)
        
        # Training Hyperparameters
        epochs = 250
        learning_rate = 0.001
        weight_decay = 0.01
        patience = 40
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop with Early Stopping
        best_val_accuracy = 0.0
        best_model_state = None
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validation Step
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
            
            # Early Stopping and Best Model Check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                break
            
            scheduler.step()

        # Load the best model state found during validation
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model