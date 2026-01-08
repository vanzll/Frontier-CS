import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class CustomMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 995, dropout_p: float = 0.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

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
                - param_limit: int (2,500,000)
                - baseline_accuracy: float (0.85)
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

        # This hidden dimension is carefully chosen to keep the model's parameter count
        # just under the 2,500,000 limit, maximizing its capacity.
        # Total trainable parameters with this architecture: 2,498,573
        hidden_dim = 995
        dropout_p = 0.4

        model = CustomMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p
        ).to(device)

        # Training hyperparameters
        epochs = 200
        learning_rate = 1e-3
        weight_decay = 1e-4
        
        # Early stopping configuration
        patience = 25
        patience_counter = 0
        best_val_acc = 0.0
        best_model_state = None

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(epochs):
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
            
            current_val_acc = val_correct / val_total
            
            # Check for improvement and update best model state
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break

            scheduler.step()

        # Load the best model weights found during training
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model