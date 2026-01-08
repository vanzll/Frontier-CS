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

        HIDDEN_DIM = 1343
        DROPOUT_RATE = 0.4
        EPOCHS = 250
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 1e-4
        PATIENCE = 25

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, HIDDEN_DIM),
                    nn.BatchNorm1d(HIDDEN_DIM),
                    nn.ReLU(inplace=True),
                    nn.Dropout(DROPOUT_RATE),
                    
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                    nn.BatchNorm1d(HIDDEN_DIM),
                    nn.ReLU(inplace=True),
                    nn.Dropout(DROPOUT_RATE),
                    
                    nn.Linear(HIDDEN_DIM, num_classes)
                )

            def forward(self, x):
                return self.layers(x)
        
        model = Net().to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_val_accuracy = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

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
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            
            scheduler.step()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model