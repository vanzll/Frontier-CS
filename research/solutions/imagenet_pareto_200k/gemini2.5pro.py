import torch
import torch.nn as nn
import torch.optim as optim
import copy

class CustomNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Architecture designed to maximize parameter usage within the 200k limit.
        # Total parameters: ~199,220
        # Layers: 384 -> 256 -> 192 -> 156 -> 128
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(192, 156),
            nn.BatchNorm1d(156),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(156, num_classes)
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
                - param_limit: int (200,000)
                - baseline_accuracy: float (0.65)
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
        
        model = CustomNet(input_dim, num_classes).to(device)

        # Hyperparameters
        NUM_EPOCHS = 350
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 1e-4

        # Optimizer and Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_val_accuracy = -1.0
        best_model_state = None

        # Training Loop
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
            
            current_val_accuracy = correct / total
            
            # Save the best model based on validation accuracy
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_model_state = copy.deepcopy(model.state_dict())
        
        # Load the best performing model state
        if best_model_state:
            model.load_state_dict(best_model_state)

        return model