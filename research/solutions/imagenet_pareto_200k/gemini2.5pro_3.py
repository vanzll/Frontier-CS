import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# It is generally good practice to define the model outside the Solution class
# to allow for easier testing, inspection, and reuse.

class EfficientMLP(nn.Module):
    """
    An MLP designed to maximize accuracy under a strict parameter budget.
    It uses a deep, narrow architecture with modern components like GELU,
    BatchNorm, and Dropout to improve generalization.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        
        # Architecture is tuned to be just under 200,000 parameters.
        # Calculation:
        # L1: 384*280 + 280 = 107,800
        # BN1: 2*280 = 560
        # L2: 280*220 + 220 = 61,820
        # BN2: 2*220 = 440
        # L3: 220*128 + 128 = 28,288
        # Total = 107800 + 560 + 61820 + 440 + 28288 = 198,908 params
        
        self.layers = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 280),
            nn.BatchNorm1d(280),
            nn.GELU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Linear(280, 220),
            nn.BatchNorm1d(220),
            nn.GELU(),
            nn.Dropout(0.3),

            # Output Layer
            nn.Linear(220, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class Solution:
    """
    Solution class for the ImageNet Pareto Optimization problem.
    """
    
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
        
        # --- Hyperparameters ---
        # These are chosen based on common practices for small datasets and MLPs
        NUM_EPOCHS = 300
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 0.01
        LABEL_SMOOTHING = 0.1

        # --- Setup ---
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # --- Model Initialization ---
        model = EfficientMLP(input_dim=input_dim, num_classes=num_classes)
        model.to(device)

        # --- Optimizer and Loss Function ---
        # AdamW is a robust optimizer with built-in weight decay
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # CosineAnnealingLR helps the model converge to a better minimum
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        
        # CrossEntropyLoss with Label Smoothing for regularization
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        # --- Training Loop ---
        for epoch in range(NUM_EPOCHS):
            model.train()
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Step the scheduler
            scheduler.step()
            
        return model