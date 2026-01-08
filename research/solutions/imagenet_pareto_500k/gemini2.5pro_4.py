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
                - param_limit: int (500,000)
                - baseline_accuracy: float (0.72)
                - train_samples: int
                - val_samples: int
                - test_samples: int
                - device: str ("cpu")
        
        Returns:
            Trained torch.nn.Module ready for evaluation
        """
        # Set a seed for reproducibility of results
        torch.manual_seed(42)

        # Extract metadata
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # Define the model architecture inside the solve method
        # This architecture is an MLP designed to maximize parameter usage within the 500k limit.
        # Parameter count check:
        # Layer 1 (Linear + BN): (384 * 768 + 768) + (2 * 768) = 296,448
        # Layer 2 (Linear + BN): (768 * 225 + 225) + (2 * 225) = 173,250
        # Layer 3 (Linear): 225 * 128 + 128 = 28,928
        # Total: 296,448 + 173,250 + 28,928 = 498,626 parameters (< 500,000)
        class EfficientMLP(nn.Module):
            def __init__(self, input_dim: int, num_classes: int, dropout_rate: float):
                super().__init__()
                h1 = 768
                h2 = 225
                
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, h1),
                    nn.BatchNorm1d(h1),
                    nn.GELU(),
                    nn.Dropout(p=dropout_rate),

                    nn.Linear(h1, h2),
                    nn.BatchNorm1d(h2),
                    nn.GELU(),
                    nn.Dropout(p=dropout_rate),

                    nn.Linear(h2, num_classes)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layers(x)

        # Hyperparameters chosen for robust training on a small dataset
        EPOCHS = 300
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 0.05
        LABEL_SMOOTHING = 0.1
        DROPOUT_RATE = 0.35

        # Instantiate model and move to the specified device
        model = EfficientMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        
        # Setup loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        # --- Training Loop ---
        # The model is trained for a fixed number of epochs. The validation loader is not used,
        # as a fixed training schedule with a cosine learning rate decay is a reliable strategy.
        for _ in range(EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update the learning rate scheduler at the end of each epoch
            scheduler.step()

        # Set the model to evaluation mode before returning
        model.eval()
        return model