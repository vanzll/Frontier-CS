import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # This MLP architecture has ~198,272 parameters, which is close to the
        # 200,000 limit, maximizing model capacity. It includes BatchNorm for
        # stable training, GELU for effective non-linearity, and Dropout for
        # regularization to combat overfitting on the small dataset.
        self.layers = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),

            # Block 2
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),

            # Output Layer
            nn.Linear(256, num_classes)
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
        device = metadata.get("device", "cpu")
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        model = Net(input_dim=input_dim, num_classes=num_classes).to(device)

        # Hyperparameters selected from modern training recipes for robustness
        EPOCHS = 200
        LEARNING_RATE = 3e-4
        WEIGHT_DECAY = 0.05
        LABEL_SMOOTHING = 0.1

        # AdamW optimizer is chosen for its effectiveness and good generalization
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # CrossEntropyLoss with label smoothing helps prevent over-confident predictions
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # A Cosine Annealing scheduler smoothly decays the learning rate, which
        # often leads to better final model performance.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=1e-6
        )

        # The model is trained for a fixed number of epochs. This is a robust
        # strategy when paired with a cosine scheduler.
        for _ in range(EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()

        # Set the model to evaluation mode before returning. This is critical for
        # layers like BatchNorm and Dropout to function correctly during inference.
        model.eval()
        
        return model