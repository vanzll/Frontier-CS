import torch
import torch.nn as nn
import torch.optim as optim

class Solution:
    """
    An expert-level solution for the ImageNet Pareto Optimization problem.

    This solution implements a deep Multi-Layer Perceptron (MLP) with residual connections,
    inspired by ResNet architectures, to maximize accuracy under a strict 5 million
    parameter budget.

    Key components:
    1.  **Architecture (Res-MLP)**:
        - A 'stem' layer projects the input features into a high-dimensional space.
        - Two residual blocks process these features. Each block contains two linear layers,
          batch normalization, ReLU activation, dropout, and a skip connection. This
          structure allows for effective training of deeper networks.
        - A final linear 'head' layer for classification.
        - The hidden dimension is carefully chosen (1054) to bring the total parameter
          count to just under the 5,000,000 limit (~4.998M).

    2.  **Regularization**:
        - **Batch Normalization**: Applied after each linear layer (except the last) to
          stabilize training and improve gradient flow.
        - **Dropout**: Used within the stem and residual blocks (rate=0.4) to combat
          overfitting on the relatively small training dataset.
        - **AdamW Optimizer with Weight Decay**: AdamW provides better weight decay
          regularization than standard Adam, which is crucial for generalization.
        - **Label Smoothing**: The cross-entropy loss uses label smoothing (0.1) to
          prevent the model from becoming overconfident in its predictions.

    3.  **Training Strategy**:
        - **Optimizer**: AdamW with a learning rate of 0.001 and weight decay of 0.01.
        - **Scheduler**: CosineAnnealingLR is used to smoothly decay the learning rate
          over the training epochs, helping the model to settle into a good minimum.
        - **Epochs**: The model is trained for 210 epochs, a number chosen to balance
          convergence with the 1-hour time limit on a CPU-only environment.
    """
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Trains a Res-MLP model and returns it.

        Args:
            train_loader: PyTorch DataLoader with training data.
            val_loader: PyTorch DataLoader with validation data (unused in this strategy).
            metadata: Dictionary with problem-specific information.

        Returns:
            A trained torch.nn.Module ready for evaluation.
        """
        # Get device from metadata, defaulting to "cpu"
        device = torch.device(metadata.get("device", "cpu"))

        # Model and training hyperparameters are fine-tuned for this specific problem
        INPUT_DIM = metadata["input_dim"]
        NUM_CLASSES = metadata["num_classes"]
        HIDDEN_DIM = 1054      # Carefully chosen to be just under 5M params
        DROPOUT_P = 0.4        # High dropout for regularization on small dataset
        NUM_EPOCHS = 210       # Balanced for convergence within 1-hour CPU limit
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0.01
        LABEL_SMOOTHING = 0.1

        # --- Model Definition (nested classes for encapsulation) ---
        class _ResBlock(nn.Module):
            """A residual block for an MLP."""
            def __init__(self, dim, dropout_p):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_p),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                # The skip connection is the core of a residual block
                return self.relu(x + self.layers(x))

        class _Model(nn.Module):
            """The main Res-MLP model."""
            def __init__(self, input_dim, num_classes, hidden_dim, dropout_p):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_p)
                )
                self.blocks = nn.Sequential(
                    _ResBlock(hidden_dim, dropout_p),
                    _ResBlock(hidden_dim, dropout_p)
                )
                self.head = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.stem(x)
                x = self.blocks(x)
                x = self.head(x)
                return x

        # --- Training Setup ---
        model = _Model(INPUT_DIM, NUM_CLASSES, HIDDEN_DIM, DROPOUT_P).to(device)
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        # --- Training Loop ---
        model.train()
        for _ in range(NUM_EPOCHS):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Step the scheduler after each epoch
            scheduler.step()

        # Set model to evaluation mode before returning
        model.eval()
        return model