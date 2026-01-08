import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Trains a high-performance, parameter-efficient model.
        
        The strategy is to use a deep Residual MLP (ResMLP) that maximizes the
        parameter budget. This architecture allows for effective training of
        deeper networks, which can capture more complex patterns in the data.

        Key components:
        1.  **Architecture**: A custom ResMLP with a stem, multiple residual blocks,
            and a final head. The dimensions (hidden_dim, num_blocks) are
            calculated to stay just under the 500,000 parameter limit.
        2.  **Regularization**: A combination of techniques is used to prevent
            overfitting on the small training set:
            - BatchNorm: Stabilizes training and acts as a regularizer.
            - Dropout: Prevents co-adaptation of neurons.
            - AdamW Optimizer: Implements decoupled weight decay for better
              generalization than standard L2 regularization in Adam.
            - Label Smoothing: Discourages the model from becoming overconfident.
        3.  **Optimization**:
            - CosineAnnealingLR: A learning rate scheduler that starts with a
              higher learning rate and gradually decreases it, which often leads
              to better convergence and finding broader minima.
        4.  **Best Model Selection**: The model state with the highest validation
            accuracy is saved and returned, ensuring the final model is the one
            that generalized best during training.
        """
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # --- Model Definition ---
        # We define the model classes inside the solve method for encapsulation.
        
        class ResBlock(nn.Module):
            """A residual block for an MLP."""
            def __init__(self, dim, dropout_p):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_p),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(p=dropout_p)

            def forward(self, x):
                residual = self.block(x)
                out = x + residual
                out = self.relu(out)
                out = self.dropout(out)
                return out

        class ResMLP(nn.Module):
            """A Residual MLP model designed to maximize parameter usage."""
            def __init__(self, input_dim, num_classes, hidden_dim, num_blocks, dropout_p):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                
                blocks = [ResBlock(hidden_dim, dropout_p) for _ in range(num_blocks)]
                self.blocks = nn.Sequential(*blocks)
                
                self.head = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.stem(x)
                x = self.blocks(x)
                x = self.head(x)
                return x

        # --- Hyperparameters ---
        # These are tuned to be effective for this specific problem setting and budget.
        # The architecture with H=293 and N=2 yields ~498k parameters.
        HIDDEN_DIM = 293
        NUM_BLOCKS = 2
        DROPOUT_P = 0.3
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 0.015
        EPOCHS = 280
        LABEL_SMOOTHING = 0.1

        # --- Initialization ---
        model = ResMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=HIDDEN_DIM,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P,
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        # --- Training Loop ---
        best_val_acc = 0.0
        best_model_state = None

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

            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
            
            scheduler.step()

        # --- Finalization ---
        # Load the best performing model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model