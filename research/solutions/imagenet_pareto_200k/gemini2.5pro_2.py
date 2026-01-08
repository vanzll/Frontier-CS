import torch
import torch.nn as nn
import torch.optim as optim
import copy

def _init_weights(m: nn.Module):
    """
    Initializes weights of Linear layers using Kaiming normal initialization
    for ReLU activations.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class _ResidualBlock(nn.Module):
    """
    A residual block with a bottleneck design, BatchNorm, and Dropout.
    This helps in training deeper networks and provides regularization.
    """
    def __init__(self, in_features: int, hidden_features: int, dropout_p: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class _ResMLP(nn.Module):
    """
    A deep Multi-Layer Perceptron with residual connections, designed to
    maximize accuracy within a strict parameter budget.
    Architecture:
    - Initial projection layer with BatchNorm, ReLU, and Dropout.
    - A series of residual blocks.
    - A final linear classifier.
    The dimensions are chosen to be close to the 200,000 parameter limit.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 210, 
                 res_hidden_dim: int = 105, num_blocks: int = 2, dropout_p: float = 0.25):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        
        self.res_blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, res_hidden_dim, dropout_p) for _ in range(num_blocks)]
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.res_blocks(x)
        x = self.classifier(x)
        return x

class Solution:
    """
    Solution class to train a model for the ImageNet Pareto Optimization problem.
    """
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Trains a custom ResMLP model and returns it.
        
        The strategy involves:
        1. A custom ResMLP architecture carefully designed to maximize capacity
           under the 200,000 parameter limit.
        2. Strong regularization techniques (BatchNorm, Dropout, AdamW weight decay,
           label smoothing) to prevent overfitting on the small dataset.
        3. A modern training recipe including the AdamW optimizer and a
           CosineAnnealingLR scheduler to achieve better convergence.
        4. Early stopping based on validation accuracy to return the best-performing
           model and avoid unnecessary training time.
        
        Args:
            train_loader: PyTorch DataLoader with training data.
            val_loader: PyTorch DataLoader with validation data.
            metadata: Dictionary with problem-specific information.
        
        Returns:
            A trained torch.nn.Module ready for evaluation.
        """
        device = torch.device(metadata.get("device", "cpu"))
        num_classes = metadata["num_classes"]
        input_dim = metadata["input_dim"]

        # Model instantiation with architecture optimized for the 200K parameter budget.
        # Parameter count for this config is ~198,368.
        model = _ResMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=210,
            res_hidden_dim=105,
            num_blocks=2,
            dropout_p=0.25
        )
        model.apply(_init_weights)
        model.to(device)

        # Training hyperparameters tuned for this problem
        NUM_EPOCHS = 350
        LEARNING_RATE = 1.2e-3
        WEIGHT_DECAY = 5e-5
        LABEL_SMOOTHING = 0.1
        PATIENCE = 50

        # Setup optimizer, scheduler, and loss function
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

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

            # Update learning rate
            scheduler.step()

            # Check for improvement and apply early stopping
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                # Use deepcopy to save the state of the best model
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break
        
        # Load the best performing model state for the final return
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model