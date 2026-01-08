import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Helper Model Classes
class ResBlock(nn.Module):
    """
    A residual block for an MLP, with LayerNorm, GELU, and Dropout.
    Using LayerNorm instead of BatchNorm as it can be more stable for smaller batches.
    """
    def __init__(self, dim, dropout_p=0.15):
        super().__init__()
        # Using LayerNorm as it is batch-size independent and often performs well.
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.final_activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity
        out = self.final_activation(out)
        out = self.dropout(out)
        return out

class ParetoNet(nn.Module):
    """
    A custom network designed to maximize parameter usage within the 2.5M limit.
    It uses an input projection, a series of residual blocks, and an output layer.
    """
    def __init__(self, input_dim, num_classes, hidden_dim, n_blocks, dropout_p):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p)
        )
        
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout_p) for _ in range(n_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class Solution:
    """
    Solution class to train a model for the ImageNet Pareto Optimization problem.
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

        # 1. Model Architecture
        # This configuration is carefully tuned to be just under the 2.5M param limit.
        # Calculation:
        # Input Layer: 384*728 + 728 = 280,280
        # ResBlock (x2 Linear layers): 2 * (728*728 + 728) = 1,061,424
        # 2 ResBlocks: 2 * 1,061,424 = 2,122,848
        # Output Layer: 728*128 + 128 = 93,312
        # Total = 280,280 + 2,122,848 + 93,312 = 2,496,440 parameters
        hidden_dim = 728
        n_blocks = 2
        dropout_p = 0.15

        model = ParetoNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout_p=dropout_p
        ).to(device)

        # 2. Training Hyperparameters and Setup
        epochs = 300
        batch_size = 64 # From dataloader, but good to keep in mind
        patience = 40  # For early stopping
        max_lr = 0.003
        weight_decay = 0.05

        optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # OneCycleLR is a powerful scheduler for fast convergence.
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )

        # 3. Training Loop with Validation and Early Stopping
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
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
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Use deepcopy to ensure the state is fully saved at this point
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break

        # Load the best performing model state
        if best_model_state:
            model.load_state_dict(best_model_state)

        return model