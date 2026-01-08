import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    """A simple residual block for an MLP."""
    def __init__(self, hidden_dim: int, dropout_p: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class Net(nn.Module):
    """
    The main network architecture. It consists of an input projection,
    a residual block, and an output layer.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, dropout_p: float):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res_block = ResidualBlock(hidden_dim, dropout_p)
        self.post_res_activation = nn.GELU()
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_block(x)
        x = self.post_res_activation(x)
        x = self.output_layer(x)
        return x

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
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # Hyperparameters chosen to maximize capacity and regularize effectively
        hidden_dim = 385    # Calculated to bring param count near 500k (~497,163)
        dropout_p = 0.3
        epochs = 450
        learning_rate = 1.3e-3
        weight_decay = 8e-5
        label_smoothing = 0.1
        noise_level = 0.01

        model = Net(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                if noise_level > 0:
                    inputs = inputs + torch.randn_like(inputs) * noise_level

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()

        model.eval()
        return model