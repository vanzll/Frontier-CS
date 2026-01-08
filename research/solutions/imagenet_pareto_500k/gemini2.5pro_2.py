import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

class ResBlock(nn.Module):
    """A residual block for an MLP."""
    def __init__(self, dim, dropout_p=0.4):
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

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out

class CustomNet(nn.Module):
    """A ResNet-style MLP designed to maximize accuracy under the parameter budget."""
    def __init__(self, input_dim, num_classes, hidden_dim=293, dropout_p=0.4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = ResBlock(hidden_dim, dropout_p)
        self.res_block2 = ResBlock(hidden_dim, dropout_p)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.head(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata["device"]
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]

        # Hyperparameters chosen for strong performance and regularization on a small dataset
        HIDDEN_DIM = 293       # Carefully chosen to stay under 500k params
        DROPOUT_P = 0.4        # Strong dropout to prevent overfitting
        EPOCHS = 800           # Ample epochs for convergence
        LEARNING_RATE = 2e-3   # Higher learning rate is effective with cosine annealing
        WEIGHT_DECAY = 5e-2    # AdamW's weight decay acts as a powerful regularizer
        LABEL_SMOOTHING = 0.1  # Regularizes the output distribution
        BATCH_SIZE = 64        # Standard batch size

        model = CustomNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=HIDDEN_DIM,
            dropout_p=DROPOUT_P
        ).to(device)

        # Use torch.compile for potential speed-up in PyTorch 2.x
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception:
                # Proceed without compilation if it fails
                pass

        # Combine training and validation sets to train on all available labeled data
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        combined_dataset = ConcatDataset([train_dataset, val_dataset])
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0 # Safest setting for unknown environments
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        model.train()
        for _ in range(EPOCHS):
            for inputs, targets in combined_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        model.eval()
        return model