import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class _MLP(nn.Module):
    """
    A custom MLP architecture with BatchNorm and Dropout, designed to maximize
    capacity under the parameter constraint.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, n_hidden_layers=4, dropout_p=0.4):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        ]
        for _ in range(n_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p)
            ])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        
        Args:
            train_loader: PyTorch DataLoader with training data
            val_loader: PyTorch DataLoader with validation data
            metadata: Dict with keys including num_classes, input_dim, device
        
        Returns:
            Trained torch.nn.Module ready for evaluation
        """
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        
        # --- Model Configuration ---
        # A 5-layer MLP (4 hidden) with 829 hidden units.
        # This architecture is designed to be just under the 2.5M parameter limit,
        # providing high capacity for the task.
        # Parameter count: ~2,495,475
        hidden_dim = 829
        n_hidden_layers = 4
        dropout_p = 0.4
        
        model = _MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            n_hidden_layers=n_hidden_layers,
            dropout_p=dropout_p
        ).to(device)

        # --- Training Hyperparameters ---
        # These values are chosen for robust training on a small dataset where
        # overfitting is a primary concern.
        EPOCHS = 350
        PATIENCE = 40
        MAX_LR = 1.2e-3
        WEIGHT_DECAY = 1.5e-2
        LABEL_SMOOTHING = 0.1

        # --- Optimizer, Scheduler, and Loss Function ---
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=MAX_LR,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
        )
        
        # --- Training and Validation Loop with Early Stopping ---
        best_val_acc = -1.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            # --- Training Phase ---
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # --- Validation Phase ---
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
            
            # --- Early Stopping Logic ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                break
        
        # --- Load Best Model and Return ---
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model