import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

class CustomMLP(nn.Module):
    """
    A custom MLP architecture designed to maximize parameter usage within the 500k limit.
    It uses modern components like GELU activation, BatchNorm, and Dropout for regularization.
    
    The architecture is:
    Linear -> BatchNorm -> GELU -> Dropout ->
    Linear -> BatchNorm -> GELU -> Dropout ->
    Linear
    
    With hidden_dim=493, the total parameters are ~498,551.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Solution:
    def solve(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, 
              metadata: dict = None) -> torch.nn.Module:
        """
        Trains a neural network model to maximize accuracy on a synthetic dataset
        while adhering to a strict parameter limit.
        
        The strategy involves:
        1. A custom MLP architecture that uses most of the 500k parameter budget.
        2. Advanced training techniques:
           - AdamW optimizer for better weight decay.
           - OneCycleLR scheduler for fast convergence.
           - Mixup regularization to improve generalization on the small dataset.
           - Label smoothing in the loss function.
        3. Tracking the best model based on validation accuracy over a sufficient number of epochs.
        """
        # --- 1. Setup and Hyperparameters ---
        device = metadata.get('device', 'cpu')
        input_dim = metadata['input_dim']
        num_classes = metadata['num_classes']

        # Model hyperparameters chosen to be close to the 500k parameter limit.
        hidden_dim = 493
        dropout_rate = 0.3

        # Training hyperparameters tuned for this specific task
        epochs = 400
        max_lr = 1.2e-2
        weight_decay = 1.5e-2
        label_smoothing = 0.1
        mixup_alpha = 1.0

        # --- 2. Model, Optimizer, Loss, Scheduler ---
        model = CustomMLP(input_dim, hidden_dim, num_classes, dropout_rate).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # OneCycleLR is highly effective for fast training and finding good solutions.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # --- 3. Training Loop with Validation ---
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # --- Training Phase ---
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply Mixup regularization
                lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
                rand_index = torch.randperm(inputs.size(0)).to(device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
                target_a, target_b = targets, targets[rand_index]
                
                outputs = model(mixed_inputs)
                
                # Loss calculation with mixed targets
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)

                optimizer.zero_grad()
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
            
            # Save the model state if it has the best validation accuracy so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

        # --- 4. Finalize and Return ---
        # Load the best performing model state before returning
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model