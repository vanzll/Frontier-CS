import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math

class ResBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.gelu(out)
        out = self.dropout(out)
        return out

class Pareto5MModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, n_blocks, dropout_rate):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        device = metadata.get("device", "cpu")
        num_classes = metadata["num_classes"]
        input_dim = metadata["input_dim"]

        # --- Hyperparameters ---
        HIDDEN_DIM = 1050
        N_BLOCKS = 2
        DROPOUT_RATE = 0.25

        EPOCHS = 250
        WARMUP_EPOCHS = 15
        BASE_LR = 0.0012
        WEIGHT_DECAY = 0.01
        LABEL_SMOOTHING = 0.1

        # --- Model Initialization ---
        model = Pareto5MModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=HIDDEN_DIM,
            n_blocks=N_BLOCKS,
            dropout_rate=DROPOUT_RATE
        )
        model.to(device)

        # --- Optimizer, Loss, and Scheduler ---
        optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        best_val_acc = -1.0
        best_model_state = None

        # --- Training Loop ---
        for epoch in range(EPOCHS):
            # Learning rate scheduling (warmup + cosine decay)
            if epoch < WARMUP_EPOCHS:
                lr = BASE_LR * (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                lr = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # --- Training Phase ---
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model