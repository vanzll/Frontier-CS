import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Helper classes for the model architecture
class BottleneckResBlock(nn.Module):
    """A residual block with a bottleneck design."""
    def __init__(self, dim, bottleneck_dim, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(bottleneck_dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.final_gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        x = self.final_gelu(x)
        x = self.dropout(x)
        return x

class ResMLP(nn.Module):
    """A Residual Multi-Layer Perceptron optimized for parameter efficiency."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks, dropout_p):
        super().__init__()
        bottleneck_dim = hidden_dim // 4
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.Sequential(
            *[BottleneckResBlock(hidden_dim, bottleneck_dim, dropout_p) for _ in range(num_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.blocks(x)
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
                - param_limit: int (1,000,000)
                - baseline_accuracy: float (0.8)
                - train_samples: int
                - val_samples: int
                - test_samples: int
                - device: str ("cpu")
        
        Returns:
            Trained torch.nn.Module ready for evaluation
        """
        
        # Extract metadata
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        device = torch.device(metadata["device"])

        # --- Hyperparameters ---
        # Architecture
        HIDDEN_DIM = 656
        NUM_BLOCKS = 3
        DROPOUT_P = 0.2
        
        # Training
        EPOCHS = 400
        BATCH_SIZE = 128 # This is determined by the loader, but good to keep in mind
        LEARNING_RATE = 2e-3
        WEIGHT_DECAY = 0.05
        LABEL_SMOOTHING = 0.1
        
        # Scheduler
        WARMUP_EPOCHS = 20
        
        # Early Stopping
        EARLY_STOPPING_PATIENCE = 50

        # --- Model Initialization ---
        model = ResMLP(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_classes=num_classes,
            num_blocks=NUM_BLOCKS,
            dropout_p=DROPOUT_P
        ).to(device)
        
        # --- Optimizer, Scheduler, and Loss Function ---
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        main_epochs = EPOCHS - WARMUP_EPOCHS
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        # --- Training Loop ---
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
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
            current_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    current_val_loss += loss.item()
            
            current_val_loss /= len(val_loader)

            # Early stopping and model checkpointing
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    break
            
            scheduler.step()

        # Load the best model state found during training
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model