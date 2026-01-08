import torch
import torch.nn as nn
import torch.optim as optim

class Solution:
    """
    A solution to the ImageNet Pareto Optimization problem (200K Parameter Variant).
    This implementation uses a small ResNet-style MLP (ResMLP) designed to maximize
    accuracy while staying under the 200,000 parameter limit.
    
    The architecture consists of:
    1. An input projection layer.
    2. A residual block containing two linear layers with GELU activation,
       BatchNorm, and Dropout.
    3. An output projection layer to the number of classes.

    The model is trained using a modern training recipe:
    - Optimizer: AdamW for better weight decay handling.
    - Scheduler: CosineAnnealingLR to smoothly adjust the learning rate.
    - Loss: CrossEntropyLoss with label smoothing to prevent overconfidence.
    - Regularization: A combination of weight decay, dropout, and batch normalization
      is used to combat overfitting on the small dataset.
    """
    
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        
        # --- Metadata and Device Setup ---
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        
        # For reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # --- Model Definition ---
        
        class ResBlock(nn.Module):
            """
            A residual block for an MLP, inspired by ResNet architectures.
            It helps in training deeper networks by allowing gradients to flow more easily.
            """
            def __init__(self, dim, dropout_rate=0.4):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                )
                self.activation = nn.GELU()

            def forward(self, x):
                residual = x
                out = self.block(x)
                out += residual
                out = self.activation(out)
                return out

        class ResMLP(nn.Module):
            """
            A ResNet-style MLP that uses a residual block.
            The hidden dimension is carefully chosen to be 211 to maximize
            parameter usage while staying under the 200k limit.
            """
            def __init__(self, input_dim, num_classes, hidden_dim=211, dropout_rate=0.4):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU()
                )
                self.res_block = ResBlock(hidden_dim, dropout_rate)
                self.output_proj = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, x):
                x = self.input_proj(x)
                x = self.res_block(x)
                return self.output_proj(x)
        
        model = ResMLP(input_dim, num_classes).to(device)

        # Parameter count check:
        # Input proj: (384 * 211 + 211) + 2*211 = 81255 + 422 = 81677
        # ResBlock: 2 * (211*211 + 211) + 2 * (2*211) = 89464 + 844 = 90308
        # Output proj: 211 * 128 + 128 = 27136
        # Total: 81677 + 90308 + 27136 = 199121 < 200,000

        # --- Hyperparameters ---
        N_EPOCHS = 250
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 1e-2
        LABEL_SMOOTHING = 0.1
        
        # --- Training Setup ---
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

        # --- Training Loop ---
        for epoch in range(N_EPOCHS):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()

        # Set model to evaluation mode before returning
        model.eval()
        return model