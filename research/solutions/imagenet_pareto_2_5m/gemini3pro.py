import torch
import torch.nn as nn
import torch.optim as optim
import copy

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=580, num_blocks=6, dropout=0.2):
        super(ResNetModel, self).__init__()
        # Input normalization to handle varying data scales
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Projection to hidden dimension
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Final classification head
        self.output_norm = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.classifier(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        """
        Train a model and return it.
        Parameter budget: 2,500,000
        Baseline Accuracy: 0.85
        """
        # 1. Setup Environment and Metadata
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        device = metadata.get("device", "cpu")
        
        # 2. Model Configuration
        # Designed to maximize capacity within 2.5M limit
        # Calculation:
        # Input: 384->580 = ~224k params
        # Blocks: 6 * (580->580) = 6 * ~338k = ~2.03M params
        # Output: 580->128 = ~75k params
        # Total: ~2.33M params (Safe margin below 2.5M)
        HIDDEN_DIM = 580
        NUM_BLOCKS = 6
        DROPOUT = 0.25
        
        model = ResNetModel(
            input_dim=input_dim, 
            num_classes=num_classes, 
            hidden_dim=HIDDEN_DIM, 
            num_blocks=NUM_BLOCKS, 
            dropout=DROPOUT
        )
        model = model.to(device)
        
        # 3. Training Hyperparameters
        EPOCHS = 70
        LR = 2e-3
        WEIGHT_DECAY = 1e-2
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # OneCycleLR is efficient for fixed epoch training
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LR, 
            epochs=EPOCHS, 
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # Label smoothing helps with synthetic data generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 4. Training Loop
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            val_acc = correct / total
            
            # Checkpoint best model
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # 5. Return Best Model
        model.load_state_dict(best_model_wts)
        model.eval()
        return model