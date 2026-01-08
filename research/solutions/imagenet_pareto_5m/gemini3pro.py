import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SubModel(nn.Module):
    def __init__(self, input_dim, num_classes, mean, std):
        super().__init__()
        # Register normalization stats as buffers so they are saved with the model
        # but not counted as trainable parameters
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        # Architecture designed to fit ~985k params
        # 5 models * 985k = ~4.925M params < 5M limit
        hidden_dim = 768
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Apply static normalization
        x = (x - self.mean) / (self.std + 1e-6)
        return self.net(x)

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        # Average the logits from all models
        # Stack shape: (num_models, batch, num_classes)
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs).mean(dim=0)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        device = metadata.get("device", "cpu")
        
        # Load all data into memory for efficiency (dataset is small)
        X_train_list, y_train_list = [], []
        for x, y in train_loader:
            X_train_list.append(x)
            y_train_list.append(y)
        X_train = torch.cat(X_train_list).to(device)
        y_train = torch.cat(y_train_list).to(device)
        
        X_val_list, y_val_list = [], []
        for x, y in val_loader:
            X_val_list.append(x)
            y_val_list.append(y)
        X_val = torch.cat(X_val_list).to(device)
        y_val = torch.cat(y_val_list).to(device)
        
        # Compute global statistics for normalization
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0)
        
        # Ensemble configuration
        num_models = 5
        epochs = 60
        batch_size = 64
        
        models = []
        
        # Train each model in the ensemble independently
        for i in range(num_models):
            # Initialize model with computed stats
            model = SubModel(input_dim, num_classes, mean, std).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()
            
            best_acc = -1.0
            best_weights = None
            
            # Create a shuffled loader for this model training run
            train_ds = TensorDataset(X_train, y_train)
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                model.train()
                for bx, by in loader:
                    # Mixup Augmentation
                    if bx.size(0) > 1:
                        alpha = 0.4
                        lam = np.random.beta(alpha, alpha)
                        idx = torch.randperm(bx.size(0)).to(device)
                        
                        mixed_x = lam * bx + (1 - lam) * bx[idx]
                        y_a, y_b = by, by[idx]
                        
                        outputs = model(mixed_x)
                        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                    else:
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # Validation to checkpoint best model
                model.eval()
                with torch.no_grad():
                    outs = model(X_val)
                    preds = outs.argmax(dim=1)
                    acc = (preds == y_val).float().mean().item()
                    
                if acc >= best_acc:
                    best_acc = acc
                    # Save state dict copy
                    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            
            # Load best weights for this member
            if best_weights is not None:
                model.load_state_dict(best_weights)
            
            models.append(model)
            
        # Return the ensemble wrapped as a single module
        return EnsembleModel(models)