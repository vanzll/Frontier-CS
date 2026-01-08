import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class SubModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(SubModel, self).__init__()
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3 Residual Blocks
        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)
        self.res3 = ResidualBlock(hidden_dim, dropout)
        
        # Output head
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.output_layer(x)

class EnsembleModel(nn.Module):
    def __init__(self, models, mean, std):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / (self.std + 1e-6)
        
        # Average predictions from all models
        outputs = [m(x) for m in self.models]
        return torch.stack(outputs).mean(dim=0)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # Default metadata handling
        if metadata is None:
            metadata = {}
            
        input_dim = metadata.get("input_dim", 384)
        num_classes = metadata.get("num_classes", 128)
        param_limit = metadata.get("param_limit", 1000000)
        device = metadata.get("device", "cpu")
        
        # Helper to extract data from loaders to tensors for speed
        def extract_data(loader):
            all_x, all_y = [], []
            for x, y in loader:
                all_x.append(x)
                all_y.append(y)
            return torch.cat(all_x).to(device), torch.cat(all_y).to(device)
            
        X_train, Y_train = extract_data(train_loader)
        X_val, Y_val = extract_data(val_loader)
        
        # Compute stats for normalization
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0)
        
        # Architecture hyperparameters
        hidden_dim = 256
        dropout = 0.3
        
        # Calculate parameters per model to determine ensemble size
        # FC1 + BN1 + 3*Block + FC_Out
        # Block = FC + BN
        # FC(in, out) = in*out + out
        # BN(dim) = 2*dim
        p_fc1 = input_dim * hidden_dim + hidden_dim
        p_bn1 = 2 * hidden_dim
        p_block = hidden_dim * hidden_dim + hidden_dim + 2 * hidden_dim
        p_out = hidden_dim * num_classes + num_classes
        
        params_per_model = p_fc1 + p_bn1 + 3 * p_block + p_out
        
        # Determine number of models that fit in budget with safety margin
        safety_limit = int(param_limit * 0.99)
        num_models = max(1, safety_limit // params_per_model)
        
        # Training configuration
        batch_size = 64
        epochs = 45 
        lr = 0.001
        
        # Prepare data loader
        train_ds = TensorDataset(X_train, Y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_models = []
        
        for i in range(num_models):
            model = SubModel(input_dim, hidden_dim, num_classes, dropout).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            best_acc = -1.0
            best_state = copy.deepcopy(model.state_dict())
            
            for epoch in range(epochs):
                model.train()
                for bx, by in train_dl:
                    # Normalize inside training loop
                    bx_norm = (bx - mean) / (std + 1e-6)
                    
                    # Mixup augmentation
                    if np.random.random() < 0.5:
                        alpha = 0.4
                        lam = np.random.beta(alpha, alpha)
                        idx = torch.randperm(bx_norm.size(0)).to(device)
                        bx_mixed = lam * bx_norm + (1 - lam) * bx_norm[idx]
                        by_a, by_b = by, by[idx]
                        
                        optimizer.zero_grad()
                        outputs = model(bx_mixed)
                        loss = lam * criterion(outputs, by_a) + (1 - lam) * criterion(outputs, by_b)
                        loss.backward()
                        optimizer.step()
                    else:
                        optimizer.zero_grad()
                        outputs = model(bx_norm)
                        loss = criterion(outputs, by)
                        loss.backward()
                        optimizer.step()
                
                scheduler.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_input = (X_val - mean) / (std + 1e-6)
                    val_out = model(val_input)
                    preds = val_out.argmax(dim=1)
                    acc = (preds == Y_val).float().mean().item()
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_state = copy.deepcopy(model.state_dict())
            
            # Load best weights and store
            model.load_state_dict(best_state)
            model.eval()
            best_models.append(model)
            
        # Create and return ensemble
        final_model = EnsembleModel(best_models, mean, std)
        return final_model