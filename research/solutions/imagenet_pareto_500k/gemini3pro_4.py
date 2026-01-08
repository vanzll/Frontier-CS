import torch
import torch.nn as nn
import torch.optim as optim
import copy

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ParamEfficientNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=288):
        super().__init__()
        # Initial normalization to handle varying feature scales
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Project to hidden dimension
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Deep residual processing (2 blocks = 4 layers)
        self.layer1 = ResBlock(hidden_dim, dropout=0.2)
        self.layer2 = ResBlock(hidden_dim, dropout=0.2)
        
        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        torch.set_num_threads(8)
        
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        device = metadata.get("device", "cpu")
        
        # H=288 results in ~485k parameters, safely under the 500k limit
        # Calculation:
        # Input BN: 768
        # Stem: 384*288 + 288 + 2*288 = 111,456
        # Block 1: 2*(288*288 + 288) + 2*2*288 = 167,616
        # Block 2: 167,616
        # Head: 288*128 + 128 = 36,992
        # Total: ~484,448
        hidden_dim = 288
        
        model = ParamEfficientNet(input_dim, num_classes, hidden_dim).to(device)
        
        # Optimization hyperparameters
        epochs = 100
        lr = 1e-3
        weight_decay = 1e-2
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Label smoothing helps with generalization on small datasets
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 32
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        best_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())
        
        for epoch in range(epochs):
            model.train()
            
            # Decaying noise injection for feature augmentation
            current_noise = 0.05 * (1.0 - epoch / epochs)
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add Gaussian noise to features
                if current_noise > 0.001:
                    noise = torch.randn_like(inputs) * current_noise
                    inputs = inputs + noise
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validation
            if epoch % 2 == 0 or epoch > epochs - 15:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)
                
                acc = correct / total if total > 0 else 0
                
                if acc >= best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(model.state_dict())
        
        model.load_state_dict(best_state)
        return model