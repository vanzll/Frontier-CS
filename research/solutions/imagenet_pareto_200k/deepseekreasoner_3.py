import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


class EfficientMLP(nn.Module):
    def __init__(self, input_dim=384, num_classes=128):
        super().__init__()
        # Carefully designed architecture to stay under 200K parameters
        # while maximizing representational capacity
        hidden1 = 384  # Same as input for residual connection
        hidden2 = 320  # Reduced for parameter efficiency
        hidden3 = 256  # Further reduction
        hidden4 = 192  # Bottleneck layer
        
        # Main sequential path with residual connections
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden1, bias=False),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Residual projection for layer2
        self.residual2 = nn.Linear(hidden1, hidden2, bias=False)
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, hidden3, bias=False),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Residual projection for layer3
        self.residual3 = nn.Linear(hidden2, hidden3, bias=False)
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden3, hidden4, bias=False),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Layer 1
        out1 = self.layer1(x)
        
        # Layer 2 with residual
        out2 = self.layer2(out1)
        res2 = self.residual2(out1)
        out2 = out2 + res2
        
        # Layer 3 with residual
        out3 = self.layer3(out2)
        res3 = self.residual3(out2)
        out3 = out3 + res3
        
        # Layer 4
        out4 = self.layer4(out3)
        
        # Classification
        return self.classifier(out4)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None) -> torch.nn.Module:
        # Extract metadata
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        device = metadata.get("device", "cpu")
        
        # Create model
        model = EfficientMLP(input_dim, num_classes).to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > 200000:
            # If over limit, create simpler model
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ).to(device)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        
        # Early stopping
        best_val_acc = 0.0
        patience = 20
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        num_epochs = 200
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                break
            
            # Timeout check (1 hour total)
            if time.time() - start_time > 3500:  # Leave 100 seconds margin
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model