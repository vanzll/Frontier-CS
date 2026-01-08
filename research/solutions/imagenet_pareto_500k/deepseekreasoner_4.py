import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class EfficientMLP(nn.Module):
    def __init__(self, input_dim=384, num_classes=128):
        super().__init__()
        # Carefully designed to stay under 500K params
        hidden1 = 512
        hidden2 = 256
        hidden3 = 256
        
        # Main trunk
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        # Residual block 1
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc2_res = nn.Linear(hidden1, hidden2)
        
        # Residual block 2
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.fc3_res = nn.Linear(hidden2, hidden3)
        
        # Head
        self.fc4 = nn.Linear(hidden3, num_classes)
        
        self.dropout = nn.Dropout(0.2)
        
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
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Residual block 1
        identity = self.fc2_res(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x + identity)
        x = self.dropout(x)
        
        # Residual block 2
        identity = self.fc3_res(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x + identity)
        x = self.dropout(x)
        
        # Head
        x = self.fc4(x)
        return x

class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        device = torch.device(metadata.get("device", "cpu"))
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        
        # Create model
        model = EfficientMLP(input_dim, num_classes).to(device)
        
        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params > 500000:
            # Adjust architecture to meet constraint
            return self._create_smaller_model(input_dim, num_classes, device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        
        # Training loop with early stopping
        best_val_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        for epoch in range(200):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)
            
            val_acc = val_correct / val_total
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def _create_smaller_model(self, input_dim, num_classes, device):
        """Create a smaller model that definitely fits within 500K params"""
        class SmallMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 384)
                self.bn1 = nn.BatchNorm1d(384)
                self.fc2 = nn.Linear(384, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.fc4 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = F.relu(self.bn2(self.fc2(x)))
                x = self.dropout(x)
                x = F.relu(self.bn3(self.fc3(x)))
                x = self.dropout(x)
                x = self.fc4(x)
                return x
        
        model = SmallMLP(input_dim, num_classes).to(device)
        
        # Train the smaller model
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        for epoch in range(100):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model