import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = None
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        return out + residual


class EfficientNet(nn.Module):
    def __init__(self, input_dim=384, num_classes=128, param_limit=500000):
        super().__init__()
        
        # Calculate dimensions to stay under parameter budget
        base_dim = 256
        expansion = 1.2
        
        # Progressive dimension reduction with residual blocks
        dim1 = min(base_dim, input_dim)
        dim2 = int(dim1 * expansion)
        dim3 = int(dim2 * expansion)
        dim4 = int(dim3 * 0.75)  # Reduce before classifier
        
        # Adjust dimensions if parameter count is too high
        total_params = self._estimate_params(input_dim, [dim1, dim2, dim3, dim4], num_classes)
        scale = math.sqrt(param_limit / total_params) if total_params > param_limit else 1.0
        
        dim1 = int(dim1 * scale)
        dim2 = int(dim2 * scale)
        dim3 = int(dim3 * scale)
        dim4 = int(dim4 * scale)
        
        # Ensure minimum dimensions
        dim1 = max(dim1, 128)
        dim2 = max(dim2, 128)
        dim3 = max(dim3, 128)
        dim4 = max(dim4, 128)
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual blocks
        self.block1 = ResidualBlock(dim1, dim2, dropout=0.15)
        self.block2 = ResidualBlock(dim2, dim3, dropout=0.15)
        self.block3 = ResidualBlock(dim3, dim4, dropout=0.2)
        
        # Final layers
        self.final_norm = nn.BatchNorm1d(dim4)
        self.final_activation = nn.ReLU()
        self.final_dropout = nn.Dropout(0.2)
        
        # Classifier with bottleneck
        bottleneck_dim = max(dim4 // 2, 64)
        self.classifier = nn.Sequential(
            nn.Linear(dim4, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Verify parameter count
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if param_count > param_limit:
            raise ValueError(f"Model exceeds parameter limit: {param_count} > {param_limit}")
    
    def _estimate_params(self, input_dim, dims, num_classes):
        """Estimate total parameters for given dimensions."""
        params = 0
        prev_dim = input_dim
        
        for dim in dims:
            params += (prev_dim * dim + dim)  # Linear layer
            params += 2 * dim  # BatchNorm
            prev_dim = dim
        
        # Classifier bottleneck
        bottleneck_dim = max(dims[-1] // 2, 64)
        params += (dims[-1] * bottleneck_dim + bottleneck_dim)
        params += 2 * bottleneck_dim  # BatchNorm
        params += (bottleneck_dim * num_classes + num_classes)
        
        return params
    
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
        x = self.input_norm(x)
        x = self.stem(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.final_norm(x)
        x = self.final_activation(x)
        x = self.final_dropout(x)
        
        return self.classifier(x)


class Solution:
    def solve(self, train_loader, val_loader, metadata: dict = None):
        device = torch.device(metadata.get('device', 'cpu'))
        input_dim = metadata['input_dim']
        num_classes = metadata['num_classes']
        param_limit = metadata.get('param_limit', 500000)
        
        # Create model
        model = EfficientNet(input_dim, num_classes, param_limit)
        model.to(device)
        
        # Verify parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count <= param_limit, f"Model has {param_count} parameters, exceeding limit {param_limit}"
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):  # Maximum epochs
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
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping with patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        return model