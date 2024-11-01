import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # Linear layers
        self.fc1 = nn.Linear(7250, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Flatten input if needed
        x = self.flatten(x)
        
        # Apply linear transformations with activations and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        out = self.fc4(x)
        
        # out = F.softmax(out, dim=1)
        
        return out