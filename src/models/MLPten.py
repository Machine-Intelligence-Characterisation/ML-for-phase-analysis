import torch.nn as nn
import torch.nn.functional as F

# A basic MLP comparable to CNNten for comparison purposes.

# TODO: Count parameters and test it.

class MLPten(nn.Module):
    def __init__(self):
        super(MLPten, self).__init__()
        
        self.fc1 = nn.Linear(3501, 2000)
        self.fc2 = nn.Linear(2000, 1500)
        self.fc3 = nn.Linear(1500, 1000)
        self.fc4 = nn.Linear(1000, 500)
        
        self.dropout = nn.Dropout(0.33)
        
        self.fc5 = nn.Linear(500, 230)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        
        x = self.fc5(x)
        
        return x