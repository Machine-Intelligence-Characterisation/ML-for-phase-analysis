import torch.nn as nn
import torch.nn.functional as F

# The FCN from: https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300140
# As described here: https://github.com/socoolblue/Advanced_XRD_Analysis/blob/main/XRD_analysis.ipynb
# Accessed 28/07/2024

# Modifications:
# I alter some of the final conv layers to account for the smaller input data (1,70??)

class smallFCN(nn.Module):
    def __init__(self):
        super(smallFCN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=6, padding=2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=6, padding=2)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=6, padding=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=6, padding=2)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=6, padding=2)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=6, padding=2)
        self.conv7 = nn.Conv1d(64, 128, kernel_size=6, padding=2)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=6, padding=2)
        self.conv9 = nn.Conv1d(128, 256, kernel_size=6, padding=2)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=6, padding=2)

        self.final_conv_1 = nn.Conv1d(256, 128, kernel_size=6, padding=2)
        self.final_conv_2 = nn.Conv1d(128, 64, kernel_size=6, padding=2)
        self.final_conv_3 = nn.Conv1d(64, 32, kernel_size=6, padding=2)
        self.final_conv_4 = nn.Conv1d(32, 16, kernel_size=6, padding=2)
        self.final_conv_5 = nn.Conv1d(16, 3, kernel_size=6, padding=2)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Apply convolutions, activations, pooling, and dropout
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))
        x = self.dropout(self.pool(F.relu(self.conv5(x))))
        x = self.dropout(self.pool(F.relu(self.conv6(x))))
        x = self.dropout(self.pool(F.relu(self.conv7(x))))
        x = self.dropout(self.pool(F.relu(self.conv8(x))))
        x = self.dropout(self.pool(F.relu(self.conv9(x))))
        x = self.dropout(self.pool(F.relu(self.conv10(x))))

        out = self.dropout(F.relu(self.final_conv_1(x)))
        out = self.dropout(F.relu(self.final_conv_2(out)))
        out = self.dropout(F.relu(self.final_conv_3(out)))
        out = self.dropout(F.relu(self.final_conv_4(out)))
        out = self.final_conv_5(out)
        
        out = self.flatten(out)

        out = F.softmax(out, dim=1)

        return out