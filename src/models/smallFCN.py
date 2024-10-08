import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The FCN from: https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300140
# As described here: https://github.com/socoolblue/Advanced_XRD_Analysis/blob/main/XRD_analysis.ipynb
# Accessed 28/07/2024

# Modifications:
# I alter some of the final conv layers to account for the smaller input data (1,3501)
# I augment a second module to allow for multi-task outputs.

# TODO: Check it is implememnted correctly..
# TODO: Implement the self attention model correctly

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

        self.spg_conv_1 = nn.Conv1d(256, 256, kernel_size=6, padding=2)
        self.spg_conv_2 = nn.Conv1d(256, 230, kernel_size=1)
        
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

        spg_out = self.dropout(F.relu(self.spg_conv_1(x)))
        spg_out = self.spg_conv_2(spg_out)
        spg_out = self.flatten(spg_out)
        
        return spg_out

class smallFCN_MultiTask(nn.Module):
    def __init__(self):
        super(smallFCN_MultiTask, self).__init__()
        
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

        # Multi-task outs
        self.crysystem_conv_1 = nn.Conv1d(256, 64, kernel_size=6, padding=2)
        self.crysystem_conv_2 = nn.Conv1d(64, 7, kernel_size=1)

        self.blt_conv_1 = nn.Conv1d(256, 64, kernel_size=6, padding=2)
        self.blt_conv_2 = nn.Conv1d(64, 6, kernel_size=1)

        self.spg_conv_1 = nn.Conv1d(256, 256, kernel_size=6, padding=2)
        self.spg_conv_2 = nn.Conv1d(256, 230, kernel_size=1)

        self.composition_conv_1 = nn.Conv1d(256, 256, kernel_size=6, padding=2)
        self.composition_conv_2 = nn.Conv1d(256, 118, kernel_size=1)
        
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

        # Multi-task specific layers
        crysystem_out = self.dropout(F.relu(self.crysystem_conv_1(x)))
        crysystem_out = self.crysystem_conv_2(crysystem_out)
        crysystem_out = self.flatten(crysystem_out)

        blt_out = self.dropout(F.relu(self.blt_conv_1(x)))
        blt_out = self.blt_conv_2(blt_out)
        blt_out = self.flatten(blt_out)

        spg_out = self.dropout(F.relu(self.spg_conv_1(x)))
        spg_out = self.spg_conv_2(spg_out)
        spg_out = self.flatten(spg_out)

        composition_out = self.dropout(F.relu(self.composition_conv_1(x)))
        composition_out = self.composition_conv_2(composition_out)
        composition_out = self.flatten(composition_out)
        
        return {
            'spg': spg_out,
            'crysystem': crysystem_out,
            'blt': blt_out,
            'composition': composition_out
        }
    
class smallFCN_SelfAttention_MultiTask(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()

        # Self attention parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads

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
        self.conv10 = nn.Conv1d(256, embed_dim, kernel_size=6, padding=2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.33)

        # Self-attention components
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # 1D positional encoding
        self.pos_encoding = nn.Parameter(self.create_1d_positional_encoding(), requires_grad=False)

        # Multi-task outs
        self.crysystem_conv_1 = nn.Conv1d(embed_dim, 64, kernel_size=6, padding=2)
        self.crysystem_conv_2 = nn.Conv1d(64, 7, kernel_size=1)

        self.blt_conv_1 = nn.Conv1d(embed_dim, 64, kernel_size=6, padding=2)
        self.blt_conv_2 = nn.Conv1d(64, 7, kernel_size=1)

        self.spg_conv_1 = nn.Conv1d(embed_dim, 256, kernel_size=6, padding=2)
        self.spg_conv_2 = nn.Conv1d(256, 230, kernel_size=1)

        self.composition_conv_1 = nn.Conv1d(embed_dim, 256, kernel_size=6, padding=2)
        self.composition_conv_2 = nn.Conv1d(256, 118, kernel_size=1)

        self.flatten = nn.Flatten()

    def create_1d_positional_encoding(self):
        length = 2  # The length of your sequence after conv10
        pe = torch.zeros(length, self.embed_dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

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

        # Apply self-attention
        b, c, l = x.shape  # batch_size, channels, length
        x = x.permute(0, 2, 1)  # (batch_size, length, channels)
        x = x + self.pos_encoding.unsqueeze(0)

        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.layer_norm2(x)

        x = x.permute(0, 2, 1)  # (batch_size, channels, length)

        # Multi-task specific layers
        crysystem_out = self.dropout(F.relu(self.crysystem_conv_1(x)))
        crysystem_out = self.crysystem_conv_2(crysystem_out)
        crysystem_out = self.flatten(crysystem_out)

        blt_out = self.dropout(F.relu(self.blt_conv_1(x)))
        blt_out = self.blt_conv_2(blt_out)
        blt_out = self.flatten(blt_out)

        spg_out = self.dropout(F.relu(self.spg_conv_1(x)))
        spg_out = self.spg_conv_2(spg_out)
        spg_out = self.flatten(spg_out)

        composition_out = self.dropout(F.relu(self.composition_conv_1(x)))
        composition_out = self.composition_conv_2(composition_out)
        composition_out = self.flatten(composition_out)

        return {
            'spg': spg_out,
            'crysystem': crysystem_out,
            'blt': blt_out,
            'composition': composition_out
        }
    
class experimentalFCN(nn.Module):
    def __init__(self):
        super(experimentalFCN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for each task
        self.blt_out = nn.Linear(512, 7)
        self.crysystem_out = nn.Linear(512, 7)
        self.spg_out = nn.Linear(512, 230)
        self.composition_out = nn.Linear(512, 118)
        
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Multi-task outputs
        spg_out = self.spg_out(x)
        crysystem_out = self.crysystem_out(x)
        blt_out = self.blt_out(x)
        composition_out = self.composition_out(x)
        
        return {
            'spg': spg_out,
            'crysystem': crysystem_out,
            'blt': blt_out,
            'composition': composition_out
        }