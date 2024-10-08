import torch.nn as nn
import torch.nn.functional as F

# TODO: Deprecate this once you understand why it doesn't work.

# I make a somewhat random model
# It over fits

# Input shape: torch.Size([32, 1, 3501])
# After conv1_1: torch.Size([32, 8, 3501])
# After conv1_2: torch.Size([32, 16, 3501])
# After conv1_3: torch.Size([32, 24, 3501])
# After conv1_4: torch.Size([32, 32, 3501])
# After conv1_5: torch.Size([32, 40, 3501])
# After conv1_6: torch.Size([32, 16, 3501])
# After avg_pool: torch.Size([32, 16, 1750])
# After conv2_1: torch.Size([32, 24, 1750])
# After conv2_2: torch.Size([32, 32, 1750])
# After conv2_3: torch.Size([32, 40, 1750])
# After conv2_4: torch.Size([32, 48, 1750])
# After conv2_5: torch.Size([32, 1, 1750])
# After flattening: torch.Size([32, 1750])
# After fc1: torch.Size([32, 512])
# After fc_spg_1: torch.Size([32, 512])
# After fc_spg_2 (Space Group output): torch.Size([32, 230])
# After fc_crysystem_1: torch.Size([32, 256])
# After fc_crysystem_2 (Crystal System output): torch.Size([32, 7])
# After fc_blt_1: torch.Size([32, 256])
# After fc_blt_2 (Bravais Lattice output): torch.Size([32, 6])
# After fc_composition_1: torch.Size([32, 512])
# After fc_composition_2: torch.Size([32, 512])
# After fc_composition_3 (Composition output): torch.Size([32, 118])

class Jackson(nn.Module):
    def __init__(self):
        super(Jackson, self).__init__()
        
        # First CNN layers
        self.conv1_1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv1d(16, 24, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv1d(24, 32, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(32, 40, kernel_size=3, padding=1)
        self.conv1_6 = nn.Conv1d(40, 16, kernel_size=1)
        
        # Average pooling
        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second CNN layers
        self.conv2_1 = nn.Conv1d(16, 24, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(24, 32, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(32, 40, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv1d(40, 48, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv1d(48, 1, kernel_size=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1750, 512)
        
        # Space Group
        self.fc_spg_1 = nn.Linear(512, 512)
        self.fc_spg_2 = nn.Linear(512, 230)

        # Crystal System
        self.fc_crysystem_1 = nn.Linear(512, 256)
        self.fc_crysystem_2 = nn.Linear(256, 7)

        # Bravis Lattice
        self.fc_blt_1 = nn.Linear(512, 256)
        self.fc_blt_2 = nn.Linear(256, 6)

        # Composition
        self.fc_composition_1 = nn.Linear(512, 512)
        self.fc_composition_2 = nn.Linear(512, 512)
        self.fc_composition_3 = nn.Linear(512, 118)
        
        # Dropout
        self.dropout = nn.Dropout(0.33)

        self.flatten = nn.Flatten()

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        # First CNN block
        x = F.relu(self.conv1_1(x))
        #print(f"After conv1_1: {x.shape}")
        x = F.relu(self.conv1_2(x))
        #print(f"After conv1_2: {x.shape}")
        x = F.relu(self.conv1_3(x))
        #print(f"After conv1_3: {x.shape}")
        x = F.relu(self.conv1_4(x))
        #print(f"After conv1_4: {x.shape}")
        x = F.relu(self.conv1_5(x))
        #print(f"After conv1_5: {x.shape}")
        x = self.conv1_6(x)
        #print(f"After conv1_6: {x.shape}")
        
        # Average pooling
        x = self.avg_pool(x)
        #print(f"After avg_pool: {x.shape}")
        
        # Second CNN block
        x = F.relu(self.conv2_1(x))
        #print(f"After conv2_1: {x.shape}")
        x = F.relu(self.conv2_2(x))
        #print(f"After conv2_2: {x.shape}")
        x = F.relu(self.conv2_3(x))
        #print(f"After conv2_3: {x.shape}")
        x = F.relu(self.conv2_4(x))
        #print(f"After conv2_4: {x.shape}")
        x = self.conv2_5(x)
        #print(f"After conv2_5: {x.shape}")
        
        # Flatten the output for the fully connected layers
        x = self.flatten(x)
        #print(f"After flattening: {x.shape}")
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        #print(f"After fc1: {x.shape}")
        x = self.dropout(x)
        
        # Space Group
        spg = F.relu(self.fc_spg_1(x))
        #print(f"After fc_spg_1: {spg.shape}")
        spg = self.dropout(spg)
        out_spg = self.fc_spg_2(spg)
        #print(f"After fc_spg_2 (Space Group output): {out_spg.shape}")
        
        # Crystal System
        crysystem = F.relu(self.fc_crysystem_1(x))
        #print(f"After fc_crysystem_1: {crysystem.shape}")
        crysystem = self.dropout(crysystem)
        out_crysystem = self.fc_crysystem_2(crysystem)
        #print(f"After fc_crysystem_2 (Crystal System output): {out_crysystem.shape}")
        
        # Bravais Lattice
        blt = F.relu(self.fc_blt_1(x))
        #print(f"After fc_blt_1: {blt.shape}")
        blt = self.dropout(blt)
        out_blt = self.fc_blt_2(blt)
        #print(f"After fc_blt_2 (Bravais Lattice output): {out_blt.shape}")
        
        # Composition
        composition = F.relu(self.fc_composition_1(x))
        #print(f"After fc_composition_1: {composition.shape}")
        composition = self.dropout(composition)
        composition = F.relu(self.fc_composition_2(composition))
        #print(f"After fc_composition_2: {composition.shape}")
        composition = self.dropout(composition)
        out_composition = self.fc_composition_3(composition)
        #print(f"After fc_composition_3 (Composition output): {out_composition.shape}")
        
        return {
            'spg': out_spg,
            'crysystem': out_crysystem,
            'blt': out_blt,
            'composition': out_composition
        }