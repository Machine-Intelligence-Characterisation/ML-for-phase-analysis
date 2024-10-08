import torch
from torch.utils.data import Dataset, DataLoader
from ase.db import connect
import numpy as np

class simXRDDataset(Dataset):
    def __init__(self, db_path):
        self.db = connect(db_path)
        self.length = self.db.count()

        # For converting element list to a composition vector
        self.element_set = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        self.element_to_index = {elem: i for i, elem in enumerate(self.element_set)}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.db.get(idx + 1)  # ASE db indexing starts at 1
        
        # Extract features 
        # IMPORTANT TO REMEMBER - THIS IS CURRENTLY normalised to 100
        intensity = np.array(eval(row.intensity), dtype=np.float32)
        
        # Extract labels
        space_group = eval(getattr(row, 'tager'))[0]
        crysystem = eval(getattr(row, 'tager'))[1]
        bravis_latt_type = eval(getattr(row, 'tager'))[2]
        element = getattr(row, 'symbols')

        # This is so that we get a 0-6 output,instead of 1-7, etc. 
        # TODO: YOU WILL NEED TO ADD THE ONE BACK IN WHEN USING FOR INFERENCE ####
        crysystem -= 1
        space_group -= 1

        # Convert Bravais lattice type to numerical encoding
        # TODO: ENCODING ARE A, B, and C, PHYSICALLY EQUIVALENT? some sources say yes which confuse me
        # TODO: http://pd.chem.ucl.ac.uk/pdnn/symm3/allsgp.htm
        blt_encoding = {"P": 0, "I": 1, "F": 2, "A": 3, "B": 4, "C": 5, "R": 6}
        blt_num = blt_encoding[bravis_latt_type]

        # Convert element list to composition vector (Elements are currently one hot encoded)   
        composition = np.zeros(len(self.element_set), dtype=np.float32)
        for elem in element:
            if elem in self.element_set:
                composition[list(self.element_set).index(elem)] = 1
        
        # Convert to tensors
        intensity_tensor = torch.from_numpy(intensity)
        space_group_tensor = torch.tensor(space_group, dtype=torch.long)
        crysystem_tensor = torch.tensor(crysystem, dtype=torch.long)
        blt_tensor = torch.tensor(blt_num, dtype=torch.long)
        element_composition_tensor = torch.from_numpy(composition).float()
        
        return intensity_tensor, space_group_tensor, crysystem_tensor, blt_tensor, element_composition_tensor

# Data loaders for training
def create_training_data_loaders(train_path, val_path, test_path, batch_size=32, num_workers=3):
    train_dataset = simXRDDataset(train_path)
    val_dataset = simXRDDataset(val_path)
    test_dataset = simXRDDataset(test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# Data loader for inference
def create_inference_data_loader(inference_path, batch_size=32, num_workers=3):
    inference_dataset = simXRDDataset(inference_path)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return inference_loader