import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import json
import os
from typing import List, Union

class TOPASXRDDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # Load the data
        self.intensities = np.load(os.path.join(data_dir, 'intensities.npy'))
        self.weight_fractions = pd.read_csv(os.path.join(data_dir, 'weight_fractions.csv'))
        self.additional_params = pd.read_csv(os.path.join(data_dir, 'all_params.csv'))
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        self.length = self.metadata['num_simulations']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extract features
        intensity = self.intensities[idx]
        
        # Extract labels (weight fractions)
        weight_fraction = self.weight_fractions.iloc[idx].values
        
        # Extract additional parameters
        add_params = self.additional_params.iloc[idx].values
        
        # Convert to tensors
        intensity_tensor = torch.from_numpy(intensity).float()
        weight_fraction_tensor = torch.from_numpy(weight_fraction).float()
        add_params_tensor = torch.from_numpy(add_params).float()
        
        return intensity_tensor, weight_fraction_tensor, add_params_tensor

def create_data_loaders(data_dirs: Union[str, List[str]], batch_size=32, num_workers=3, train_split=0.8, val_split=0.1):
    """
    Create data loaders from one or multiple data directories.
    
    Args:
        data_dirs: Either a single directory path (str) or a list of directory paths
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading
        train_split: Proportion of data to use for training
        val_split: Proportion of data to use for validation
    """
    # Convert single directory to list
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    # Create datasets for each directory
    datasets = []
    for data_dir in data_dirs:
        dataset = TOPASXRDDataset(data_dir)
        datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    # Calculate split sizes
    total_size = len(combined_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader