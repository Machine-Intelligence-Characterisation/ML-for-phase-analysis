import os
import numpy as np
import pandas as pd
import json

def read_xy_file(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    return data[:, 1]  # Return only intensity values

def normalize_intensity(intensity):
    max_intensity = np.max(intensity)
    return (intensity / max_intensity) * 100

def prepare_ml_data(batch_dir):
    processed_dir = os.path.join(batch_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)

    weight_fractions = pd.read_csv(os.path.join(batch_dir, 'weight_fractions.csv'))
    
    with open(os.path.join(batch_dir, 'all_params.json'), 'r') as f:
        all_params = json.load(f)
    
    intensities = []
    params_list = []
    
    for i, params in enumerate(all_params):
        xy_file = os.path.join(batch_dir, 'xy_files', f'S1_{i}.xy')
        
        if os.path.exists(xy_file):
            intensity = read_xy_file(xy_file)
            normalized_intensity = normalize_intensity(intensity)
            intensities.append(normalized_intensity)
            params_list.append(params)
        else:
            print(f"Warning: XY file not found for simulation {i}")
    
    intensities = np.array(intensities)
    params_df = pd.DataFrame(params_list)
    
    # Save the data in the processed_data subdirectory
    np.save(os.path.join(processed_dir, 'intensities.npy'), intensities)
    weight_fractions.to_csv(os.path.join(processed_dir, 'weight_fractions.csv'), index=False)
    params_df.to_csv(os.path.join(processed_dir, 'all_params.csv'), index=False)
    
    # Save metadata about the dataset
    metadata = {
        "num_simulations": len(params_list),
        "intensity_shape": intensities.shape,
        "params_columns": list(params_df.columns),
    }
    with open(os.path.join(processed_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data preparation complete. Files saved in {processed_dir}")
    print(f"Intensities shape: {intensities.shape}")
    print(f"Weight fractions shape: {weight_fractions.shape}")
    print(f"All parameters shape: {params_df.shape}")

if __name__ == "__main__":
    batch_dir = "TOPAS_sim/simulations/batch_20241010_212052_NumSims10000"
    prepare_ml_data(batch_dir)