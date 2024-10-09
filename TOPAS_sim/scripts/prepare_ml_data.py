import os
import numpy as np
import pandas as pd
import json
from config_simulation import GENERAL, TOPAS

# TODO: The normalise function finds the max intensity each call. If it is slow this is a quick speed up.
  
def read_xy_file(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    return data[:, 1]  # Return only intensity values

def normalize_intensity(intensity):
    max_intensity = np.max(intensity)
    return (intensity / max_intensity) * 100

def prepare_ml_data(batch_dir):
    # Create a subdirectory for processed data
    processed_dir = os.path.join(batch_dir, 'processed_data')
    os.makedirs(processed_dir, exist_ok=True)

    weight_fractions = pd.read_csv(os.path.join(batch_dir, 'weight_fractions.csv'))
    
    with open(os.path.join(batch_dir, 'all_params.json'), 'r') as f:
        all_params = json.load(f)
    
    intensities = []
    additional_params = []
    
    for i in range(GENERAL['num_simulations']):
        xy_file = os.path.join(batch_dir, 'xy_files', f'S1_{i+1}.xy')
        
        if os.path.exists(xy_file):
            intensity = read_xy_file(xy_file)
            normalized_intensity = normalize_intensity(intensity)
            intensities.append(normalized_intensity)
            
            params = all_params[i]
            additional_params.append([
                params['Zero_Error'],
                params['Specimen_Displacement'],
                params['Absorption'],
                params['Slit_Width'],
                params['primary_soller_angle'],
                params['secondary_soller_angle']
            ])
        else:
            print(f"Warning: XY file not found for simulation {i+1}")
    
    intensities = np.array(intensities)
    additional_params = np.array(additional_params)
    
    # Save the data in the processed_data subdirectory
    np.save(os.path.join(processed_dir, 'intensities.npy'), intensities)
    weight_fractions.to_csv(os.path.join(processed_dir, 'weight_fractions.csv'), index=False)
    pd.DataFrame(additional_params, columns=['Zero_Error', 'Specimen_Displacement', 'Absorption', 'Slit_Width', 
                                             'primary_soller_angle', 'secondary_soller_angle']).to_csv(
        os.path.join(processed_dir, 'additional_params.csv'), index=False)
    
    # Save metadata about the dataset
    metadata = {
        "num_simulations": GENERAL['num_simulations'],
        "intensity_shape": intensities.shape,
        "two_theta_start": TOPAS['two_theta_start']['value'],
        "two_theta_end": TOPAS['two_theta_end']['value'],
        "two_theta_step": TOPAS['two_theta_step']['value'],
    }
    with open(os.path.join(processed_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data preparation complete. Files saved in {processed_dir}")
    print(f"Intensities shape: {intensities.shape}")
    print(f"Weight fractions shape: {weight_fractions.shape}")
    print(f"Additional parameters shape: {additional_params.shape}")

if __name__ == "__main__":
    batch_dir = "TOPAS_sim/simulations/batch_20241008_173439_TEST"
    prepare_ml_data(batch_dir)