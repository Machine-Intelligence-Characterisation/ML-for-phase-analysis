import h5py
import numpy as np
import pandas as pd
import json
import os

def prepare_ml_data(batch_dir, output_file):
    # Load mapping
    with open(f"{batch_dir}/mapping.json", 'r') as f:
        mapping = json.load(f)

    # Load weight fractions and scale factors
    weight_fractions = pd.read_csv(f"{batch_dir}/weight_fractions.csv")
    scale_factors = pd.read_csv(f"{batch_dir}/scale_factors.csv")

    with h5py.File(output_file, 'w') as hf:
        for run_number, map_info in mapping.items():
            xy_file = f"{batch_dir}/xy_files/S1_{int(run_number):03d}.xy"
            
            if not os.path.exists(xy_file):
                print(f"Warning: {xy_file} not found. Skipping.")
                continue

            # Load intensity data
            intensity = np.loadtxt(xy_file, usecols=1)

            # Create dataset for this simulation
            sim_group = hf.create_group(f"simulation_{run_number}")
            sim_group.create_dataset("intensity", data=intensity)
            sim_group.create_dataset("weight_fraction", data=weight_fractions.iloc[map_info['weight_fraction']])
            sim_group.create_dataset("scale_factor", data=scale_factors.iloc[map_info['scale_factor']])

if __name__ == "__main__":
    prepare_ml_data("simulations/batch_001", "ml_data/training_data.h5")