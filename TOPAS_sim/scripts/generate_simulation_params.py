import numpy as np
import pandas as pd
import json
import os
from config_simulation import GENERAL, TOPAS, PHASES

def generate_random_value(param):
    if param['randomize']:
        return np.random.uniform(param['range'][0], param['range'][1])
    return param['value']

def generate_simulation_params(sim_number):
    params = {}

    # Generate TOPAS parameters
    for key, value in TOPAS.items():
        if isinstance(value, dict):
            params[key] = generate_random_value(value)
        else:
            params[key] = value

    # Generate phase parameters
    for phase in PHASES:
        for key, value in phase.items():
            if key == 'name':
                continue
            if isinstance(value, dict):
                params[f"{phase['name']}_{key}"] = generate_random_value(value)
            else:
                params[f"{phase['name']}_{key}"] = value

    # Generate weight fractions
    weight_fractions = np.random.dirichlet([1, 1, 1])
    for i, phase in enumerate(PHASES):
        params[f"{phase['name']}_weight_fraction"] = weight_fractions[i]

    return params

def generate_batch(batch_number):
    batch_dir = GENERAL['output_directory']
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(f"{batch_dir}/xy_files", exist_ok=True)
    os.makedirs(f"{batch_dir}/simulation_params", exist_ok=True)

    all_params = []
    weight_fractions = []

    for i in range(GENERAL['num_simulations']):
        sim_params = generate_simulation_params(i+1)
        all_params.append(sim_params)
        weight_fractions.append([
            sim_params['Corundum_weight_fraction'],
            sim_params['Fluorite_weight_fraction'],
            sim_params['Zincite_weight_fraction']
        ])

        # Write individual simulation parameter file
        with open(f"{batch_dir}/simulation_params/simulation_params_{i+1}.txt", 'w') as f:
            for key, value in sim_params.items():
                f.write(f"{key} = {value}\n")

    # Save weight fractions
    pd.DataFrame(weight_fractions, columns=['Corundum', 'Fluorite', 'Zincite']).to_csv(f"{batch_dir}/weight_fractions.csv", index=False)

    # Save all parameters for potential later use
    with open(f"{batch_dir}/all_params.json", 'w') as f:
        json.dump(all_params, f, indent=2)

    # Generate TOPAS input file
    generate_topas_input(batch_dir)

def generate_topas_input(batch_dir):
    with open(TOPAS['template_file'], 'r') as f:
        template = f.read()

    # Here you might need to adjust the template replacement based on your specific TOPAS template structure
    topas_input = template.replace("NUM_SIMULATIONS", str(GENERAL['num_simulations']))

    with open(f"{batch_dir}/topas_input.inp", 'w') as f:
        f.write(topas_input)

if __name__ == "__main__":
    np.random.seed(GENERAL['random_seed'])
    generate_batch(1)