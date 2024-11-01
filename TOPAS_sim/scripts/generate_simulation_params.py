import numpy as np
import pandas as pd
import json
import os
from config_simulation import GENERAL, TOPAS, PHASES

# TODO: get rid of ad hoc wavelength creation when processing config.

def generate_random_value(param):
    if param['randomize']:
        value = np.random.uniform(param['range'][0], param['range'][1])
        return value, param['range'][0], param['range'][1]
    return param['value']

def generate_simulation_params():
    all_params = []
    weight_fractions = []

    for i in range(GENERAL['num_simulations']):
        params = {'seq_no': i}

        # Generate TOPAS parameters
        for key, value in TOPAS.items():
            if isinstance(value, dict):
                if key == 'wavelength_distribution_var':
                    for idx, dist in enumerate(value['value']):
                        for subkey, subvalue in dist.items():
                            params[f'wavelength_{subkey}_{idx}'] = subvalue
                elif value['randomize']:
                    params[f"{key}"], params[f"{key}_min"], params[f"{key}_max"] = generate_random_value(value)
                else:
                    params[key] = generate_random_value(value)
            else:
                params[key] = value

        # Generate phase parameters
        for phase in PHASES:
            for key, value in phase.items():
                if key == 'name':
                    continue
                if isinstance(value, dict):
                    if value['randomize']:
                        params[f"{phase['name']}_{key}"], params[f"{phase['name']}_{key}_min"], params[f"{phase['name']}_{key}_max"] = generate_random_value(value)
                    else:
                        params[f"{phase['name']}_{key}"] = generate_random_value(value)
                else:
                    params[f"{phase['name']}_{key}"] = value

        # Generate weight fractions
        weight_fraction = np.random.dirichlet([1, 1, 1])
        for j, phase in enumerate(PHASES):
            params[f"{phase['name']}_weight_fraction"] = weight_fraction[j]

        all_params.append(params)
        weight_fractions.append(weight_fraction)

    return all_params, weight_fractions

def generate_simulation_params_file(batch_dir, all_params):
    with open(f"{batch_dir}/simulation_params.txt", 'w') as f:
        # Write the header
        f.write("#list")
        for key in all_params[0].keys():
            f.write(f" {key}")
        f.write("\n{\n")
        
        # Write the data
        for params in all_params:
            f.write(" ".join(str(value) for value in params.values()))
            f.write("\n")
        
        f.write("}\n")
        f.write(f"num_runs {GENERAL['num_simulations']}\n")

def generate_topas_input(batch_dir):
    with open(GENERAL['template_file'], 'r') as f:
        template = f.read()

    with open(f"{batch_dir}/topas_input.inp", 'w') as f:
        f.write(template)

def generate_batch():
    batch_dir = GENERAL['output_directory']
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(f"{batch_dir}/xy_files", exist_ok=True)

    all_params, weight_fractions = generate_simulation_params()

    # Save weight fractions
    pd.DataFrame(weight_fractions, columns=['Corundum', 'Fluorite', 'Zincite']).to_csv(f"{batch_dir}/weight_fractions.csv", index=False)

    # Save all parameters for potential later use
    with open(f"{batch_dir}/all_params.json", 'w') as f:
        json.dump(all_params, f, indent=2)

    # Generate simulation_params.txt file
    generate_simulation_params_file(batch_dir, all_params)

    # Generate TOPAS input file
    generate_topas_input(batch_dir)

if __name__ == "__main__":
    np.random.seed(GENERAL['random_seed'])
    generate_batch()