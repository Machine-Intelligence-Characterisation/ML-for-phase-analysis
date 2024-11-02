import numpy as np
import pandas as pd
import json
import os
from config_simulation import GENERAL, TOPAS, PHASES

def compute_scale_factors(weight_fraction):

    # Constants for scale factor calculations
    # ZM values
    ZM_Cor, ZM_Flu, ZM_Zin = 611.768, 312.299, 162.817
    # V values
    V_Cor, V_Flu, V_Zin = 254.123, 162.023, 47.290
    # LAC values
    LAC_Cor, LAC_Flu, LAC_Zin = 126.283, 303.163, 278.222
    # Density values
    density_Cor, density_Flu, density_Zin = 3.997530, 3.200700, 5.717180
    # MAC values
    MAC_Cor = LAC_Cor / density_Cor
    MAC_Flu = LAC_Flu / density_Flu
    MAC_Zin = LAC_Zin / density_Zin
    # External K value
    K_external = 427.6

    # Compute mixture MAC
    Um = (weight_fraction[0] * MAC_Cor + 
          weight_fraction[1] * MAC_Flu + 
          weight_fraction[2] * MAC_Zin)
    
    # Compute scale factors
    S_Cor = (weight_fraction[0] * K_external) / (ZM_Cor * V_Cor * Um)
    S_Flu = (weight_fraction[1] * K_external) / (ZM_Flu * V_Flu * Um)
    S_Zin = (weight_fraction[2] * K_external) / (ZM_Zin * V_Zin * Um)
    
    return S_Cor, S_Flu, S_Zin

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
                if key == 'name' or key == 'scale_var':  # Skip scale_var as we'll compute it
                    continue
                if isinstance(value, dict):
                    if value['randomize']:
                        params[f"{phase['name']}_{key}"], params[f"{phase['name']}_{key}_min"], params[f"{phase['name']}_{key}_max"] = generate_random_value(value)
                    else:
                        params[f"{phase['name']}_{key}"] = generate_random_value(value)
                else:
                    params[f"{phase['name']}_{key}"] = value

        # Generate weight fractions and compute scale factors
        weight_fraction = np.random.dirichlet([1, 1, 1])
        scale_factors = compute_scale_factors(weight_fraction)
        
        # Set weight fractions and scale factors
        params['Corundum_scale_var'] = scale_factors[0]
        params['Fluorite_scale_var'] = scale_factors[1]
        params['Zincite_scale_var'] = scale_factors[2]
        
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