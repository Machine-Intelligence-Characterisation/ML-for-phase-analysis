# TOPAS Sim

Generation of simulated X-ray diffraction patterns using TOPAS, with a focus on a three-phase system consisting of Corundum, Fluorite, and Zincite.

** You may need to play with file paths a bit to get this to work on your own machine **

## Workflow

1. **Configuration Setup**
   - Modify `config_simulation.py` to set desired parameters for simulations.

2. **Generate Simulation Parameters**
   - Run `python TOPAS_sim/scripts/generate_simulation_params.py`
   - This creates a new batch directory in the `simulations` subdirectory with:
     - Individual `simulation_params_X.txt` files for each simulation
     - `weight_fractions.csv` containing phase fractions
     - `all_params.json` with all generated parameters
     - `topas_input.inp` for TOPAS

3. **Run TOPAS Simulations**
   - Copy the generated batch folder to a TOPAS-enabled machine
   - Run the `topas_input.inp` file using TOPAS
   - This will generate XY files (diffraction patterns) in the `xy_files` subdirectory

4. **Prepare Data for Machine Learning**
   - Run `python TOPAS_sim/scripts/prepare_ml_data.py` to compile XY files and save the processed data. Change to the batch directory you want to process in __main__ of prepare_ml_data.py

