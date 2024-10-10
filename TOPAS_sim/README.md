# TOPAS Sim

Generation of simulated X-ray diffraction patterns using TOPAS, with a focus on a three-phase system consisting of Corundum, Fluorite, and Zincite.

** You may need to play with file paths a bit to get this to work on your own machine **

## Workflow

1. **Configuration Setup**
   - Modify `scripts/config_simulation.py` to set desired parameters for simulations.

2. **Generate Simulation Parameters**
   - Run `python TOPAS_sim/scripts/generate_simulation_params.py`
   - This creates a new batch directory in the `simulations` subdirectory with:
     - The combined `simulation_params.txt` file
     - `weight_fractions.csv` containing phase fractions
     - `all_params.json` with all the used parameters
     - `topas_input.inp` to run on TOPAS

3. **Run TOPAS Simulations**
   - Copy the generated batch folder to a TOPAS-enabled machine
   - Run the `topas_input.inp` file using TOPAS
   - This will generate XY files (diffraction patterns) in the `xy_files` subdirectory

4. **Prepare Data for Machine Learning**
   - Run `python TOPAS_sim/scripts/prepare_ml_data.py` to compile XY files and save the processed data. Your ML data_loader should point to this processed_data folder.

