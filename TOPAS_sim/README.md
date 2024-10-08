# TOPAS Sim

Generation of simulated X-ray diffraction patterns using TOPAS, with a focus on a three-phase system consisting of Corundum, Fluorite, and Zincite.

** You may need to play with file paths a bit to get this to work on your own machine **

## Project Structure

- `scripts/config_simulation.py`: Configuration file containing all simulation parameters.
- `scripts/generate_params.py`: Script to generate simulation parameters.
- `scripts/prepare_ml_data.py`: Script to prepare data for machine learning.
- `input/topas_template.inp`: TOPAS input file template.

## Workflow

1. **Configuration Setup**
   - Modify `config_simulation.py` to set desired parameters for simulations.

2. **Generate Simulation Parameters**
   - Run `python generate_params.py`
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
   - Run `prepare_ml_data.py` to compile XY files and save the processed data. Change to the batch directory you want to process in __main__

## Key Files

- `config_simulation.py`: Contains all configurable parameters, separated into Fixed, Randomized, and Not Sure categories for each phase and for global TOPAS parameters.
- `generate_params.py`: Generates randomized parameters based on `config.py`, creates individual parameter files, and prepares the TOPAS input file.
- `topas_input.inp`: The main TOPAS input file that references individual parameter files for each simulation.
- `simulation_params_X.txt`: Individual parameter files for each simulation run.
- `weight_fractions.csv`: CSV file containing the weight fractions for each simulation.
- `all_params.json`: JSON file containing all parameters for all simulations.

## Usage

1. Configure `config.py` with desired parameters and ranges.
2. Run `python generate_params.py` to create a new batch of simulations.
3. Copy the generated batch folder to a TOPAS-enabled machine.
4. Run TOPAS using the generated `topas_input.inp` file.
5. Collect the generated XY files for further processing and ML training.

## Notes

- Ensure TOPAS is properly set up on the machine where simulations will be run.
- Adjust file paths in `topas_input.inp` if necessary when moving to the TOPAS machine.
- The number of simulations and other global parameters can be adjusted in the `GENERAL` section of `config.py`.
- Review and adjust randomization ranges in `config.py` as needed for your specific use case.

## Future Work

- Implement `prepare_ml_data.py` to process generated XY files for machine learning.
- Develop and train an ML model using the prepared data.
- Consider automating the TOPAS execution process if possible.

## Requirements

- Python 3.x
- NumPy
- Pandas
- TOPAS (for running simulations)