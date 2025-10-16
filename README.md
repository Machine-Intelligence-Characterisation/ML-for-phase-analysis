# ML for XRD Quantitative Phase Analysis

Training and evaluation pipeline for neural networks performing quantitative phase analysis (QPA) on X-ray diffraction patterns. Benchmarked against the Madsen 2001 round-robin study.

## Overview

This repository contains code to:
1. Train neural networks for XRD phase quantification
2. Evaluate model performance on synthetic test data
3. Benchmark predictions against the Madsen 2001 expert dataset

**Current implementation:** Three-phase system (Corundum, Fluorite, Zincite)  
**Input:** XRD intensity patterns (7,250 datapoints, 5-150° 2θ, 0.02° steps)  
**Output:** Weight fraction predictions for each phase

Multiple model architectures are implemented (MLPs, FCNs, tc). Best results achieved with MLP architectures (see Results section).

## The Madsen Benchmark

In 2001, Madsen et al. conducted a round-robin study where 128 expert crystallographers analyzed 8 samples containing known mixtures of three phases. Experts used their preferred analysis methods including:
- Rietveld refinement with various instruments (lab X-ray, neutron, synchrotron)
- Classical pattern decomposition (CPD)
- Various software packages

The study quantifies real-world expert variability and provides ground truth for comparison. If an ML model performs within the range of expert variation, it demonstrates practical viability.

**Reference:** Madsen, I. C., et al. (2001). "Outcomes of the International Union of Crystallography Commission on Powder Diffraction Round Robin on Quantitative Phase Analysis." *Journal of Applied Crystallography*, 34(4), 409-426. https://doi.org/10.1107/S0021889801007476

## Results

Best model (MLP7): Mean absolute error ~0.85% on Madsen samples

### Benchmark Comparison

[TODO: Results thingy ]

ML predictions fall within expert standard deviation across all analysis methods.

### Model Performance

[TODO: predicted vs actual weight fractions]

Strong correlation across the compositional space. Points on diagonal = perfect prediction.

## Repository Structure

```
├── analysis/
│   ├── Madsen_Round_Robin_2004/     # Real benchmark data
│   │   ├── samples/*.xy              # 8 XRD patterns
│   │   ├── weight_fractions.csv      # Ground truth
│   │   └── *_weight_fractions.csv    # Expert results by method
│   └── analysis.ipynb                # Main evaluation notebook
├── scripts/
│   ├── training/
│   │   ├── config_training.py       # Training configuration
│   │   └── main_training.py         # Training script
│   └── inference/
│       ├── config_inference.py      # Inference configuration
│       └── infer.py                 # Inference script
├── src/
│   ├── data_loading/                # PyTorch DataLoaders
│   ├── models/
│   │   ├── MLP.py                   # MLP variants (MLP, MLP2-7)
│   │   └── smallFCN.py             # Fully convolutional network
│   ├── training/                    # Training utilities
│   ├── inference/                   # Inference utilities
│   └── utils/                       # Helper functions
├── TOPAS_sim/                       # TOPAS input file generation scripts
└── using_Monash_HPC/               # SLURM scripts for HPC
```

## Getting Started

### Prerequisites

```bash
pip install torch numpy pandas matplotlib seaborn scipy
```

See requirements based on imports in `analysis/analysis.ipynb`.

### Understanding the Code

**Start here:**

1. **Read `analysis/analysis.ipynb`** - This notebook shows the complete workflow:
   - Loading trained models
   - Evaluating on test data
   - Comparing predictions on Madsen benchmark samples
   - Generating all visualizations

2. **Check model architectures** in `src/models/`:
   - `MLP.py`: Multiple MLP variants tested (MLP, MLP2, MLP3, MLP4, MLP5, MLP7)
   - `smallFCN.py`: Fully convolutional network (performed worse)
   - Others were deleted. MLPs are sufficient to scale to harder tasks. The more important thing is creating pipelines to run those harder tasks.

3. **Review configuration files** before training:
   - `scripts/training/config_training.py`: Hyperparameters, data paths, model selection, training parameters
   - `scripts/inference/config_inference.py`: Inference settings

## Training Data

**The training data is NOT included in this repository** due to size (~72k samples). You will need to generate your own.

### Current Method (TOPAS-based)

The `TOPAS_sim/` directory contains scripts to generate TOPAS `.inp` files. These can be run through TOPAS to create synthetic XRD patterns.

**Process:**
1. Scripts generate `.inp` files with randomized phase fractions and parameters
2. Run each `.inp` file through TOPAS (manual or scripted)
3. TOPAS outputs `.xy` files with XRD patterns
4. Process into training format (see Expected Format below)

**Problems with this approach:**
- Requires TOPAS license (commercial software)
- Manual/slow generation process
- Not easily reproducible
- Hard to scale

### Recommended Alternative: Python-Based Simulation

**You should explore Python-based XRD simulation instead.** Options include:

1. **GSAS-II** - Free Rietveld software with Python scripting
2. **pymatgen** - Materials Project's Python library with diffraction module
3. **Custom kinematic simulation** - Fast forward model for XRD patterns

Benefits:
- No licensing barriers
- Fully automated data generation
- Easy to integrate with training pipeline
- Reproducible

### Expected Data Format

To integrate with this repo, training data should be organized as:

```
training_data/
├── processed_data_p1/
│   ├── intensities.npy          # Shape: (N_samples, 7250)
│   ├── weight_fractions.csv     # Columns: [Corundum, Fluorite, Zincite]
│   ├── all_params.csv           # Additional parameters (optional)
│   └── metadata.json            # Metadata about generation
```

**intensities.npy:**
- NumPy array of shape `(N_samples, 7250)`
- Each row is one XRD pattern
- Values are intensities from 5° to 150° 2θ in 0.02° steps
- Normalize to max intensity = 100

**weight_fractions.csv:**
- Columns: `Corundum`, `Fluorite`, `Zincite`
- Values: weight fractions [0, 1] that sum to 1
- One row per sample (must match intensities.npy row count)

**all_params.csv (optional):**
- Additional parameters varied during simulation (crystallite size, strain, etc.)
- Not used by current models but logged for analysis

**metadata.json:**
- Information about simulation parameters
- Not used by training code but helpful for documentation

### Training Data Distribution

[TODO: Ternary diagram showing compositional coverage]

Training data should cover the full ternary compositional space uniformly.

### Example Pattern

[TODO GRAPH: Single XRD pattern with composition]

## Training a Model

```bash
cd scripts/training
# IMPORTANT: Edit config_training.py first
python main_training.py
```

**Before training, edit `config_training.py`:**
- `data_dirs`: Point to your processed data directories
- `model_type`: Choose which model to train (MLP, MLP2-7, smallFCN)
- Hyperparameters: learning rate, batch size, epochs, etc.
- Paths: where to save trained models
- Device: CPU vs GPU settings

The config file controls ALL aspects of training, not just hyperparameters.

## Evaluation

Run `analysis/analysis.ipynb` to:
1. Load a trained model
2. Evaluate on holdout test set
3. Generate prediction vs actual plots
4. Run inference on and compare against Madsen benchmark samples

The notebook shows the complete evaluation pipeline used to generate results in this README.

## Model Details

Multiple architectures tested. Best performer: **MLP7**

```python
# From src/models/MLP.py
Input: 7,250 intensity values (flattened)
Hidden: [4096, 2048, 1024, 512, 256] with dropout
Output: 3 weight fractions
Loss: Mean Absolute Error (MAE)
Optimizer: AdamW
```

Simple deep MLPs consistently outperformed more complex architectures.

## Notes on future usage.

### 1. TOPAS Dependency
Current data generation requires commercial TOPAS license. This:
- Blocks reproducibility
- Makes data generation slow and manual
- Creates licensing barriers for future work

**Replacing with Python-based simulation is high priority.**

### 2. Three-Phase System Only
Model is completely specialized to Corundum-Fluorite-Zincite. Cannot:
- Handle different phases
- Work with 2, 4, or more phases
- Generalize to other materials systems

### 3. No Uncertainty Quantification
Model outputs point predictions without confidence intervals. Rietveld refinement provides error estimates; ML should too.

Preliminary exploration attempted (ensemble methods, dropout) but results were unreliable. Needs proper development.


## Future Work. I would suggest placing your focus in this order:

### Python-Based Data Generation

**Replace TOPAS to enable:**
- Fully automated data generation
- No licensing barriers
- Easy integration with training
- Reproducible research

**Recommended approach:**

1. **Start with GSAS-II:**
   - Free, mature, Python-scriptable
   - Can generate realistic XRD patterns
   - Good documentation

2. **Generate validation set:**
   - Create small dataset (~1000 patterns) with both GSAS-II and TOPAS
   - Compare patterns to ensure quality
   - Validate that ML model trains equivalently

3. **Full replacement:**
   - Generate full training set with validated Python pipeline
   - Document generation parameters
   - Make fully reproducible

4. **Possible optimization:**
   - GSAS-II + custom noise models for speed
   - Parallel generation on HPC

### Multi-Phase Generalization

**The three-phase limitation must be addressed.** Approaches to explore:

**Option A: Universal Model**
- Train on many different phase combinations
- Model learns general XRD → composition mapping
- Challenge: Enormous training data requirements
- Challenge: Output dimensionality (how to handle variable phase count?)

**Option B: Phase-System-Specific Models**
- Separate model for each common phase assemblage
- Example: One model for cement phases, another for mineralogy phases
- More practical for near-term deployment
- Start with 4-5 phase systems

**Practical next steps:**
1. Identify target application (cement, minerals, pharmaceuticals)
2. Select 4-5 common phases in that domain
3. Generate training data across full compositional space
4. Train and validate following current methodology

### Uncertainty Quantification

Expert analysis provides error bars. ML predictions need them too.

**Approaches to implement:**

**Ensemble Methods:**
- Train 5-10 models with different initializations
- Use prediction spread as uncertainty estimate
- Computationally expensive but straightforward

**Monte Carlo Dropout:**
- Keep dropout active during inference
- Sample multiple predictions per input
- Use variance as uncertainty
- Fast but needs careful tuning

**Bayesian Neural Networks:**
- Model weight distributions rather than point estimates
- Principled uncertainty but computationally expensive
- May be overkill for this application

**Conformal Prediction:**
- Distribution-free uncertainty intervals
- No model changes required
- Worth exploring

**Validation is critical:** Any uncertainty method must be calibrated against real expert error estimates from Madsen data.

## Development Notes

**Code Quality:**
- Research-quality code: expect hardcoded paths etc.
- Refactor as needed for your use case
- Add logging, tests, documentation as you go

**Data Requirements:**
The current training used ~50k+ samples. Based on model performance, you likely need:
- Minimum: ~10-20k samples for reasonable performance
- Recommended: 50k+ for best results
- More phases will require significantly more data
