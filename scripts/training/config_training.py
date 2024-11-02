import os

import torch.nn as nn
import torch.optim as optim

# Import models
from src.models.smallFCN import smallFCN
from src.models.MLP import MLP

# TODO: Is model class necessary?
# TODO: Put train file with model file. Or are you going to use it across models? Decide.

# Paths (Data should point to processed data)
DATA_DIR = 'training_data/processed_data_3'
MODEL_SAVE_DIR = 'trained_models'

# Model Setup
MODEL_TYPE = "MLP"            # Options: Any of the imported models. It should be a string. e.g. "smallFCN", "MLP"

# IF SINGLE TASK, loss
CRITERION_TYPE = "MSELoss"  # Options: "CrossEntropyLoss", "MSELoss", "KLDivLoss"

# Hyper Params
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
NUM_EPOCHS = 400

# Optimiser
OPTIMIZER_TYPE = "Adam" # Options: "Adam", "SGD"

# Data Loading Settings
NUM_WORKERS = 6

# WandB configuration (Note that there is already a basic WandB log in train.py)
USE_WANDB = True        # Set to False if you don't want to use WandB at all.
WANDB_PROJECT_NAME = "Phase_Analysis_1"
WANDB_SAVE_DIR = "/wandb"
SAVE_MODEL_TO_WANDB_SERVERS = False
WANDB_LOG_ARCHITECTURE = False

###########################################################################################
############# DON'T TOUCH - These classes contain the options for above ##################
MODEL_CLASS = {
    "smallFCN": smallFCN,
    "MLP": MLP
}
CRITERION_CLASS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "KLDivLoss": lambda: nn.KLDivLoss(reduction='batchmean')
}
OPTIMIZER_CLASS = {
    "Adam": optim.Adam,
    "SGD": optim.SGD
}