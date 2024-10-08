import os

import torch.nn as nn
import torch.optim as optim

# Import models
from src.models.CNNten import CNNten, CNNten_MultiTask, smallCNNten_MultiTask
from src.models.CNNeleven import CNNeleven, CNNeleven_MultiTask
from src.models.smallFCN import smallFCN, smallFCN_MultiTask, smallFCN_SelfAttention_MultiTask, experimentalFCN
from src.models.ViT import ViT1D_MultiTask
from src.models.MLPten import MLPten

# TODO: Is model class necessary?

# Paths
DATA_DIR = 'training_data/simXRD_partial_data'
MODEL_SAVE_DIR = 'trained_models'

# Data
TRAIN_DATA = os.path.join(DATA_DIR, 'train.db')
VAL_DATA = os.path.join(DATA_DIR, 'val.db')
TEST_DATA = os.path.join(DATA_DIR, 'test.db')

# Model Setup
MODEL_TYPE = "CNNeleven_MultiTask"                 # Options: Any of the imported models. It should be a string. e.g. "smallFCN"
MULTI_TASK = True                                  # Set to True for multi-task learning (points train function to train_multi_spg_cryssystem_blt_element.py)

# IF SINGLE TASK, loss
CRITERION_TYPE = "CrossEntropyLoss"  # Options: "CrossEntropyLoss", "MSELoss"

# IF MULTI-TASK, loss
MULTI_TASK_CRITERIA = {
    'spg': nn.CrossEntropyLoss(),
    'crysystem': nn.CrossEntropyLoss(),
    'blt': nn.CrossEntropyLoss(),
    'composition': nn.BCEWithLogitsLoss()
}

# Hyper Params
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 100

# Optimiser
OPTIMIZER_TYPE = "Adam" # Options: "Adam", "SGD"

# Data Loading Settings
NUM_WORKERS = 6

# WandB configuration (Note that there is already a basic WandB log in train.py)
USE_WANDB = True        # Set to False if you don't want to use WandB at all.
WANDB_PROJECT_NAME = "FirstModelExperiments"
WANDB_SAVE_DIR = "/wandb"
SAVE_MODEL_TO_WANDB_SERVERS = False
WANDB_LOG_ARCHITECTURE = False

###########################################################################################
############# DON'T TOUCH - CThese classes contain the options for above ##################
MODEL_CLASS = {
    "CNNten": CNNten,
    "CNNten_MultiTask": CNNten_MultiTask,
    "CNNeleven_MultiTask": CNNeleven_MultiTask,
    "smallCNNten_MultiTask": smallCNNten_MultiTask,
    "smallFCN": smallFCN,
    "smallFCN_MultiTask": smallFCN_MultiTask,
    "smallFCN_SelfAttention_MultiTask": smallFCN_SelfAttention_MultiTask,
    "experimentalFCN": experimentalFCN,
    "MLPten": MLPten,
    "ViT1D_MultiTask": ViT1D_MultiTask
}
CRITERION_CLASS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss
}
OPTIMIZER_CLASS = {
    "Adam": optim.Adam,
    "SGD": optim.SGD
}