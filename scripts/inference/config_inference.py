import os

# Import models
from src.models.CNNten import CNNten, CNNten_multi_task
from src.models.smallFCN import smallFCN, smallFCN_multi_task
from src.models.MLPten import MLPten

# TODO: MAKE "infer.py"
# TODO: MAKE this config easier to navigate
# TODO: MAKE THE SAVE NAME ALTERABLE

# Paths
DATA_DIR = 'training_data/simXRD_full_data'
MODEL_SAVE_DIR = 'trained_models'
INFERENCE_SAVE_DIR = 'inference_data'

# Data
INFERENCE_DATA = os.path.join(DATA_DIR, 'test.db')

# Model settings
MODEL_TYPE = "smallFCN_multi_task"   # Options: "CNNten", CNNten_multi_task", "smallFCN", "smallFCN_multi_task"
MULTI_TASK = True                    # Note, you can use multi-task models on single task inference!
MODEL_NAME = "smallFCN_multi_task_spg_acc_95.30_20240730_215123.pth" # Copy the model name as a string. For example: "smallFCN_multi_task_spg_acc_94.6500_20240728_235958.pth"

# Inference settings
BATCH_SIZE = 32
NUM_WORKERS = 7

####################################################################################
############# DON'T TOUCH - Classes contain the options for above ##################
MODEL_CLASS = {
    "CNNten": CNNten,
    "CNNten_multi_task": CNNten_multi_task,
    "smallFCN": smallFCN,
    "smallFCN_multi_task": smallFCN_multi_task,
    "MLPten": MLPten
}

# Task settings
TASKS = ['spg', 'crysystem', 'blt', 'composition'] if MULTI_TASK else ['spg']

# Label information
LABELS = {
    'spg': list(range(230)),
    'crysystem': list(range(7)),
    'blt': ['P', 'I', 'F', 'A', 'B', 'C', 'R'],
    'composition': list(range(118))
}