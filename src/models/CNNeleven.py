from torch import nn
import torch.nn.functional as F

# Model from:
# https://github.com/compasszzn/XRDBench/blob/main/model/CNN11.py
# Accessed 2/8/24
# As described in simXRD paper.

# Mofidications:
# A basic augmentation to make CNNeleven a multi-task learner in CNNeleven_MultiTask.
# 

class CNNeleven(nn.Module):
    def __init__(self):
        super(CNNeleven, self).__init__()
        self.cnn = NoPoolCNN()
        mlp_in_features = 12160
        self.MLP = Predictor(mlp_in_features, 230)
        
    def forward(self, x):
        x = F.interpolate(x,size=8500,mode='linear', align_corners=False)
        x = self.cnn(x)
        x = self.MLP(x)
        return x
    
class CNNeleven_MultiTask(nn.Module):
    def __init__(self):
        super(CNNeleven_MultiTask, self).__init__()

        self.cnn = NoPoolCNN()

        mlp_in_features = 12160

        self.MLP_spg_out = Predictor(mlp_in_features, 230)
        self.MLP_crysystem_out = Predictor(mlp_in_features, 7)
        self.MLP_blt_out = Predictor(mlp_in_features, 7)
        self.MLP_composition_out = Predictor(mlp_in_features, 118)
        
    def forward(self, x):
        x = F.interpolate(x,size=8500,mode='linear', align_corners=False)
        x = self.cnn(x)

        spg_out = self.MLP_spg_out(x)
        crysystem_out = self.MLP_crysystem_out(x)
        blt_out = self.MLP_blt_out(x)
        composition_out = self.MLP_composition_out(x)

        return {
            'spg': spg_out,
            'crysystem': crysystem_out,
            'blt': blt_out,
            'composition': composition_out
        }
    
# Classes  
class NoPoolCNN(nn.Module):
    def __init__(self):
        super(NoPoolCNN, self).__init__()
        self.CNN = nn.Sequential(
                nn.Conv1d(1, 80, 100, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 50, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 25, 2),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

    def forward(self, x):
        return self.CNN(x)

class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, out_features))

    def forward(self, x):
        return self.MLP(x)