# Import models
#from src.models.CNNten import CNNten, CNNten_MultiTask, smallCNNten_MultiTask
#from src.models.MLPten import MLPten
#from src.models.CNNeleven import CNNeleven, CNNeleven_MultiTask
from src.models.smallFCN import smallFCN #, smallFCN_MultiTask, smallFCN_SelfAttention_MultiTask, experimentalFCN
#from src.models.ViT import ViT1D_MultiTask

def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(f"{name}: {param}")
        total_params += param
    print(f"Total trainable params: {total_params}")

if __name__ == "__main__":
    model = smallFCN()
    print_model_parameters(model)

    #model = smallFCN_multi_task()
    #print_model_parameters(model)