import os
import torch
import pickle

# Import config
import scripts.inference.config_inference as config_inference

# Import functions
from src.data_loading.simXRD_data_loader import create_inference_data_loader

# TODO: Add a function to save data dynamically.
# TODO: Note that this only uses one GPU.

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = config_inference.MODEL_CLASS[config_inference.MODEL_TYPE]
    model = model_class()
    
    # Load the state dict to the appropriate device
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model.to(device), device

def run_inference(model, test_loader, device):
    model.eval()
    all_predictions = {task: [] for task in config_inference.TASKS}
    all_labels = {task: [] for task in config_inference.TASKS}

    with torch.no_grad():
        for batch in test_loader:
            intensity, spg, crysystem, blt, composition = [t.to(device) for t in batch]
            
            if config_inference.MULTI_TASK:
                outputs = model(intensity.unsqueeze(1))
                
                for task in config_inference.TASKS:
                    preds = outputs[task].argmax(dim=1)
                    all_predictions[task].extend(preds.cpu().numpy())
                    
                all_labels['spg'].extend(spg.cpu().numpy())
                all_labels['crysystem'].extend(crysystem.cpu().numpy())
                all_labels['blt'].extend(blt.cpu().numpy())
                all_labels['composition'].extend(composition.cpu().numpy())
            else:
                output = model(intensity.unsqueeze(1))
                preds = output.argmax(dim=1)
                all_predictions['spg'].extend(preds.cpu().numpy())
                all_labels['spg'].extend(spg.cpu().numpy())

    return all_predictions, all_labels

def save_results(results, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")

def main(model_path):

    # Setup device
    model, device = load_model(model_path)

    # Create data loader for test data
    test_loader = create_inference_data_loader(
        config_inference.INFERENCE_DATA, 
        config_inference.BATCH_SIZE, 
        config_inference.NUM_WORKERS
    )

    # Run inference
    predictions, labels = run_inference(model, test_loader, device)

    # Save results
    results = {
        'predictions': predictions,
        'labels': labels
    }

    # TODO: ALTERABLE SAVE NAME
    # Save results
    save_dir = config_inference.INFERENCE_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(config_inference.INFERENCE_SAVE_DIR, 'TEST_full.pkl')
    save_results(results, save_path)

    return
    
if __name__ == "__main__":
    model_path = f'{config_inference.MODEL_SAVE_DIR}/{config_inference.MODEL_NAME}'
    main(model_path)