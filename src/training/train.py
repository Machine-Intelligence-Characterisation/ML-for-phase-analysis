import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import numpy as np

# Config
import scripts.training.config_training as config_training

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    best_val_loss = float('inf')

    # Gradient check setup
    def hook_fn(grad):
        print(f"Gradient norm: {grad.norm().item():.9f}")

    # Register hook for the first convolutional layer
    first_conv_layer = next(layer for layer in model.modules() if isinstance(layer, torch.nn.Conv1d))
    first_conv_layer.weight.register_hook(hook_fn)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            # Move data to device
            data = data.unsqueeze(1).to(device)  # Add channel dimension
            target = target.to(device)
            
            # Print input and target information for the first batch of the first epoch
            if epoch == 0 and batch_idx == 0:
                print("Input data shape:", data.shape)
                print("Sample input:", data[0].cpu().numpy())
                print("Max input value:", data.max().item())  # Add this line
                print("Target data shape:", target.shape)
                print("Sample target:", target[0].cpu().numpy())

            optimizer.zero_grad()
            output = model(data)
            
            # Print raw model output for the first batch of the first epoch
            if epoch == 0 and batch_idx == 0:
                print("Raw model output:", output[0].detach().cpu().numpy())

            # Convert model output to log probabilities
            log_output = torch.log(output + 1e-8)  # Add small constant to avoid log(0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate on Val
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log metrics to wandb every epoch
        if config_training.USE_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics['mae'],
                "val_mse": val_metrics['mse'],
                "val_r2": val_metrics['r2']
            })
        
        print(f'Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val MAE: {val_metrics["mae"]:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on the test set
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f'Test loss: {test_loss:.4f}, Test MAE: {test_metrics["mae"]:.4f}, Test R2: {test_metrics["r2"]:.4f}')

    if config_training.USE_WANDB:
        wandb.log({
            "test_loss": test_loss,
            "test_mae": test_metrics['mae'],
            "test_mse": test_metrics['mse'],
            "test_r2": test_metrics['r2']
        })

    return model, test_loss, test_metrics

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data, target, _ in tqdm(data_loader, desc="Evaluation"):
            data = data.unsqueeze(1).to(device)  # Add channel dimension
            target = target.to(device)
            
            output = model(data)
            # Convert model output to log probabilities
            log_output = torch.log(output + 1e-8)  # Add small constant to avoid log(0)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            all_targets.append(target.cpu().numpy())
            all_outputs.append(output.cpu().numpy())
    
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    avg_loss = total_loss / len(data_loader)
    mae = np.mean(np.abs(all_targets - all_outputs))
    mse = np.mean((all_targets - all_outputs)**2)
    r2 = 1 - np.sum((all_targets - all_outputs)**2) / np.sum((all_targets - np.mean(all_targets))**2)
    
    return avg_loss, {'mae': mae, 'mse': mse, 'r2': r2}