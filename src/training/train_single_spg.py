import torch
import wandb
from tqdm import tqdm

# Config
import scripts.training.config_training as config_training

# TODO: Maybe add in function hyper param tuning?
# TODO: Save best model.
# TODO: Residual XRD analysis

def train_single_spg(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            
            # Unpack
            data, space_group = batch[0], batch[1]
            
            # Reshape data: [batch_size, 3501] -> [batch_size, 1, 3501]
            data = data.unsqueeze(1).to(device)
            target = space_group.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate on Val
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        # Log metrics to wandb every epoch
        if config_training.USE_WANDB:
            wandb.log({
                "train_spg_loss": train_loss,
                "val_spg_loss": val_loss,
                "val_spg_accuracy": val_accuracy
            })
        
        print(f'Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Finish with an evaluate on the test set
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    print(f'Test loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    if config_training.USE_WANDB:
        wandb.log({
            "test_spg_loss": test_loss,
            "test_spg_accuracy": test_accuracy
        })

    return model, test_loss, test_accuracy

# Used for both val and test data_sets
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):

            data, space_group = batch[0], batch[1]
            
            # Reshape data: [batch_size, 3501] -> [batch_size, 1, 3501]
            # Move to device
            data = data.unsqueeze(1).to(device)
            target = space_group.to(device)
            
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy