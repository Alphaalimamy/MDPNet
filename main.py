import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader



from utils import  *
from train import *
from model import MDPNet
from loaders import create_dataloaders, load_datasets

# Hyperparameters
BATCH_SIZE = 4  # 8 16
EPOCHS = 200
LEARNING_RATE = 0.0001
IMAGE_SIZE = 256




# Create DataLoaders
def create_dataloaders(train_dataset, val_dataset, test_dataset):
    num_workers = torch.cuda.device_count() * 4 if device == "cuda" else 0
    print(f"num_workers:{num_workers}")
          
    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers,
                                pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=num_workers,
                                 pin_memory=False,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader



# Define the early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "mpdnet_kvasir_best_model.pth")
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    
                    
                    
                

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeding(42)
    create_directory("files")

    """ Training logfile """
    train_log_path = "train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("train_log.txt", "w")
        train_log.write("\n")
        train_log.close()
        
    
    train_dataset, val_dataset, test_dataset = load_datasets()
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    
    model = MDPNet(1).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCEWithLogitsLoss()
    
    data_str = f"BATCH SIZE: {BATCH_SIZE} - EPOCH: {EPOCHS} - LEARNING_RATE: {LEARNING_RATE}, OPTIMIZER: AdamW"
    print_and_save(train_log_path, data_str)

    data_str = f"train_dataloader: {len(train_dataloader)} - val_dataloader: {len(val_dataloader)} test_dataloader: {len(test_dataloader)}"
    print_and_save(train_log_path, data_str)
    
    train_metrics, val_metrics = [], []
    epochs = range(EPOCHS)
    early_stopping = EarlyStopping(patience=5, verbose=True)  # Initialize early stopping
    

    
    best_valid_loss = float('inf')
    for epoch in epochs:
        train_result = train(model, train_dataloader, optimizer, criterion)
        val_result = evaluate(model, val_dataloader, criterion)
        scheduler.step(val_result[0])
        
        
        train_metrics.append(train_result)
        val_metrics.append(val_result)
        train_data = f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {train_result[0]:.4f},Train DICE: {train_result[1]:.4f},Train IoU: {train_result[2]:.4f},Train Precision: {train_result[3]:.4f}, Train Accuracy: {train_result[4]:.4f}"
       
        print_and_save(train_log_path, train_data)
        
        
        validation_data = f"Val Loss: {val_result[0]:.4f}, Val DICE: {val_result[1]:.4f}, Val IoU: {val_result[2]:.4f}, Val Precision: {val_result[3]:.4f},Val Accuracy: {val_result[4]:.4f}"
            
        print_and_save(train_log_path, validation_data)
        # Check for early stopping
        
        early_stopping(val_result[0], model)  # Use validation loss for early stopping

        if early_stopping.early_stop:
            print("Early stopping initiated.")
            break
        
        
        
    # Load the best model before testing
    model.load_state_dict(torch.load("mpdnet_kvasir_best_model.pth", weights_only=True))
    plot_metrics(np.array(train_metrics).T, np.array(val_metrics).T)

    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    test_result = evaluate(model, test_dataloader, criterion)
    data_str = f"Test Loss: {test_result[0]:.4f}, Test DICE: {test_result[1]:.4f}, Test IoU: {test_result[2]:.4f}, Test Precision: {test_result[3]:.4f},Test Accuracy: {test_result[4]:.4f}"
    print_and_save(train_log_path, data_str)
    
if __name__ == "__main__":
    # Call the function to set the seed
    main()