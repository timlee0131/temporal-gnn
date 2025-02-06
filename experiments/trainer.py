import numpy as np
import torch
import torch.nn as nn

import importlib
from tqdm import tqdm
import time

from data_pipeline.loader import load_dataset_tsl
from experiments.get_models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.get_config()

# trainer for torch TSL datasets
def train(config, model, train_loader, val_loader, verbose=False):
    epochs = config.epochs
    criterion = nn.MSELoss()
    eval_criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Training loop
    print()
    print("Training Model...\n")

    for epoch in range(epochs):
        model.train()        
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        
        for batch in train_loader:            
            x, edge_index, edge_weight, y = batch.x.to(device), batch.edge_index.to(device), batch.edge_weight.to(device), batch.y.to(device)
            
            y_hat = model(x, edge_index, edge_weight)
            loss = criterion(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, edge_weight, y = batch.x.to(device), batch.edge_index.to(device), batch.edge_weight.to(device), batch.y.to(device)
                
                y_hat = model(x, edge_index, edge_weight)
                val_epoch_loss += criterion(y_hat, y).item()
        
        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss (MSE): {avg_train_epoch_loss:.4f} - Val Loss (MSE): {avg_val_epoch_loss:.4f}")
    
    return model

def test(config, model, test_loader):
    eval_criterion = nn.L1Loss()
    # Test loop
    total_test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, edge_index, edge_weight, y = batch.x.to(device), batch.edge_index.to(device), batch.edge_weight.to(device), batch.y.to(device)
            
            y_hat = model(x, edge_index, edge_weight)
            total_test_loss += eval_criterion(y_hat, y).item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    print()
    print(f"Test Loss: {avg_test_loss:.4f}")    
    
    print()
    print("Model Run Complete...\n")

def driver(config_name, model_name):
    config_path = f'./configs/{config_name}.py'
    config = get_config(config_path)
    
    data = load_dataset_tsl(config)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
    model = get_model(config, model_name).to(device)
    
    # print the config description
    print(config.description)
    
    start_time = time.time()
    trained_model = train(config, model, train_loader, val_loader, verbose=config.verbose)
    end_time = time.time()
    
    test(config, trained_model, test_loader)
    
    if config.time_verbose:
        print(f"Total time taken for trainer execution: {((end_time - start_time)/60.0):.2f} minutes\n")