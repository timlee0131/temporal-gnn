import numpy as np
import torch
import torch.nn as nn

import importlib
from tqdm import tqdm
import time

from models.models import TTS_RNN_GCN, TTS_TRF_GAT
from experiments.loader import load_dataset_tsl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.get_config()

# trainer for torch TSL datasets
def train_tsl(config, model, data, verbose=False):
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
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
            optimizer.zero_grad()
            
            x, edge_index, edge_weight, y = batch.x.to(device), batch.edge_index.to(device), batch.edge_weight.to(device), batch.y.to(device)
            
            y_hat = model(x, edge_index, edge_weight)
            loss = criterion(y_hat, y)
            
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
        
        print(f"Epoch {epoch + 1}/{epochs} - Average Train Loss (MSE): {avg_train_epoch_loss:.4f} - Average Val Loss (MSE): {avg_val_epoch_loss:.4f}")
        
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

# TODO: implement this function for training on custom datasets
def train_custom(config, model, data):
    pass

def driver(config_name):
    config_path = f'./experiments/configs/{config_name}.py'
    config = get_config(config_path)
    
    data = load_dataset_tsl(config, device)
    
    # print the config description
    print(config.description)
    
    tts_rnn_gcn = TTS_RNN_GCN(
        input_size=config.num_channels,
        n_nodes=config.num_nodes,
        horizon=config.horizon,
        hidden_size=config.hidden_dim,
        rnn_layers=config.rnn_layers
    ).to(device)
    
    tts_transformer = TTS_TRF_GAT(
        input_size=config.num_channels,
        n_nodes=config.num_nodes,
        window=config.window,
        horizon=config.horizon,
        hidden_size=config.trf_hidden_dim,
        n_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        ff_dropout=config.ff_dropout,
        n_layers=config.num_trf_layers
    ).to(device)
    
    start_time = time.time()
    train_tsl(config, tts_rnn_gcn, data, verbose=config.verbose)
    end_time = time.time()
    
    if config.time_verbose:
        print(f"Total time taken for trainer execution: {((end_time - start_time)/60.0):.2f} minutes\n")