import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

import importlib
from tqdm import tqdm
import wandb
import os
import time

from data_pipeline.loader import load_dataset_tsl, load_dataset_benchmarks
from experiments.get_models import get_model
from experiments.sweep_config import sweep_configuration
from experiments.utils import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

# trainer for torch TSL datasets
def train(config, model, train_loader, val_loader):
    epochs = config.epochs
    criterion = nn.MSELoss()
    eval_criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    verbose = config.verbose
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=config.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.lr_min)
    
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_epochs])
    
    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_delta, verbose=verbose)
    
    avg_epoch_runtime = 0.0
    
    # Training loop
    print()
    print("Training Model...\n")

    for epoch in range(epochs):
        model.train()        
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        val_mae_epoch_loss = 0.0
        
        epoch_start_time = time.time()
        for batch in train_loader:
            x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
            edge_weight = batch.edge_weight.to(device) if batch.edge_weight is not None else None
            
            y_hat = model(x, edge_index, edge_weight)
            loss = criterion(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        scheduler.step()
        epoch_end_time = time.time()
        
        avg_epoch_runtime += (epoch_end_time - epoch_start_time)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
                edge_weight = batch.edge_weight.to(device) if batch.edge_weight is not None else None
                
                y_hat = model(x, edge_index, edge_weight)
                val_epoch_loss += criterion(y_hat, y).item()
                
                val_mae_epoch_loss += eval_criterion(y_hat, y).item()
        
        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        avg_mae_epoch_loss = val_mae_epoch_loss / len(val_loader)
        
        if config.early_stopping:
            early_stopping(avg_val_epoch_loss, model)
            if early_stopping.early_stop:
                if verbose:
                    print("Early stopping triggered at epoch: ", epoch + 1)
                break
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss (MSE): {avg_train_epoch_loss:.4f} - Val Loss (MSE): {avg_val_epoch_loss:.4f}")
            
        if epoch > 4:
            # wandb.log({"train_loss": avg_train_epoch_loss, "val_loss": avg_val_epoch_loss, "learning rate": scheduler.get_last_lr()[0]})
            wandb.log({f"train_loss (mse)": avg_train_epoch_loss, f"val_loss (mse)": avg_val_epoch_loss, f'val_loss (mae)': avg_mae_epoch_loss, f'runtime_epoch (sec)': (epoch_end_time - epoch_start_time)})
    
    avg_epoch_runtime /= epochs
    
    print(f"Average runtime per epoch: {avg_epoch_runtime:.2f} seconds")
    wandb.log({f"avg_epoch_runtime (sec)": avg_epoch_runtime})
    
    return model

def test(config, model, test_loader):
    eval_criterion = nn.L1Loss()
    # Test loop
    total_test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, edge_index, y = batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
            edge_weight = batch.edge_weight.to(device) if batch.edge_weight is not None else None
            
            y_hat = model(x, edge_index, edge_weight)
            total_test_loss += eval_criterion(y_hat, y).item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    
    return avg_test_loss

"""
wandb hyperparameter sweep
- train_sweep()
- tuner() <- entrypoint
"""
def train_sweep():
    wandb.init()
    
    config_path = f'./configs/metrLA.py'
    config = get_config(config_path)
    
    config.lr = wandb.config.learning_rate
    config.epochs = wandb.config.epochs
    config.num_heads = wandb.config.num_heads
    config.dstan_hidden = wandb.config.hidden
    config.attention_dropout = wandb.config.attention_dropout
    
    data = load_dataset_tsl(config)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
    model = MP_DSTAN(
        config=config,
        input_size=config.num_channels,
        hidden_size=config.dstan_hidden,
        window_size=config.window,
        horizon=config.horizon,
        num_heads=config.num_heads,
        dropout=config.attention_dropout
    ).to(device)
    
    start_time = time.time()
    trained_model = train(config, model, train_loader, val_loader, verbose=config.verbose)
    end_time = time.time()
    
    test(config, trained_model, test_loader)
    
    if config.time_verbose:
        print(f"Total time taken for trainer execution: {((end_time - start_time)/60.0):.2f} minutes\n")
        wandb.log({"total_runtime (min)": ((end_time - start_time)/60.0)})

def tuner(config_name, model_name):
    # sweep_id = wandb.sweep(
    #     sweep=sweep_configuration,
    #     project="DSTAN"
    # )
    sweep_id = "humingamelab/DSTAN/c9byzf2c"
    
    wandb.agent(sweep_id, function=train_sweep, count=10)

def driver(config_name, model_name, wandb_mode='disabled'):
    config_path = f'./configs/{config_name}.py'
    config = get_config(config_path)

    wandb_mode = 'online' if wandb_mode == 'e' or wandb_mode == 'enabled' else 'disabled'
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    wandb_name = f'{SLURM_JOB_ID}-{model_name}'
    wandb.init(
        project="DSTAN",
        name=wandb_name,
        mode=wandb_mode,
        config={
            "config": config.__dict__,
            "dataset": config.dataset,
        },
    )
    

    data = load_dataset_tsl(config)
    
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    
    losses = []
    for run in range(config.runs):
        model = get_model(config, model_name).to(device)
    
        start_time = time.time()
        trained_model = train(config, model, train_loader, val_loader)
        end_time = time.time()
        
        test_loss = test(config, trained_model, test_loader)
        losses.append(test_loss)
        
        print()
        print(f"Test Loss: {test_loss:.4f}")    
        wandb.log({f"test_loss (mae)": test_loss})

    avg_test_loss = np.mean(losses)
    std_test_loss = np.std(losses)
    
    print(f"Average Test Loss over {config.runs} runs: {avg_test_loss:.4f}")
    print(f"Std Test Loss over {config.runs} runs: {std_test_loss:.4f}")
    wandb.log({f"avg_test_loss (mae)": avg_test_loss, f"std_test_loss": std_test_loss})