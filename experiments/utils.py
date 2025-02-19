import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, checkpoint_path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait for an improvement before stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints messages for each improvement.
            checkpoint_path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        """Checks if validation loss has improved and saves model checkpoint if so."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter if improvement is seen
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased, saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
