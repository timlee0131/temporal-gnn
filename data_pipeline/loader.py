from tsl.datasets import MetrLA, PemsBay
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler

def load_dataset_tsl(config):
    if config.dataset == 'metrLA':
        dataset = MetrLA(root=config.data_dir)
    elif config.dataset == 'pemsbay':
        dataset = PemsBay(root=config.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    connectivity = dataset.get_connectivity(
        threshold=config.connectivity_threshold,
        include_self=config.include_self,
        normalize_axis=config.normalize_axis,
        layout=config.layout
    )
    
    # torch-ifying the dataset
    torch_dataset = SpatioTemporalDataset(
        target = dataset.dataframe(),
        connectivity = connectivity,
        mask = dataset.mask,
        horizon=config.horizon,
        window=config.window,
        stride=config.stride,
    )
    
    scalers = {'target': StandardScaler(axis=(0,1))}
    splitter = TemporalSplitter(val_len=config.val_ratio, test_len=config.test_ratio)
    
    data_module = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=config.batch_size
    )
    data_module.setup()
    
    return data_module

def load_dataset_custom(config):
    # TODO: implement this function for loading in custom data (from torch pickle)
    pass