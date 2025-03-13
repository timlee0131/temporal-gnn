from tsl.datasets import MetrLA, PemsBay
from tsl.datasets import ElectricityBenchmark, ExchangeBenchmark, SolarBenchmark, TrafficBenchmark
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler

from data_pipeline.utils import create_ramanujan_expander

def load_dataset_tsl(config):
    connectivity = None
    if config.dataset == 'metrLA':
        dataset = MetrLA(root=config.data_dir)
        
        connectivity = dataset.get_connectivity(
            threshold=config.connectivity_threshold,
            include_self=config.include_self,
            normalize_axis=config.normalize_axis,
            layout=config.layout
        )
    elif config.dataset == 'pemsbay':
        dataset = PemsBay(root=config.data_dir)
     
        connectivity = dataset.get_connectivity(
            threshold=config.connectivity_threshold,
            include_self=config.include_self,
            normalize_axis=config.normalize_axis,
            layout=config.layout
        )
    
    # loading benchmark datasets
    elif config.dataset == 'electricity':
        dataset = ElectricityBenchmark(root=config.data_dir)
        edge_index = create_ramanujan_expander(num_nodes=len(dataset.nodes), d=config.d_ramanujan)
        connectivity = (edge_index, None)
    elif config.dataset == 'exchange':
        dataset = ExchangeBenchmark(root=config.data_dir)
        edge_index = create_ramanujan_expander(num_nodes=len(dataset.nodes), d=config.d_ramanujan)
        connectivity = (edge_index, None)
    elif config.dataset == 'solar':
        dataset = SolarBenchmark(root=config.data_dir)
        edge_index = create_ramanujan_expander(num_nodes=len(dataset.nodes), d=config.d_ramanujan)
        connectivity = (edge_index, None)
    elif config.dataset == 'traffic':
        dataset = TrafficBenchmark(root=config.data_dir)
        edge_index = create_ramanujan_expander(num_nodes=len(dataset.nodes), d=config.d_ramanujan)
        connectivity = (edge_index, None)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    
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

def load_dataset_benchmarks(config):
    if config.dataset != 'benchmarks':
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    benchmark_electricity = ElectricityBenchmark(root=f'{config.data_dir}/electricity')
    benchmark_exchange = ExchangeBenchmark(root=f'{config.data_dir}/exchange')
    benchmark_solar = SolarBenchmark(root=f'{config.data_dir}/solar')
    benchmark_traffic = TrafficBenchmark(root=f'{config.data_dir}/traffic')

    benchmarks = [benchmark_electricity, benchmark_exchange, benchmark_solar, benchmark_traffic]
    benchmark_loader_dict = {}

    for benchmark in benchmarks:
        print(f"Processing {benchmark.name}...")
        
        edge_index = create_ramanujan_expander(num_nodes=len(benchmark.nodes), d=config.d_ramanujan)

        # subclass of torch.utils.data.Dataset
        torch_dataset = SpatioTemporalDataset(
            target=benchmark.dataframe(),
            connectivity=(edge_index, None),
            mask=benchmark.mask,
            horizon=config.horizon,
            window=config.window,
            stride=config.stride,
        )

        scalers = {'target': StandardScaler(axis=(0, 1))}

        # Split data sequentially:
        #   |------------ dataset -----------|
        #   |--- train ---|- val -|-- test --|
        splitter = TemporalSplitter(val_len=config.val_ratio, test_len=config.test_ratio)

        dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=splitter,
            batch_size=config.batch_size,
        )

        dm.setup()

        # train_loader = dm.train_dataloader()
        # val_loader = dm.val_dataloader()
        # test_loader = dm.test_dataloader()
        
        # benchmark_loader_dict[benchmark.name] = {
        #     'train': train_loader,
        #     'val': val_loader,
        #     'test': test_loader
        # }
        
        benchmark_loader_dict[benchmark.name] = dm
    
    return benchmark_loader_dict

def load_dataset_custom(config):
    # TODO: implement this function for loading in custom data (from torch pickle)
    pass
